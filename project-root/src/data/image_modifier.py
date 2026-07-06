"""
image_modifier.py
==================

Port of the logo-insertion logic from Section 3 of the notebook
(`paste_wm_logo_big`), extended with the blur / replace / crop variants
that scripts/accuracy_delta.py already expects to find in
outputs/inference/inference_{blur,replace,crop}.csv (per FR-5 in the SAD
document).

All methods operate on PIL Images and return PIL Images, so they can be
dropped straight into the CLIP preprocessing pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont


class ImageModifier:
    """Applies logo-insertion and logo-removal-style modifications to images.

    `logo_boxes.json` (in data/) records where the logo(s) were placed for
    each image, so the blur/replace/crop operations know which region to
    target without re-detecting anything.
    """

    def __init__(self, logo_boxes_path: Optional[Path] = None):
        self.logo = self._build_default_logo()
        self.logo_boxes = {}
        if logo_boxes_path is not None and Path(logo_boxes_path).exists():
            with open(logo_boxes_path, "r") as f:
                self.logo_boxes = json.load(f)

    # ------------------------------------------------------------------
    # Logo construction (verbatim port of the notebook's inline logo)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_default_logo(logo_w: int = 800, logo_h: int = 120) -> Image.Image:
        logo = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(logo)
        draw.rectangle([0, 0, logo_w - 1, logo_h - 1], fill=(0, 100, 0, 240))
        try:
            font_big = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42
            )
            font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22
            )
        except OSError:
            font_big = ImageFont.load_default()
            font_small = font_big
        draw.text((15, 10), "WASTE MANAGEMENT", fill=(255, 255, 255, 255), font=font_big)
        draw.text(
            (15, 65),
            "Recycling & Disposal Services",
            fill=(200, 255, 200, 220),
            font=font_small,
        )
        return logo

    # ------------------------------------------------------------------
    # Insertion (the "Clever Hans shortcut" injection used for detection)
    # ------------------------------------------------------------------
    def paste_logo(self, img: Image.Image) -> Image.Image:
        """Paste the logo bottom-left (large) and top-right (small) — matches
        `paste_wm_logo_big` from the notebook exactly."""
        img = img.convert("RGBA")
        w, h = img.size

        new_logo_width = int(w * 0.6)
        new_logo_height = int(self.logo.size[1] * (new_logo_width / self.logo.size[0]))
        new_logo = self.logo.resize((new_logo_width, new_logo_height))
        offset = 5
        img.paste(new_logo, (offset, h - new_logo.size[1] - offset), new_logo)

        top_logo_width = int(w * 0.4)
        top_logo_height = int(self.logo.size[1] * (top_logo_width / self.logo.size[0]))
        top_logo = self.logo.resize((top_logo_width, top_logo_height))
        img.paste(top_logo, (w - top_logo_width - offset, offset), top_logo)

        return img.convert("RGB")

    # ------------------------------------------------------------------
    # Removal / neutralisation variants (FR-5) — these are the "does the
    # shortcut disappear if we take the logo away" experiments used by
    # accuracy_delta.py's blur/replace/crop methods.
    # ------------------------------------------------------------------
    def logo_regions(self, img: Image.Image) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Returns (bottom_left_box, top_right_box) in the same geometry as
        paste_logo, so blur/replace/crop target exactly where the logo was
        (or would be) placed."""
        w, h = img.size
        offset = 5

        bl_w = int(w * 0.6)
        bl_h = int(self.logo.size[1] * (bl_w / self.logo.size[0]))
        bottom_left = (offset, h - bl_h - offset, offset + bl_w, h - offset)

        tr_w = int(w * 0.4)
        tr_h = int(self.logo.size[1] * (tr_w / self.logo.size[0]))
        top_right = (w - tr_w - offset, offset, w - offset, offset + tr_h)

        return bottom_left, top_right

    def blur_region(self, img: Image.Image, radius: int = 25) -> Image.Image:
        """Gaussian-blur the region(s) where the logo sits, leaving the rest
        of the image untouched. Used to test whether accuracy recovers once
        the logo is no longer legible."""
        img = img.convert("RGB")
        blurred = img.filter(ImageFilter.GaussianBlur(radius))
        out = img.copy()
        for box in self.logo_regions(img):
            region = blurred.crop(box)
            out.paste(region, box)
        return out

    def replace_region(self, img: Image.Image, fill_color: Tuple[int, int, int] = (128, 128, 128)) -> Image.Image:
        """Replace the logo region(s) with a flat neutral color, removing the
        shortcut entirely rather than just degrading it."""
        img = img.convert("RGB")
        out = img.copy()
        draw = ImageDraw.Draw(out)
        for box in self.logo_regions(img):
            draw.rectangle(box, fill=fill_color)
        return out

    def crop_region(self, img: Image.Image, margin_frac: float = 0.15) -> Image.Image:
        """Crop in from the edges enough to remove both logo placements
        (bottom-left and top-right are both near the border), then resize
        back to the original size so downstream preprocessing is unaffected."""
        w, h = img.size
        img = img.convert("RGB")
        left = int(w * margin_frac)
        top = int(h * margin_frac)
        right = w - int(w * margin_frac)
        bottom = h - int(h * margin_frac)
        cropped = img.crop((left, top, right, bottom))
        return cropped.resize((w, h))

    def apply(self, img: Image.Image, method: str) -> Image.Image:
        """Dispatch by method name, matching accuracy_delta.py's METHODS list."""
        dispatch = {
            "logo": self.paste_logo,
            "blur": self.blur_region,
            "replace": self.replace_region,
            "crop": self.crop_region,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown modification method: {method!r}. Expected one of {list(dispatch)}")
        return dispatch[method](img)
