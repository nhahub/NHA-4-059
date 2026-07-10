"""
image_modifier.py
==================

Port of the logo-insertion logic from Section 3 of the notebook
(`paste_wm_logo_big`), extended with blur / replace / crop variants that
neutralize the logo instead of inserting it (used to test whether the
shortcut disappears once the logo is removed).

All methods operate on PIL Images and return PIL Images, so they can be
dropped straight into the CLIP preprocessing pipeline.

Per-category logo text
-----------------------
The original notebook pasted the exact same "WASTE MANAGEMENT" watermark
onto every truck class, including ones it makes no sense on (a pickup, a
fire engine). `paste_logo` now takes an optional `label` (an ImageNet
class index from TRUCK_CLASSES, or the category name string directly) and
picks a fake watermark whose text is plausible for that vehicle type
instead. Geometry (box size/position) is shared across categories — every
rendered logo uses the same 800x160 canvas scaled to 80% (bottom-left) /
55% (top-right) of the image width, so `logo_regions`/blur/replace/crop
don't need to know which text was used.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ImageNet class-index -> truck-subset label name. Duplicated from
# src.models.clip_model.TRUCK_CLASSES (values must stay in sync) rather than
# imported from it: clip_model.py hard-imports torch/clip, and this module
# otherwise has no ML dependencies at all (pure PIL) — importing it here
# would force every caller of ImageModifier, including its own no-torch
# unit tests, to have torch/clip installed just to build a watermark image.
TRUCK_CLASSES: Dict[str, int] = {
    "minivan": 656,
    "moving_van": 675,
    "police_van": 734,
    "fire_engine": 555,
    "garbage_truck": 569,
    "pickup": 717,
    "tow_truck": 864,
    "trailer_truck": 867,
}

# (main_text, sub_text) per truck category, rendered onto the same 800x120
# green watermark banner `_build_default_logo` used to hardcode. Keeping
# main_text roughly <=16 chars keeps it from overflowing the banner at the
# original font size (matches "WASTE MANAGEMENT"'s width budget).
LOGO_TEXT_BY_CATEGORY: Dict[str, Tuple[str, str]] = {
    "minivan": ("SUNSHINE SHUTTLE", "Family Transport Service"),
    "moving_van": ("ALLIED MOVERS", "Household & Office Relocation"),
    "police_van": ("METRO POLICE", "To Protect & Serve"),
    "fire_engine": ("CITY FIRE DEPT", "Emergency Response Unit"),
    "garbage_truck": ("WASTE MANAGEMENT", "Recycling & Disposal Services"),
    "pickup": ("ACE HARDWARE", "Contractor & Trade Supply"),
    "tow_truck": ("RAPID TOWING CO.", "24/7 Roadside Assistance"),
    "trailer_truck": ("GLOBAL FREIGHT", "Long-Haul Transport Services"),
}
DEFAULT_LOGO_CATEGORY = "garbage_truck"  # used when no label is given (back-compat)

# Corner-box geometry, as fractions of the (square, CLIP-preprocessed) image
# size. Shared with src.models.ch_mitigation so its spatial masking always
# covers whatever region paste_logo actually draws in, instead of a
# hand-picked pixel offset that silently stops covering the logo the next
# time these fractions change (see git history: the top-right box grew from
# 0.4 to 0.55 without CHMitigator's masking ever being updated to match).
BL_WIDTH_FRAC = 0.8   # bottom-left logo width, as a fraction of image width
TR_WIDTH_FRAC = 0.55  # top-right logo width, as a fraction of image width
LOGO_ASPECT_RATIO = 160 / 800  # logo_h / logo_w in _build_logo_image's default canvas


class ImageModifier:
    """Applies logo-insertion and logo-removal-style modifications to images."""

    def __init__(self):
        self._label_to_category = {v: k for k, v in TRUCK_CLASSES.items()}
        self._logos_by_category = {
            category: self._build_logo_image(main_text, sub_text)
            for category, (main_text, sub_text) in LOGO_TEXT_BY_CATEGORY.items()
        }
        # Kept as the default/back-compat logo: same object callers relied on
        # before per-category logos existed, and `logo_regions` only needs
        # its size (identical across all categories) for box geometry.
        self.logo = self._logos_by_category[DEFAULT_LOGO_CATEGORY]

    def _resolve_logo(self, label: Optional[Union[int, str]]) -> Image.Image:
        """Map a label (ImageNet class index or category name) to its
        pre-rendered logo image, falling back to the default when the
        label is missing or unrecognized."""
        if label is None:
            return self.logo
        category = label if isinstance(label, str) else self._label_to_category.get(label)
        return self._logos_by_category.get(category, self.logo)

    # ------------------------------------------------------------------
    # Logo construction (verbatim port of the notebook's inline logo,
    # parameterized on text so each category gets its own watermark)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_logo_image(main_text: str, sub_text: str, logo_w: int = 800, logo_h: int = 160) -> Image.Image:
        logo = Image.new("RGBA", (logo_w, logo_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(logo)
        draw.rectangle([0, 0, logo_w - 1, logo_h - 1], fill=(0, 100, 0, 255))

        # Bold border + separator rule + sunburst badge: early conv layers
        # (e.g. bn3) respond to edges/texture, not flat color fills, so a
        # plain solid banner barely registers there (its Grad-CAM focus score
        # actually *dropped* after logo insertion in a test run). These add
        # sharp, high-contrast edges packed into a small area without
        # touching the text's legibility.
        draw.rectangle([4, 4, logo_w - 5, logo_h - 5], outline=(255, 255, 255, 255), width=6)
        draw.line([(20, 86), (logo_w - 140, 86)], fill=(255, 255, 255, 200), width=3)

        cx, cy, r = logo_w - 70, logo_h // 2, 46
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, 255), width=4)
        for angle_deg in range(0, 360, 30):
            angle = math.radians(angle_deg)
            x2, y2 = cx + r * math.cos(angle), cy + r * math.sin(angle)
            draw.line([(cx, cy), (x2, y2)], fill=(255, 255, 255, 255), width=3)
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(255, 255, 255, 255))

        try:
            font_big = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54
            )
            font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28
            )
        except OSError:
            font_big = ImageFont.load_default()
            font_small = font_big
        draw.text((20, 18), main_text, fill=(255, 255, 255, 255), font=font_big)
        draw.text(
            (20, 95),
            sub_text,
            fill=(200, 255, 200, 255),
            font=font_small,
        )
        return logo

    # ------------------------------------------------------------------
    # Insertion (the "Clever Hans shortcut" injection used for detection)
    # ------------------------------------------------------------------
    def paste_logo(self, img: Image.Image, label: Optional[Union[int, str]] = None) -> Image.Image:
        """Paste a logo bottom-left (large) and top-right (small) — same
        geometry as the notebook's original `paste_wm_logo_big`.

        `label` picks which category's fake watermark text to use (an
        ImageNet class index from TRUCK_CLASSES, or the category name
        string directly, e.g. "fire_engine"). Defaults to the
        "WASTE MANAGEMENT" text when omitted, for callers that don't know
        the category.
        """
        logo = self._resolve_logo(label)
        img = img.convert("RGBA")
        w, h = img.size

        new_logo_width = int(w * BL_WIDTH_FRAC)
        new_logo_height = int(logo.size[1] * (new_logo_width / logo.size[0]))
        new_logo = logo.resize((new_logo_width, new_logo_height))
        offset = 5
        img.paste(new_logo, (offset, h - new_logo.size[1] - offset), new_logo)

        top_logo_width = int(w * TR_WIDTH_FRAC)
        top_logo_height = int(logo.size[1] * (top_logo_width / logo.size[0]))
        top_logo = logo.resize((top_logo_width, top_logo_height))
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

        bl_w = int(w * BL_WIDTH_FRAC)
        bl_h = int(self.logo.size[1] * (bl_w / self.logo.size[0]))
        bottom_left = (offset, h - bl_h - offset, offset + bl_w, h - offset)

        tr_w = int(w * TR_WIDTH_FRAC)
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

    def apply(self, img: Image.Image, method: str, label: Optional[Union[int, str]] = None) -> Image.Image:
        """Dispatch by method name, matching accuracy_delta.py's METHODS list.

        `label` is only used by the "logo" method (see `paste_logo`)."""
        dispatch = {
            "logo": lambda im: self.paste_logo(im, label=label),
            "blur": self.blur_region,
            "replace": self.replace_region,
            "crop": self.crop_region,
        }
        if method not in dispatch:
            raise ValueError(f"Unknown modification method: {method!r}. Expected one of {list(dispatch)}")
        return dispatch[method](img)
