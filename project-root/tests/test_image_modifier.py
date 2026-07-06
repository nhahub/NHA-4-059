from PIL import Image

from src.data.image_modifier import ImageModifier


def _sample_image(w=400, h=300, color=(10, 80, 200)):
    return Image.new("RGB", (w, h), color)


def test_paste_logo_preserves_size_and_mode():
    im = ImageModifier()
    img = _sample_image()
    out = im.apply(img, "logo")
    assert out.size == img.size
    assert out.mode == "RGB"


def test_blur_region_only_changes_logo_area():
    im = ImageModifier()
    img = _sample_image()
    out = im.apply(img, "blur")
    assert out.size == img.size

    # A pixel far from both logo regions (top-left corner) should be untouched.
    assert out.getpixel((2, 2)) == img.getpixel((2, 2))


def test_replace_region_fills_flat_color():
    im = ImageModifier()
    img = _sample_image()
    out = im.apply(img, "replace")
    (bl_box, tr_box) = im.logo_regions(img)
    # center of the bottom-left logo box should now be the fill color
    cx = (bl_box[0] + bl_box[2]) // 2
    cy = (bl_box[1] + bl_box[3]) // 2
    assert out.getpixel((cx, cy)) == (128, 128, 128)


def test_crop_region_returns_original_size():
    im = ImageModifier()
    img = _sample_image()
    out = im.apply(img, "crop")
    assert out.size == img.size


def test_unknown_method_raises():
    im = ImageModifier()
    img = _sample_image()
    try:
        im.apply(img, "nonexistent")
        assert False, "expected ValueError"
    except ValueError:
        pass
