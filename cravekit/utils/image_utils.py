from PIL import Image


def transparency_paste(bg_img, fg_img, box=(0, 0)):
    """
    Inserts an image into another image while maintaining transparency.
    :param bg_img: Background pil image
    :param fg_img: Foreground pil image
    :param box: Bounding box
    :return: Pil Image
    """
    fg_img_trans = Image.new("RGBA", bg_img.size)
    fg_img_trans.paste(fg_img, box, mask=fg_img)
    new_img = Image.alpha_composite(bg_img, fg_img_trans)
    return new_img


def add_margin(pil_img, top, right, bottom, left, color):
    """
    Adds fields to the image.
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result