from PIL import Image


def __extact_alpha_channel__(image):
    """
    Extracts alpha channel from RGBA image
    :param image: RGBA pil image
    :return: RGB Pil image
    """
    # Extract just the alpha channel
    alpha = image.split()[-1]
    # Create a new image with an opaque black background
    bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
    # Copy the alpha channel to the new image using itself as the mask
    bg.paste(alpha, mask=alpha)
    return bg.convert("RGBA")