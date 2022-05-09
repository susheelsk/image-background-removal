def process_remove_bg(params, image, bg, is_json_or_www_encoded=False):
    """
    Handles a request to the removebg api method
    :param params: parameters
    :param image: foreground pil image
    :param is_json_or_www_encoded: is "json" or "x-www-form-urlencoded" content-type
    :param bg: background pil image
    :return: tuple or dict
    """
    h, w = image.size
    if h < 2 or w < 2:
        return error_dict("Image is too small. Minimum size 2x2"), 400  # TODO: вынести наружу
    if "size" in params.keys():
        value = params["size"]
        if value == "preview" or value == "small" or value == "regular":
            image.thumbnail((625, 400), resample=Image.ANTIALIAS)  # 0.25 mp
        elif value == "medium":
            image.thumbnail((1504, 1000), resample=Image.ANTIALIAS)  # 1.5 mp
        elif value == "hd":
            image.thumbnail((2000, 2000), resample=Image.ANTIALIAS)  # 2.5 mp
        else:
            image.thumbnail((6250, 4000), resample=Image.ANTIALIAS)  # 25 mp

    roi_box = [0, 0, image.size[0], image.size[1]]
    if "type" in params.keys():
        value = params["type"]
        if len(value) > 0 and Config.prep_method:
            Config.prep_method.foreground = value

    if "roi" in params.keys():
        value = params["roi"].split(" ")
        if len(value) == 4:
            for i, coord in enumerate(value):
                if "px" in coord:
                    coord = coord.replace("px", "")
                    try:
                        coord = int(coord)
                    except BaseException:
                        return error_dict("Error converting roi coordinate string to number!"), 400
                    if coord < 0:
                        error_dict(
                            "Bad roi coordinate."), 400
                    if (i == 0 or i == 2) and coord > image.size[0]:
                        return error_dict(
                            "The roi coordinate cannot be larger than the image size."), 400
                    elif (i == 1 or i == 3) and coord > image.size[1]:
                        return error_dict(
                            "The roi coordinate cannot be larger than the image size."), 400
                    roi_box[i] = int(coord)
                elif "%" in coord:
                    coord = coord.replace("%", "")
                    try:
                        coord = int(coord)
                    except BaseException:
                        return error_dict("Error converting roi coordinate string to number!"), 400
                    if coord > 100:
                        return error_dict("The coordinate cannot be more than 100%"), 400
                    elif coord < 0:
                        return error_dict("Coordinate cannot be less than 0%"), 400
                    if i == 0 or i == 2:
                        coord = int(image.size[0] * coord / 100)
                    elif i == 1 or i == 3:
                        coord = int(image.size[1] * coord / 100)
                    roi_box[i] = coord
                else:
                    return error_dict("Something wrong with roi coordinates!"), 400

    new_image = image.copy()
    new_image = new_image.crop(roi_box)
    h, w = new_image.size
    if h < 2 or w < 2:
        return error_dict("Image is too small. Minimum size 2x2"), 400
    new_image = Config.model.process_image(new_image, Config.prep_method, Config.post_method)

    if Config.prep_method:
        Config.prep_method.foreground = "auto"

    scaled = False
    if "scale" in params.keys() and params['scale'] != 100:
        value = params["scale"]
        new_image.thumbnail((int(image.size[0] * value / 100),
                             int(image.size[1] * value / 100)), resample=Image.ANTIALIAS)
        scaled = True
    if "crop" in params.keys():
        value = params["crop"]
        if value:
            new_image = new_image.crop(new_image.getbbox())
            if "crop_margin" in params.keys():
                crop_margin = params["crop_margin"]
                if "px" in crop_margin:
                    crop_margin = crop_margin.replace("px", "")
                    crop_margin = abs(int(crop_margin))
                    if crop_margin > 500:
                        return error_dict(
                            "The crop_margin cannot be larger than the original image size."), 400
                    new_image = add_margin(new_image, crop_margin,
                                           crop_margin, crop_margin, crop_margin, (0, 0, 0, 0))
                elif "%" in crop_margin:
                    crop_margin = crop_margin.replace("%", "")
                    crop_margin = int(crop_margin)
                    new_image = add_margin(new_image, int(new_image.size[1] * crop_margin / 100),
                                           int(new_image.size[0] * crop_margin / 100),
                                           int(new_image.size[1] * crop_margin / 100),
                                           int(new_image.size[0] * crop_margin / 100), (0, 0, 0, 0))
        else:
            if "position" in params.keys() and scaled is False:
                value = params["position"]
                if len(value) == 2:
                    new_image = transparancy_paste(Image.new("RGBA", image.size), new_image,
                                                   (int(image.size[0] * value[0] / 100),
                                                    int(image.size[1] * value[1] / 100)))
                else:
                    new_image = transparancy_paste(Image.new("RGBA", image.size), new_image, roi_box)
            elif scaled is False:
                new_image = transparancy_paste(Image.new("RGBA", image.size), new_image, roi_box)

    if "channels" in params.keys():
        value = params["channels"]
        if value == "alpha":
            new_image = __extact_alpha_channel__(new_image)
        else:
            bg_chaged = False
            if "bg_color" in params.keys():
                value = params["bg_color"]
                if len(value) > 0:
                    color = ImageColor.getcolor(value, "RGB")
                    bg = Image.new("RGBA", new_image.size, color)
                    bg = transparancy_paste(bg, new_image, (0, 0))
                    new_image = bg.copy()
                    bg_chaged = True
            if "bg_image_url" in params.keys() and bg_chaged is False:
                value = params["bg_image_url"]
                if len(value) > 0:
                    try:
                        bg = Image.open(io.BytesIO(requests.get(value).content))
                    except BaseException:
                        return error_dict("Error download background image!"), 400
                    bg = bg.resize(new_image.size)
                    bg = bg.convert("RGBA")
                    bg = transparancy_paste(bg, new_image, (0, 0))
                    new_image = bg.copy()
                    bg_chaged = True
            if not is_json_or_www_encoded:
                if bg and bg_chaged is False:
                    bg = bg.resize(new_image.size)
                    bg = bg.convert("RGBA")
                    bg = transparancy_paste(bg, new_image, (0, 0))
                    new_image = bg.copy()
    if "format" in params.keys():
        value = params["format"]
        if value == "jpg":
            new_image = new_image.convert("RGB")
            img_io = io.BytesIO()
            new_image.save(img_io, 'JPEG', quality=100)
            img_io.seek(0)
            return {"type": "jpg", "data": [img_io, new_image.size]}
        elif value == "zip":
            mask = __extact_alpha_channel__(new_image)
            mask_buff = io.BytesIO()
            mask.save(mask_buff, 'PNG')
            mask_buff.seek(0)
            image_buff = io.BytesIO()
            image.save(image_buff, 'JPEG')
            image_buff.seek(0)
            fileobj = io.BytesIO()
            with zipfile.ZipFile(fileobj, 'w') as zip_file:
                zip_info = zipfile.ZipInfo(filename="color.jpg")
                zip_info.date_time = time.localtime(time.time())[:6]
                zip_info.compress_type = zipfile.ZIP_DEFLATED
                zip_file.writestr(zip_info, image_buff.getvalue())
                zip_info = zipfile.ZipInfo(filename="alpha.png")
                zip_info.date_time = time.localtime(time.time())[:6]
                zip_info.compress_type = zipfile.ZIP_DEFLATED
                zip_file.writestr(zip_info, mask_buff.getvalue())
            fileobj.seek(0)
            return {"type": "zip", "data": [fileobj.read(), new_image.size]}
        else:
            buff = io.BytesIO()
            new_image.save(buff, 'PNG')
            buff.seek(0)
            return {"type": "png", "data": [buff, new_image.size]}
    return error_dict("Something wrong with request or http api. Please, open new issue on Github! This is error in "
                      "code."), 400
