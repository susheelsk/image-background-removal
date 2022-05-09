def error_dict(error_text: str):
    """
    Generates a dictionary containing $error_text error
    :param error_text: Error text
    :return: error dictionary
    """
    resp = {"errors": [{"title": error_text}]}
    return resp
