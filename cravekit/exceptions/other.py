from pydantic import ValidationError


class WrongParametersException(ValidationError):
    """This exception risen when wrong parameters specified."""
