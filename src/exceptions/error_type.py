# POSIBLES ERRORES:
from enum import Enum

class ErrorType(Enum):
    """
    Enum for error types.
    """
    NOT_STRING_OR_EMPTY = "Not a string or is empty"
    NOT_INFORMATION = "Does not contains information"
    NOT_VIAL_TYPE = "Does not have a vial type or has less than 2 numbers"
    NOT_VALID_LEFT_SIDE = "Does not have a valid left side"
    PROBLEMATIC_N = "Contains a problematic N character"