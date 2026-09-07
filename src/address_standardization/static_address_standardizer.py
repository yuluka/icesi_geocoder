import re
import logging
from re import Match
from typing import List, Set, Union

from exceptions.error_type import ErrorType

"""
This module provides functionality to standardize address strings according to predefined rules.

The main idea is to clean and format addresses to a consistent structure to facilitate further processing or geocoding.

Standarizes the addresses in the following format:

`TIPO_VÍA[SUFIJO] [ORIENTACIÓN] # NÚMERO_CRUCE - NÚMERO_EDIFICACIÓN o bien INTERSECCIÓN via1 CON via2.`
"""


logger: logging.Logger = logging.getLogger(__name__)


# ------------------------------------- Maps and Regular expressions -------------------------------------
_VIAL_TYPE_MAP = {
    "CALLE": "CL",
    "C": "CL",
    "CLLE": "CL",
    "CLE": "CL",
    "CLL": "CL",
    "ALLE": "CL",
    "CAL": "CL",
    "CALL": "CL",
    "CARRERA": "KR",
    "CRA": "KR",
    "CR": "KR",
    "CRR": "KR",
    "K": "KR",
    "KRA": "KR",
    "AVENIDA": "AV",
    "AVE": "AV",
    "A": "AV",
    "AC": "AV",
    "AVDA": "AV",
    "AVD": "AV",
    "DIAGONAL": "DG",
    "DIG": "DG",
    "DIA": "DG",
    "DNAL": "DG",
    "TRANSVERSAL": "TV",
    "TR": "TV",
    "PASEO": "PS",
    "PS": "PS",
}

# Consultar si en Cali hay vías que lleguen hasta la N (no norte)
# Por ahora, las que tengan N las debemos poner en los logs. Lo mismo para O, Ñ y W
_ORIENTATIONS = {
    "N": "NORTE",
    "NTE": "NORTE",
    "NRTE": "NORTE",
    "NORTE": "NORTE",
    "O": "OESTE",
    "OE": "OESTE",
    "OESTE": "OESTE",
    "OR": "ORIENTE",
    "ORI": "ORIENTE",
    "ORIENTE": "ORIENTE",
    "NO": "NORTE",
    "NE": "NORESTE",
    "SO": "SUROESTE",
    "SE": "SURESTE",
    "S": "SUR",
    "E": "ESTE",
}

_STOP_WORDS_LIST = [
    "APARTAMENTO",
    "APTO",
    "APT",
    "AP",
    "TO",
    "CASA",
    "TORRE",
    "OFICINA",
    "BLOQUE",
    "UNIDAD",
    "BLQ",
    "BARRIO",
    "URBANIZACION",
    "URBANIZACIÓN",
    "CORREGIMIENTO",
    "SECTOR",
    "KM",
    "FINCA",
    "VEREDA",
    "CO",
    "PQDPSS",
    "PQ",
    "PQS",
]
_STOP_WORDS_FORMAT = ["BIS", "SUR", "ESTE", "OESTE", "NORTE"]

_STOP_WORDS_RE = re.compile(
    r"(" + "|".join(_STOP_WORDS_LIST) + r")", 
    flags=re.IGNORECASE
)
_NO_ADDRESS_RE = re.compile(
    r"^(SIN\s+INFORMACI[ÓO]N|SIN\s+DATOS?|NO\s+SABE|SD)\b", 
    flags=re.IGNORECASE
)
_SUFFIX_RE = re.compile(
    r"^(BIS|[A-Z]{1,2})$", 
    flags=re.IGNORECASE
)


# ------------------------------------- Aux functions -------------------------------------


def _clean_building(s: str) -> str:
    """
    Cleans the building part of the address by removing non-digit characters.

    :param s: Building part of the address.
    :type s: str
    :return: Cleaned building part with only digits.
    :rtype: str
    """

    return re.sub(r"\D", "", s) if s else s


def _clean_cross_street(c: str) -> str:
    """
    Cleans the cross street part of the address by removing suffixes.

    :param c: Cross street part of the address.
    :type c: str
    :return: Cleaned cross street part without suffixes.
    :rtype: str
    """

    m = re.match(r"^(\d+[A-Z]?)(\d*)", c)
    return m.group(1) if m else c


def _correct_spaces_between_numbers_n_letters(s: str) -> str:
    """
    Corrects spaces between numbers and letters in the address string.

    :param s: Address string.
    :type s: str
    :return: Address string with corrected spaces.
    :rtype: str
    """

    return re.sub(r"(\d+)\s+([A-Z])\s*(\d*)", r"\1\2\3", s)


def _insert_missing_hyphen(s: str) -> str:
    """
    Inserts missing hyphens between number-letter combinations in the address string.

    :param s: Address string.
    :type s: str
    :return: Address string with inserted hyphens.
    :rtype: str
    """

    return re.sub(r"(\d+[A-Z])(\d+)", r"\1-\2", s)


def _chequear_limite(tipo, numero) -> bool:
    val = int(re.match(r"\d+", numero).group())
    limites = {"KR": 168, "CL": 126, "AV": 11, "DG": 80, "TV": 100}
    return val > limites.get(tipo, 9999)


def _mapear_orientacion(tok: str) -> str:
    return _ORIENTATIONS.get(tok.upper(), "")


def format_output(address: str, stop_words: List[str]) -> str:
    """
    Formats the output address by adjusting spaces around stop words.

    :param address: Address string to be formatted.
    :type address: str
    :param stop_words: List of stop words to adjust spacing for.
    :type stop_words: List[str]
    :return: Formatted address string.
    :rtype: str
    """

    address = address.upper()
    address = re.sub(r"(\d)([A-Z])", r"\1 \2", address)
    address = re.sub(r"([A-Z])(\d)", r"\1 \2", address)
    address = re.sub(r"(\W)", r" \1 ", address)
    pattern = rf'([A-Z])(?=(?:{"|".join(stop_words)}))'
    address = re.sub(pattern, r"\1 ", address)
    address = re.sub(r"\s+", " ", address).strip()

    return address


def validate_correctness(address: str) -> bool:
    """
    Validate that the left side of the address (before '#') contains at least one number.

    :param address: Address string to be validated.
    :type address: str
    :return: True if the left side contains at least one number, False otherwise.
    :rtype: bool
    """

    address_parts: List[str] = address.split("#")

    if len(address_parts) > 1:
        left_side: str = address_parts[0].strip()

        return bool(re.search(r"(\d+)", left_side))

    return True


# ------------------------------------- Main function -------------------------------------


def standardize_address(address: str) -> Union[str, ErrorType]:
    """
    Standardizes a given address string according to predefined rules.

    :param address: Address string to be standardized.
    :type address: str
    :return: Standardized address string.
    :rtype: str
    """

    if not validate_correctness(address):
        return ErrorType.NOT_VALID_LEFT_SIDE

    if not isinstance(address, str) or not address.strip():
        return ErrorType.NOT_STRING_OR_EMPTY

    address = address.upper().strip().replace(".", "")

    if _NO_ADDRESS_RE.match(address):
        return ErrorType.NOT_INFORMATION

    valid_vial_types: Set[str] = set(_VIAL_TYPE_MAP.keys()) | set(
        _VIAL_TYPE_MAP.values()
    )
    tokens: List[str] = address.split()

    # Is the first token a valid vial type?
    is_valid_vial_type: bool = tokens and tokens[0] in valid_vial_types

    # Numbers amount in the address
    numbers_amount: int = len(re.findall(r"\d+", address))

    if not is_valid_vial_type or numbers_amount < 2:
        return ErrorType.NOT_VIAL_TYPE

    address = re.sub(r"\s+", " ", address)

    # Separate the address into left and right sides
    if "#" in address:
        addr_left_side, addr_right_side = map(str.strip, address.split("#", 1))
    else:
        addr_left_side, addr_right_side = address, ""

    # Find stop words on left side
    stop_match_left: Match[str] = _STOP_WORDS_RE.search(addr_left_side)

    if stop_match_left:
        addr_left_side = addr_left_side[: stop_match_left.start()].strip()

    # Find stop words on right side
    stop_match_right: Match[str] = _STOP_WORDS_RE.search(addr_right_side)

    if stop_match_right:
        addr_right_side = addr_right_side[: stop_match_right.start()].strip()

    # Clear spaces between letters and numbers
    addr_left_side = _correct_spaces_between_numbers_n_letters(addr_left_side)
    addr_left_side = _insert_missing_hyphen(addr_left_side)
    addr_right_side = _correct_spaces_between_numbers_n_letters(addr_right_side)
    addr_right_side = _insert_missing_hyphen(addr_right_side)

    words: List[str] = addr_left_side.split()

    if not words:
        return ErrorType.NOT_VALID_LEFT_SIDE

    addr_first_token: str = words[0]
    vial_type: str = _VIAL_TYPE_MAP.get(addr_first_token, addr_first_token)

    if addr_first_token == "K" and len(words) > 1 and words[1][0].isdigit():
        vial_type = "KR"
    elif addr_first_token.startswith("K") and addr_first_token[1:].isdigit():
        vial_type = addr_first_token

    base: str = vial_type + (" " + " ".join(words[1:]) if len(words) > 1 else "")

    if addr_right_side:
        # Separate cross and building
        if "-" in addr_right_side:
            cross_street_str, building_str = map(str.strip, addr_right_side.split("-", 1))

            cross_street = _clean_cross_street(cross_street_str)
            building = _clean_building(building_str)
            result = f"{base} # {cross_street} - {building}"

        else:
            addr_right_side_tokens: List[str] = addr_right_side.strip().split()

            if len(addr_right_side_tokens) >= 2:
                cross_street_str = addr_right_side_tokens[0]
                building_str = " ".join(addr_right_side_tokens[1:])
                cross_street = _clean_cross_street(cross_street_str)
                building = _clean_building(building_str)
                result = f"{base} # {cross_street} - {building}"

            else:
                cross_street = _clean_cross_street(addr_right_side.strip())
                result = f"{base} # {cross_street}"
    else:
        result = base

    result = re.sub(r"[-\s]+$", "", result)
    result = re.sub(r"\s+", " ", result).strip()

    if re.search(r"(N|Ñ)", result):
        return ErrorType.PROBLEMATIC_N

    result = format_output(result, _STOP_WORDS_FORMAT)

    return result
