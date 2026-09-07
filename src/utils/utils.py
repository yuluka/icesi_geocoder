import os
import logging
import pandas as pd
from typing import Any, Dict, Optional


## ----------- Logging -----------

logger: logging.Logger = logging.getLogger(__name__)


## ----------- Data Managing -----------

def read_data(
    file_path: str, 
    encoding: str = 'utf-8', 
    delimiter: str = ';',
    dtype_map: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Read a data file from the specified path, and return its contents as a pandas DataFrame.

    :param file_path: Path to the file.
    :type file_path: str
    :param encoding: Encoding of the file. Default is 'utf-8'.
    :type encoding: str
    :param delimiter: Delimiter used in the CSV file. Default is ';'.
    :type delimiter: str
    :param dtype_map: Map to force some columns to a specific type.
    :type dtype_map: Optional[Dict[str, Any]]
    :return: DataFrame containing the CSV data.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file type is not supported.
    :raises Exception: If there is an error reading the file.
    """

    try:
        if not os.path.isfile(file_path):
            logger.error(f"El archivo '{file_path}' no existe.")

            raise FileNotFoundError(f"The {file_path} file doesn't exist.")

        suffix: str = file_path.split('.')[-1].lower()

        if suffix not in ['csv', 'xlsx', 'xls']:
            logger.error(f"Tipo de archivo no soportado. El archivo '{file_path}' no es de tipo CSV o Excel.")

            raise ValueError(f"File type not supported. The {file_path} file is not a CSV or Excel.")


        read_kwargs = {}
        if dtype_map:
            read_kwargs["dtype"] = dtype_map

        df: pd.DataFrame = (
            pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                **read_kwargs
            ) 
            if suffix == 'csv' 
            else pd.read_excel(
                file_path,
                engine='openpyxl',
                **read_kwargs
            )
        )

        return df

    except Exception as e:
        logger.error(f"Error leyendo archivo '{file_path}': {e}")

        raise Exception(f"Error reading CSV file: {e}")


def read_excel_multiple_sheets(
    file_path: str,
    dtype_map: Optional[Dict[str, Any]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Read an Excel file with multiple sheets and return a dictionary where keys are sheet names and values are pandas DataFrames.

    :param file_path: Path to the Excel file.
    :type file_path: str
    :param dtype_map: Map to force some columns to a specific type.
    :type dtype_map: Optional[Dict[str, Any]]
    :return: Dictionary with sheet names as keys and DataFrames as values.
    :rtype: Dict[str, pd.DataFrame]
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file type is not supported.
    :raises Exception: If there is an error reading the file.
    """

    try:
        if not os.path.isfile(file_path):
            logger.error(f"El archivo '{file_path}' no existe.")

            raise FileNotFoundError(f"The {file_path} file doesn't exist.")

        suffix: str = file_path.split('.')[-1].lower()

        if suffix not in ['xlsx', 'xls']:
            logger.error(f"Tipo de archivo no soportado. El archivo '{file_path}' no es de tipo Excel.")

            raise ValueError(f"File type not supported. The {file_path} file is not an Excel file.")

        read_kwargs = {}
        if dtype_map:
            read_kwargs["dtype"] = dtype_map

        sheets: Dict[str, pd.DataFrame] = pd.read_excel(
            file_path,
            sheet_name=None,
            engine='openpyxl',
            **read_kwargs
        )

        return sheets

    except Exception as e:
        logger.error(f"Error leyendo archivo '{file_path}': {e}")
        raise Exception(f"Error reading Excel file: {e}")


def save_csv(
    df: pd.DataFrame,
    folder_path: str,
    file_name: str,
    encoding: str = 'utf-8',
    delimiter: str = ','
) -> None:
    """
    Save a pandas DataFrame to a CSV file in the specified folder.

    :param df: DataFrame to save.
    :type df: pd.DataFrame
    :param folder_path: Path to the folder where the CSV file will be saved.
    :type folder_path: str
    :param file_name: Name of the CSV file.
    :type file_name: str
    :param encoding: Encoding for the CSV file. Default is 'utf-8'.
    :type encoding: str
    :param delimiter: Delimiter for the CSV file. Default is ';'.
    :type delimiter: str
    :return: None
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, f"{file_name}.csv")

    try:
        df.to_csv(file_path, index=False, encoding=encoding, sep=delimiter)

        logger.info(f"DataFrame guardado en '{file_path}'")

    except Exception as e:
        logger.error(f"Error guardando DataFrame en CSV: {e}")


def save_excel(
    data: Dict[str, pd.DataFrame],
    folder_path: str,
    file_name: str,
) -> None:
    """
    Save a set of pandas DataFrames to a XLSX file in the specified folder.
    
    :param data: The data to be saved. Keys represent sheet names. Values represent the data.
    :type data: Dict[str, pd.DataFrame]
    :param folder_path: Path to the folder where the file will be saved.
    :type folder_path: str
    :param file_name: Name of the file.
    :type file_name: str
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = file_name if file_name.endswith('.xlsx') else f"{file_name}.xlsx"

    file_path = os.path.join(folder_path, file_name)

    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Datos guardados en '{file_path}'")

    except Exception as e:
        logger.error(f"Error guardando datos: {e}")
        raise
