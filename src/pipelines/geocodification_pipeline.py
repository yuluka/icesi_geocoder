import os
import geopy
import time
import logging
import re
import pandas as pd
from geopy.geocoders import ArcGIS
from pathlib import Path
from typing import Set, Tuple, Optional, List, Union, Callable, Dict, Any

from config import settings
from logger.logger_config import create_log
from exceptions.error_type import ErrorType
from address_standardization.model_resources import AddressStandardizationModel
from address_standardization import static_address_standardizer as static_standardizer
from address_standardization import s2s_address_standardizer as s2s_standardizer
from utils import utils


## ----------- Constants -----------

GEOCODING_FAILED_MESSAGE: str = "Geocodificación fallida"
STANDARDIZING_FAILED_MESSAGE: str = "Estandarización fallida"


## ----------- Logging -----------

create_log()
logger: logging.Logger = logging.getLogger(__name__)


## ----------- Standardization -----------

def standardize_df_addresses(
    df: pd.DataFrame, 
    raw_address_column: str = "dir_orig", 
    standardized_address_column: str = "direccion_norm",
    method_column: str = "metodo_estandarizacion",
    model: Optional[AddressStandardizationModel] = None,
    batch_size: int = 64,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    stage_progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
) -> pd.DataFrame:
    """
    Standardize addresses in a DataFrame using a hybrid fallback strategy:
    1. First, apply fast rule-based static address standardizer.
    2. For addresses that fail static standardization, send raw addresses in batch to S2S model.
    3. Addresses that fail both are flagged with STANDARDIZING_FAILED_MESSAGE.

    Adds metadata to `method_column` ('ESTATICO', 'S2S_RESCATADO', 'FALLIDA').

    :param df: DataFrame with the addresses to standardize.
    :type df: pd.DataFrame
    :param raw_address_column: Column name containing the raw addresses.
    :type raw_address_column: str
    :param standardized_address_column: Column name to store the standardized addresses.
    :type standardized_address_column: str
    :param method_column: Column name to store the standardization method.
    :type method_column: str
    :param model: S2S model to use for standardization (optional).
    :type model: Optional[AddressStandardizationModel]
    :param batch_size: Batch size for S2S model.
    :type batch_size: int
    :param progress_callback: Callback function to report progress.
    :type progress_callback: Optional[Callable[[int, int, str], None]]
    :param stage_progress_callback: Stage callback for granular progress tracking (e.g. CLI).
    :type stage_progress_callback: Optional[Callable[[str, int, int, str], None]]
    :return: DataFrame with standardized addresses and metadata.
    :rtype: pd.DataFrame
    """

    logger.info("Iniciando paso de estandarización de direcciones...")

    df[raw_address_column] = df[raw_address_column].fillna("").astype(str)
    df[standardized_address_column] = None
    df[method_column] = "FALLIDA"

    failed_indices: List[int] = []
    failed_raw_addresses: List[str] = []
    failed_reasons: List[str] = []

    total_rows = len(df)
    if stage_progress_callback:
        stage_progress_callback("static", 0, total_rows, "Iniciando estandarización estática")

    # 1. Fase Estática
    for i in range(total_rows):
        original_address: str = df.loc[i, raw_address_column]
        standardized_address: Union[str, ErrorType] = static_standardizer.standardize_address(original_address)

        if isinstance(standardized_address, ErrorType):
            if model is not None:
                failed_indices.append(i)
                failed_raw_addresses.append(original_address)
                failed_reasons.append(standardized_address.value)
            else:
                logger.data_quality(f"Error estandarizando dirección en índice {i+2} '{original_address}': {standardized_address.value}")

                df.loc[i, standardized_address_column] = STANDARDIZING_FAILED_MESSAGE
                df.loc[i, method_column] = "FALLIDA"

        else:
            clean_address: str = (
                standardized_address[:-1].strip()
                if standardized_address.strip().endswith("-")
                else standardized_address
            )

            df.loc[i, standardized_address_column] = clean_address
            df.loc[i, method_column] = "ESTATICO"

        if stage_progress_callback and (i % 25 == 0 or i == total_rows - 1):
            stage_progress_callback("static", i + 1, total_rows, "")

        if progress_callback and (i % 50 == 0 or i == total_rows - 1):
            static_pct = int((i + 1) / total_rows * (20 if model is not None else 100))
            progress_callback(static_pct, 100, f"Estandarizando con reglas estáticas: fila {i+1}/{total_rows}...")

    if stage_progress_callback:
        stage_progress_callback("static", total_rows, total_rows, "Completado")

    # 2. Fase S2S (Rescate en lote)
    if model is not None and failed_raw_addresses:
        static_ok = total_rows - len(failed_indices)
        num_failed = len(failed_indices)

        logger.info(
            f"Estandarización estática completó {static_ok}/{total_rows}. "
            f"Enviando {num_failed} casos a rescate con modelo S2S..."
        )

        total_batches = (num_failed + batch_size - 1) // batch_size if num_failed > 0 else 1

        if progress_callback:
            progress_callback(
                20, 100, f"Iniciando modelo S2S sobre {num_failed} casos ({total_batches} lotes)..."
            )

        if stage_progress_callback:
            stage_progress_callback("s2s", 0, num_failed, f"Iniciando {total_batches} lotes")

        def s2s_batch_callback(processed: int, total_to_pred: int, batch_idx: int = 1, total_b: int = 1):
            if stage_progress_callback:
                stage_progress_callback("s2s", processed, total_to_pred, f"Lote {batch_idx}/{total_b}")

            if progress_callback:
                s2s_frac = processed / total_to_pred if total_to_pred > 0 else 1.0
                current_pct = int(20 + s2s_frac * 80)
                progress_callback(
                    current_pct,
                    100,
                    f"Estandarizando con modelo S2S: lote {batch_idx}/{total_b} ({processed}/{total_to_pred} casos)...",
                )

        s2s_predictions: Union[str, List[Union[str, ErrorType]]] = s2s_standardizer.predict_batch(
            model=model,
            texts=failed_raw_addresses,
            num_beams=4,
            max_new_tokens=64,
            batch_size=batch_size,
            progress_callback=s2s_batch_callback,
        )

        if isinstance(s2s_predictions, str):
            s2s_predictions = [s2s_predictions]

        rescued_count: int = 0

        for idx, raw_addr, static_reason, pred in zip(
            failed_indices, failed_raw_addresses, failed_reasons, s2s_predictions
        ):
            if isinstance(pred, ErrorType) or not pred or not isinstance(pred, str):
                s2s_reason = pred.value if isinstance(pred, ErrorType) else "Predicción vacía"

                logger.data_quality(
                    f"Error estandarizando dirección en índice {idx+2} '{raw_addr}'. "
                    f"Estático: {static_reason} | S2S: {s2s_reason}"
                )

                df.loc[idx, standardized_address_column] = STANDARDIZING_FAILED_MESSAGE
                df.loc[idx, method_column] = "FALLIDA"

            else:
                clean_pred: str = pred[:-1].strip() if pred.strip().endswith("-") else pred
                df.loc[idx, standardized_address_column] = clean_pred
                df.loc[idx, method_column] = "S2S_RESCATADO"
                rescued_count += 1

                logger.info(f"Dirección rescatada con S2S en índice {idx+2}: '{raw_addr}' -> '{clean_pred}'")

        logger.info(f"Rescate S2S finalizado: {rescued_count}/{len(failed_indices)} direcciones recuperadas.")

        if stage_progress_callback:
            stage_progress_callback("s2s", num_failed, num_failed, f"Rescatadas: {rescued_count}/{num_failed}")

    elif model is not None and not failed_raw_addresses:
        if progress_callback:
            progress_callback(100, 100, "Estandarización completada (100% resuelto con reglas estáticas)")

    if progress_callback:
        progress_callback(100, 100, "Estandarización finalizada")

    return df


## ----------- Geocodification -----------

def geocode_address(
    address: str, 
    geocoder: ArcGIS, 
    city: str = "Cali", 
    country: str = "Valle del Cauca, Colombia",
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> Tuple[str, Union[float, str], Union[float, str]]:
    """
    Geocode an address using ArcGIS geocoder with retry support and return (validated_address, lat, lon).

    :param address: Address to geocode.
    :type address: str
    :param geocoder: ArcGIS geocoder.
    :type geocoder: ArcGIS
    :param city: City to use for geocoding.
    :type city: str
    :param country: Country to use for geocoding.
    :type country: str
    :param max_retries: Number of retry attempts on network error/timeout.
    :type max_retries: int
    :param retry_delay: Base delay in seconds between retries.
    :type retry_delay: float
    :return: Tuple with (validated_address, latitude, longitude).
    :rtype: Tuple[str, Union[float, str], Union[float, str]]
    """

    query = f"{address}, {city}, {country}" if city or country else address

    for attempt in range(max_retries + 1):
        try:
            result: Optional[geopy.Location] = geocoder.geocode(query)

            if result:
                return result.address, result.latitude, result.longitude
            else:
                logger.data_quality(f"Error geocodificando la dirección '{address}' en '{query}'")
                return (GEOCODING_FAILED_MESSAGE,) * 3

        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"Intento {attempt + 1}/{max_retries + 1} falló para '{address}': {e}. Reintentando en {retry_delay * (attempt + 1)}s..."
                )
                time.sleep(retry_delay * (attempt + 1))
                continue

            logger.exception(f"Excepción geocodificando dirección '{address}' tras {max_retries + 1} intentos: {e}")
            return (GEOCODING_FAILED_MESSAGE,) * 3

    return (GEOCODING_FAILED_MESSAGE,) * 3


def geocode_df_addresses(
    df: pd.DataFrame, 
    geocoder: ArcGIS, 
    original_address_column: str = "direccion_norm", 
    validated_address_column: str = "validated_address", 
    latitude_column: str = "latitud", 
    longitude_column: str = "longitud",
    status_column: str = "estado_geocodificacion",
    city: str = "Cali",
    country: str = "Valle del Cauca, Colombia",
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    stage_progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    checkpoint_interval: int = 50,
    checkpoint_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Geocode addresses in a DataFrame using ArcGIS with support for checkpoints and resumption.
    Adds `status_column` ('GEOCODIFICADA', 'NO_LOCALIZABLE', 'NO_ESTANDARIZADA').

    :param df: DataFrame with the addresses to geocode.
    :type df: pd.DataFrame
    :param geocoder: ArcGIS geocoder.
    :type geocoder: ArcGIS
    :param original_address_column: Column name containing the raw addresses.
    :type original_address_column: str
    :param validated_address_column: Column name to store the standardized addresses.
    :type validated_address_column: str
    :param latitude_column: Column name to store the latitudes.
    :type latitude_column: str
    :param longitude_column: Column name to store the longitudes.
    :type longitude_column: str
    :param status_column: Column name to store the geocoding status.
    :type status_column: str
    :param city: City to use for geocoding.
    :type city: str
    :param country: Country to use for geocoding.
    :type country: str
    :param progress_callback: Callback function to report progress.
    :type progress_callback: Optional[Callable[[int, int, str], None]]
    :param stage_progress_callback: Stage callback for granular progress tracking (e.g. CLI).
    :type stage_progress_callback: Optional[Callable[[str, int, int, str], None]]
    :param checkpoint_path: Path to checkpoint file (.csv) for saving interim progress.
    :type checkpoint_path: Optional[Union[str, Path]]
    :param checkpoint_interval: Number of rows between checkpoint saves.
    :type checkpoint_interval: int
    :param checkpoint_df: Pre-loaded checkpoint DataFrame for resumption.
    :type checkpoint_df: Optional[pd.DataFrame]
    :return: DataFrame with geocoded addresses and metadata.
    :rtype: pd.DataFrame
    """

    logger.info("Iniciando paso de geocodificación...")

    group_df: pd.DataFrame = df.copy()

    for col in [validated_address_column, latitude_column, longitude_column]:
        if col not in group_df.columns:
            group_df[col] = None

    if status_column not in group_df.columns:
        group_df[status_column] = None

    if checkpoint_df is not None:
        for col in [validated_address_column, latitude_column, longitude_column, status_column]:
            if col in checkpoint_df.columns:
                group_df[col] = checkpoint_df[col]

    total_rows: int = len(df)
    valid_done_statuses: Set[str] = {"GEOCODIFICADA", "NO_LOCALIZABLE", "GEOCODIFICACION_GENERICA", "GEOCODIFICACION_PARCIAL", "NO_ESTANDARIZADA"}
    is_already_done = group_df[status_column].isin(valid_done_statuses)
    num_resumed = int(is_already_done.sum())

    if num_resumed > 0:
        logger.info(f"Reanudando geocodificación: {num_resumed}/{total_rows} registros ya estaban procesados previamente.")

        if stage_progress_callback:
            stage_progress_callback("geocoding", num_resumed, total_rows, f"Reanudadas {num_resumed}/{total_rows}")

    def save_checkpoint(row_idx: int):
        if checkpoint_path:
            try:
                cp_p = Path(checkpoint_path)
                cp_p.parent.mkdir(parents=True, exist_ok=True)
                temp_cp = cp_p.with_suffix(".tmp")
                group_df.to_csv(temp_cp, sep=";", index=False, encoding="utf-8-sig")
                temp_cp.replace(cp_p)
                logger.debug(f"Checkpoint guardado en fila {row_idx}: {cp_p}")

            except Exception as cp_err:
                logger.warning(f"No se pudo guardar el checkpoint en la fila {row_idx}: {cp_err}")

    current_idx = 0

    try:
        for i in range(total_rows):
            current_idx = i

            if is_already_done.iloc[i]:
                continue

            original_address: str = group_df.loc[i, original_address_column]

            if (
                original_address is None
                or original_address == STANDARDIZING_FAILED_MESSAGE
                or not str(original_address).strip()
            ):
                logger.data_quality(f"Error geocodificando dirección en índice {i+2}. '{original_address}': No estandarizada")
                group_df.loc[i, status_column] = "NO_ESTANDARIZADA"
                group_df.loc[i, [validated_address_column, latitude_column, longitude_column]] = (None, None, None)

            else:
                validated_address, latitude, longitude = geocode_address(
                    address=original_address,
                    geocoder=geocoder,
                    city=city,
                    country=country,
                )

                address_numbers = re.findall(r'\d+', validated_address.split(",")[0]) if validated_address else []

                # Posibles casos de fallo en geocodificación
                if validated_address == GEOCODING_FAILED_MESSAGE:
                    logger.data_quality(f"Error geocodificando dirección en índice {i+2} '{original_address}': Resultado no encontrado")

                    group_df.loc[i, [validated_address_column, latitude_column, longitude_column, status_column]] = (
                        GEOCODING_FAILED_MESSAGE, None, None, "NO_LOCALIZABLE"
                    )


                elif validated_address == "Cali, Valle del Cauca":
                    logger.data_quality(f"Error geocodificando dirección en índice {i+2} '{original_address}': Geocodificación genérica")

                    group_df.loc[i, [validated_address_column, latitude_column, longitude_column, status_column]] = (
                        GEOCODING_FAILED_MESSAGE, None, None, "GEOCODIFICACION_GENERICA"
                    )

                elif len(address_numbers) == 1:
                    logger.data_quality(f"Error geocodificando dirección en índice {i+2} '{original_address}': Geocodificación parcial")

                    group_df.loc[i, [validated_address_column, latitude_column, longitude_column, status_column]] = (
                        GEOCODING_FAILED_MESSAGE, None, None, "GEOCODIFICACION_PARCIAL"
                    )

                else:
                    group_df.loc[i, [validated_address_column, latitude_column, longitude_column, status_column]] = (
                        validated_address, latitude, longitude, "GEOCODIFICADA"
                    )

            if checkpoint_path and ((i + 1) % checkpoint_interval == 0 or i == total_rows - 1):
                save_checkpoint(i + 1)

            if stage_progress_callback:
                stage_progress_callback("geocoding", i + 1, total_rows, "")

            if progress_callback and (i % 10 == 0 or i == total_rows - 1):
                progress_callback(i + 1, total_rows, f"Geocodificando fila {i+1}/{total_rows} con ArcGIS...")

        if checkpoint_path:
            save_checkpoint(total_rows)

        return group_df

    except KeyboardInterrupt:
        logger.warning("Interrupción detectada en geocodificación (Ctrl+C). Guardando estado actual...")

        if checkpoint_path:
            save_checkpoint(current_idx + 1)

        raise

    except Exception as e:
        logger.exception(f"Error durante la geocodificación: {e}")

        if checkpoint_path:
            save_checkpoint(current_idx + 1)

        raise Exception(f"Error geocodificando DataFrame: {e}")


# ----------- Metrics Calculation -----------

def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute structured metrics from a processed DataFrame for dashboarding.

    :param df: DataFrame with the processed addresses.
    :type df: pd.DataFrame
    :return: Dictionary with metrics.
    :rtype: Dict[str, Any]
    """

    total_records: int = len(df)

    if total_records == 0:
        return {}

    method_col = "metodo_estandarizacion" if "metodo_estandarizacion" in df.columns else None
    status_col = "estado_geocodificacion" if "estado_geocodificacion" in df.columns else None

    static_standardized_count: int = int((df[method_col] == "ESTATICO").sum()) if method_col else 0
    s2s_rescued_count: int = int((df[method_col] == "S2S_RESCATADO").sum()) if method_col else 0
    failed_standardization_count: int = int((df[method_col] == "FALLIDA").sum()) if method_col else 0
    total_standardized_count: int = static_standardized_count + s2s_rescued_count

    geocoded_success_count: int = int((df[status_col] == "GEOCODIFICADA").sum()) if status_col else 0
    generic_geocoding_count: int = int((df[status_col] == "GEOCODIFICACION_GENERICA").sum()) if status_col else 0
    partial_geocoding_count: int = int((df[status_col] == "GEOCODIFICACION_PARCIAL").sum()) if status_col else 0
    unlocatable_count: int = int((df[status_col] == "NO_LOCALIZABLE").sum()) if status_col else 0
    unstandardized_count: int = int((df[status_col] == "NO_ESTANDARIZADA").sum()) if status_col else 0

    return {
        "total_records": total_records,
        "standardized_static": static_standardized_count,
        "rescued_s2s": s2s_rescued_count,
        "total_standardized": total_standardized_count,
        "failed_standardization": failed_standardization_count,
        "geocoded_success": geocoded_success_count,
        "generic_geocoding": generic_geocoding_count,
        "partial_geocoding": partial_geocoding_count,
        "unlocatable": unlocatable_count,
        "unstandardized": unstandardized_count,
        "standardization_percentage": round((total_standardized_count / total_records) * 100, 2) if total_records else 0.0,
        "geocoding_success_percentage": round((geocoded_success_count / total_records) * 100, 2) if total_records else 0.0,
        "generic_percentage": round((generic_geocoding_count / total_records) * 100, 2) if total_records else 0.0,
        "partial_percentage": round((partial_geocoding_count / total_records) * 100, 2) if total_records else 0.0,
        "s2s_contribution_percentage": round((s2s_rescued_count / total_records) * 100, 2) if total_records else 0.0,

        # Aliases for backward compatibility
        "total_registros": total_records,
        "estandarizadas_estatico": static_standardized_count,
        "rescatadas_s2s": s2s_rescued_count,
        "total_estandarizadas": total_standardized_count,
        "fallidas_estandarizacion": failed_standardization_count,
        "geocodificadas_exito": geocoded_success_count,
        "geocodificacion_generica": generic_geocoding_count,
        "geocodificacion_parcial": partial_geocoding_count,
        "no_localizables": unlocatable_count,
        "no_estandarizadas": unstandardized_count,
        "porcentaje_estandarizacion": round((total_standardized_count / total_records) * 100, 2) if total_records else 0.0,
        "porcentaje_geocodificacion": round((geocoded_success_count / total_records) * 100, 2) if total_records else 0.0,
        "porcentaje_generica": round((generic_geocoding_count / total_records) * 100, 2) if total_records else 0.0,
        "porcentaje_parcial": round((partial_geocoding_count / total_records) * 100, 2) if total_records else 0.0,
        "porcentaje_s2s_aporte": round((s2s_rescued_count / total_records) * 100, 2) if total_records else 0.0,
    }


# ----------- Main Pipeline -----------

def geocode_pipeline(
    data: Union[pd.DataFrame, str],
    raw_address_column: str = "dir_orig",
    standardized_address_column: str = "direccion_norm",
    method_column: str = "metodo_estandarizacion",
    validated_address_column: str = "validated_address",
    latitude_column: str = "latitud",
    longitude_column: str = "longitud",
    status_column: str = "estado_geocodificacion",
    city: str = "Cali",
    country: str = "Valle del Cauca, Colombia",
    model: Optional[AddressStandardizationModel] = None,
    model_path: Optional[str] = None,
    use_s2s: bool = True,
    batch_size: int = 64,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    stage_progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    checkpoint_interval: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete pipeline to load, standardize (Hybrid: Static + S2S fallback), and geocode addresses.
    Supports incremental checkpoints and resumable processing.
    Returns (processed_df, metrics_dict).

    :param data: DataFrame with the addresses to geocode or path to CSV file.
    :type data: Union[pd.DataFrame, str]
    :param raw_address_column: Column name containing the raw addresses.
    :type raw_address_column: str
    :param standardized_address_column: Column name to store the standardized addresses.
    :type standardized_address_column: str
    :param method_column: Column name to store the standardization method.
    :type method_column: str
    :param validated_address_column: Column name to store the validated geocoded addresses.
    :type validated_address_column: str
    :param latitude_column: Column name to store the latitudes.
    :type latitude_column: str
    :param longitude_column: Column name to store the longitudes.
    :type longitude_column: str
    :param status_column: Column name to store the geocoding status.
    :type status_column: str
    :param city: City to use for geocoding.
    :type city: str
    :param country: Country to use for geocoding.
    :type country: str
    :param model: Pre-loaded S2S model instance (optional, avoids reloading).
    :type model: Optional[AddressStandardizationModel]
    :param model_path: Path to the S2S model (used if model instance is not provided).
    :type model_path: Optional[str]
    :param use_s2s: Whether to use S2S model for standardization.
    :type use_s2s: bool
    :param batch_size: Batch size for S2S model.
    :type batch_size: int
    :param progress_callback: Callback function to report progress.
    :type progress_callback: Optional[Callable[[int, int, str], None]]
    :param stage_progress_callback: Stage callback for granular progress tracking (e.g. CLI).
    :type stage_progress_callback: Optional[Callable[[str, int, int, str], None]]
    :param checkpoint_path: Path to checkpoint file (.csv) for saving interim progress.
    :type checkpoint_path: Optional[Union[str, Path]]
    :param checkpoint_interval: Number of rows between checkpoint saves.
    :type checkpoint_interval: int
    :return: Tuple with (processed_df, metrics_dict).
    :rtype: Tuple[pd.DataFrame, Dict[str, Any]]
    """

    try:
        logger.info("Iniciando pipeline de geocodificación...")

        start_time = time.time()

        # S2S Model Loading
        if not use_s2s:
            model = None
        elif model is None:
            target_model_path: Optional[str] = model_path or (settings.S2S_MODEL_PATH if hasattr(settings, "S2S_MODEL_PATH") else None)

            if target_model_path:
                abs_model_path = os.path.abspath(target_model_path)

                if os.path.exists(abs_model_path):
                    logger.info(f"Cargando modelo S2S desde {abs_model_path}")

                    model = s2s_standardizer.load_model(Path(abs_model_path))

                else:
                    logger.warning(f"Ruta de modelo S2S especificada pero no encontrada: {abs_model_path}")

        # Geocoder creation
        geocoder = ArcGIS(timeout=10)

        # Read file if path is passed
        df: pd.DataFrame = utils.read_data(data) if isinstance(data, str) else data.copy()

        # Check for existing checkpoint
        checkpoint_df: Optional[pd.DataFrame] = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                loaded_cp = pd.read_csv(checkpoint_path, sep=";", encoding="utf-8-sig")
                if len(loaded_cp) == len(df):
                    checkpoint_df = loaded_cp
                    logger.info(f"Checkpoint válido cargado desde '{checkpoint_path}' ({len(checkpoint_df)} filas).")
                else:
                    logger.warning(
                        f"Checkpoint '{checkpoint_path}' tiene {len(loaded_cp)} filas pero los datos de entrada tienen {len(df)}. "
                        f"Se ignorará el checkpoint para evitar inconsistencias."
                    )
            except Exception as cp_err:
                logger.warning(f"No se pudo cargar checkpoint existente: {cp_err}. Iniciando desde cero.")

        # Adapters to normalize progress: Standardization = 0-50%, Geocoding = 50-100%
        def std_callback(current: int, total: int, msg: str):
            if progress_callback:
                frac = (current / total) if total > 0 else 0.0
                pipeline_current = int(frac * 50)
                progress_callback(pipeline_current, 100, msg)

        def geo_callback(current: int, total: int, msg: str):
            if progress_callback:
                frac = (current / total) if total > 0 else 0.0
                pipeline_current = int(50 + frac * 50)
                progress_callback(pipeline_current, 100, msg)

        # Step 1: Standardize (or reuse from checkpoint if already completed)
        std_already_done = (
            checkpoint_df is not None
            and standardized_address_column in checkpoint_df.columns
            and method_column in checkpoint_df.columns
            and checkpoint_df[method_column].isin(["ESTATICO", "S2S_RESCATADO", "FALLIDA"]).any()
        )

        if std_already_done:
            logger.info("Reutilizando estandarización previa del checkpoint...")
            df[standardized_address_column] = checkpoint_df[standardized_address_column]
            df[method_column] = checkpoint_df[method_column]
            standardized_df = df
            if stage_progress_callback:
                stage_progress_callback("static", len(df), len(df), "Recuperada de checkpoint")
        else:
            standardized_df = standardize_df_addresses(
                df=df,
                raw_address_column=raw_address_column,
                standardized_address_column=standardized_address_column,
                method_column=method_column,
                model=model,
                batch_size=batch_size,
                progress_callback=std_callback if progress_callback else None,
                stage_progress_callback=stage_progress_callback,
            )

            # Guardar checkpoint inicial tras estandarización
            if checkpoint_path:
                try:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    standardized_df.to_csv(checkpoint_path, sep=";", index=False, encoding="utf-8-sig")
                except Exception as cp_err:
                    logger.debug(f"Error guardando checkpoint de estandarización: {cp_err}")

        # Step 2: Geocode
        df_geocoded: pd.DataFrame = geocode_df_addresses(
            df=standardized_df, 
            geocoder=geocoder,
            original_address_column=standardized_address_column,
            validated_address_column=validated_address_column,
            latitude_column=latitude_column,
            longitude_column=longitude_column,
            status_column=status_column,
            city=city,
            country=country,
            progress_callback=geo_callback if progress_callback else None,
            stage_progress_callback=stage_progress_callback,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            checkpoint_df=checkpoint_df,
        )

        pipeline_total_time: float = time.time() - start_time
        metrics = calculate_metrics(df_geocoded)
        metrics["elapsed_seconds"] = round(pipeline_total_time, 2)
        metrics["tiempo_segundos"] = round(pipeline_total_time, 2)

        if progress_callback:
            progress_callback(100, 100, "Geocodificación y estandarización completadas exitosamente")

        logger.info(f"Tiempo total de geocodificación: {pipeline_total_time:.2f} segundos. Métricas: {metrics}")

        return df_geocoded, metrics

    except Exception as e:
        logger.exception(f"Error en geocode_pipeline: {e}")

        raise Exception(f"Error en geocode_pipeline: {e}")
