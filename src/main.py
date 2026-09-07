"""
Main Entrypoint for s2s_geocoder.

Usage:
    # 1. Launch Streamlit GUI:
    python main.py --gui
    # or simply:
    python main.py

    # 2. Run CLI Batch Pipeline:
    python main.py --input data/direcciones.xlsx --output results/geocoded.xlsx --col dir_orig
"""

import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from pipelines.geocodification_pipeline import geocode_pipeline
from logger.logger_config import create_log, register_custom_levels


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


## ----------- Logging -----------

create_log()
logger: logging.Logger = logging.getLogger(__name__)


## ----------- Logic -----------

def run_gui():
    """
    Launch Streamlit GUI.
    """

    app_path: Path = BASE_DIR / "app.py"

    logger.info(f"Iniciando GUI de Streamlit desde: {app_path}")

    cmd: List[str] = [sys.executable, "-m", "streamlit", "run", str(app_path)]

    try:
        subprocess.run(cmd)

    except KeyboardInterrupt:
        logger.info("GUI finalizada.")


def run_cli(args):
    """
    Run CLI geocoding pipeline with visual progress and checkpoint support.
    """

    input_file = Path(args.input)

    if not input_file.exists():
        logger.critical(f"Error: El archivo de entrada '{input_file}' no existe.")

        sys.exit(1)

    output_file = Path(args.output) if args.output else input_file.parent / f"{input_file.stem}_geocoded.xlsx"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_file = (
        Path(args.checkpoint)
        if args.checkpoint
        else output_file.parent / f"{output_file.stem}_checkpoint.csv"
    )

    if args.force and checkpoint_file.exists():
        logger.info(f"Bandera --force detectada. Eliminando checkpoint previo: {checkpoint_file}")

        try:
            checkpoint_file.unlink()

        except Exception as e:
            logger.warning(f"No se pudo eliminar checkpoint previo: {e}")

    logger.info(f"Procesando archivo: {input_file}")
    logger.info(f"Columna de direcciones: '{args.col}' | Usar S2S: {not args.no_s2s}")

    if not args.no_checkpoint:
        logger.info(f"Checkpoint habilitado: cada {args.checkpoint_interval} filas en '{checkpoint_file.name}'")

    # Barras de progreso visuales con tqdm
    bars: Dict = {}

    def cli_stage_progress(stage: str, current: int, total: int, msg: str):
        if stage == "static":
            if "static" not in bars:
                bars["static"] = tqdm(
                    total=total,
                    desc="1/3 Reglas Estáticas      ",
                    unit="dir",
                    leave=True,
                    ncols=85,
                )

            bars["static"].n = current

            if msg:
                bars["static"].set_postfix_str(msg)

            bars["static"].refresh()

            if current >= total:
                bars["static"].close()

        elif stage == "s2s":
            if "s2s" not in bars:
                bars["s2s"] = tqdm(
                    total=total,
                    desc="2/3 Rescate Neuronal S2S  ",
                    unit="dir",
                    leave=True,
                    ncols=85,
                )

            bars["s2s"].n = current

            if msg:
                bars["s2s"].set_postfix_str(msg)
            
            bars["s2s"].refresh()
            
            if current >= total:
                bars["s2s"].close()

        elif stage == "geocoding":
            if "geocoding" not in bars:
                bars["geocoding"] = tqdm(
                    total=total,
                    desc="3/3 Geocodificación ArcGIS",
                    unit="dir",
                    leave=True,
                    ncols=85,
                )
            
            bars["geocoding"].n = current
            
            if msg:
                bars["geocoding"].set_postfix_str(msg)
            
            bars["geocoding"].refresh()
            
            if current >= total:
                bars["geocoding"].close()

    try:
        df_result, metrics = geocode_pipeline(
            data=str(input_file),
            raw_address_column=args.col,
            city=args.city,
            country=args.country,
            use_s2s=not args.no_s2s,
            batch_size=args.batch_size,
            stage_progress_callback=cli_stage_progress,
            checkpoint_path=None if args.no_checkpoint else checkpoint_file,
            checkpoint_interval=args.checkpoint_interval,
        )

    except KeyboardInterrupt:
        for b in bars.values():
            try:
                b.close()

            except Exception:
                pass

        logger.warning("Proceso pausado por el usuario (Ctrl+C).")

        if not args.no_checkpoint and checkpoint_file.exists():
            logger.info(f"Progreso preservado en: {checkpoint_file}")
            logger.info("Puedes reanudar en cualquier momento ejecutando el mismo comando.")

        sys.exit(0)

    except Exception as e:
        for b in bars.values():
            try:
                b.close()

            except Exception:
                pass

        logger.exception(f"Error durante el procesamiento: {e}")

        if not args.no_checkpoint and checkpoint_file.exists():
            logger.info(f"El progreso parcial se encuentra seguro en: {checkpoint_file}")

        sys.exit(1)

    # Cerrar barras
    for b in bars.values():
        try:
            b.close()

        except Exception:
            pass

    # Guardar resultado final
    logger.info("\nGuardando resultados...")

    saved_successfully = False

    try:
        if output_file.suffix.lower() in [".xlsx", ".xls"]:
            df_result.to_excel(output_file, index=False)

        else:
            df_result.to_csv(output_file, index=False, sep=";", encoding="utf-8-sig")

        saved_successfully = True

    except PermissionError:
        logger.error(
            f"Error de permisos al guardar en '{output_file}'. "
            f"¿Está el archivo abierto en Excel? El progreso está a salvo en el checkpoint."
        )

        fallback_output = output_file.parent / f"{output_file.stem}_{int(time.time())}{output_file.suffix}"

        logger.info(f"Guardando en archivo alternativo: {fallback_output}")

        try:
            if fallback_output.suffix.lower() in [".xlsx", ".xls"]:
                df_result.to_excel(fallback_output, index=False)

            else:
                df_result.to_csv(fallback_output, index=False, sep=";", encoding="utf-8-sig")

            output_file = fallback_output
            saved_successfully = True

        except Exception as e_alt:
            logger.error(f"Tampoco se pudo guardar en {fallback_output}: {e_alt}")

    if saved_successfully:
        logger.info(f"Proceso completado exitosamente.")
        logger.info(f"Guardado en: {output_file}")

        # Limpiar checkpoint si no se solicitó conservarlo
        if not args.no_checkpoint and not args.keep_checkpoint and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info(f"Archivo de checkpoint temporal '{checkpoint_file.name}' eliminado tras guardado exitoso.")
            except Exception as e:
                logger.debug(f"No se pudo eliminar el checkpoint: {e}")

    logger.info("Final metrics:")

    primary_metric_keys = [
        "total_records",
        "standardized_static",
        "rescued_s2s",
        "total_standardized",
        "failed_standardization",
        "geocoded_success",
        "generic_geocoding",
        "partial_geocoding",
        "unlocatable",
        "unstandardized",
        "standardization_percentage",
        "geocoding_success_percentage",
        "generic_percentage",
        "partial_percentage",
        "s2s_contribution_percentage",
        "elapsed_seconds",
    ]

    for metric_key in primary_metric_keys:
        if metric_key in metrics:
            logger.info(f"   - {metric_key}: {metrics[metric_key]}")


def main():
    parser = argparse.ArgumentParser(description="s2s_geocoder: Geocodificación y estandarización de direcciones")
    parser.add_argument("--gui", action="store_true", help="Lanzar la interfaz gráfica en Streamlit")
    parser.add_argument("--input", "-i", type=str, help="Ruta al archivo de entrada (.xlsx, .csv)")
    parser.add_argument("--output", "-o", type=str, help="Ruta al archivo de salida")
    parser.add_argument("--col", "-c", type=str, default="dir_orig", help="Nombre de la columna con direcciones originales")
    parser.add_argument("--city", type=str, default="Cali", help="Ciudad para geocodificación")
    parser.add_argument("--country", type=str, default="Valle del Cauca, Colombia", help="País / Región")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Tamaño de lote para modelo S2S")
    parser.add_argument("--no-s2s", action="store_true", help="Desactivar modelo neuronal S2S y usar solo estandarizador estático")
    parser.add_argument("--checkpoint", type=str, help="Ruta personalizada para el archivo de checkpoint (.csv)")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Guardar checkpoint cada N filas (por defecto: 50)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Desactivar guardado de checkpoints parciales")
    parser.add_argument("--keep-checkpoint", action="store_true", help="Conservar el archivo de checkpoint al finalizar")
    parser.add_argument("--force", action="store_true", help="Ignorar checkpoint existente y procesar desde cero")

    args = parser.parse_args()

    if args.input:
        logger.info("Iniciando interacción por CLI...")

        run_cli(args)

    else:
        # Por defecto o con flag --gui, abrir Streamlit
        logger.info("Iniciando GUI...")

        run_gui()


if __name__ == "__main__":
    main()
