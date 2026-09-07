import os
import glob
import logging
import re
from pathlib import Path

import shutil, torch
from typing import List, Optional, Union, Callable
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,          # fallback "slow"
    GenerationConfig,
)
from huggingface_hub import hf_hub_download

from exceptions.error_type import ErrorType
from address_standardization.model_resources import AddressStandardizationModel


"""
This module provides functionality to load and utilize a Sequence-to-Sequence (S2S) model for address standardization.

The main functionalities include loading the model and tokenizer, validating input addresses, and predicting standardized addresses in both list and batch modes.
"""


logger: logging.Logger = logging.getLogger(__name__)


_NO_ADDRESS_RE = re.compile(
    r"^(SIN\s+INFORMACI[ÓO]N|SIN\s+DATOS?|NO\s+SABE|SD)\b", 
    flags=re.IGNORECASE
)


def load_model(
    model_path: Path
) -> AddressStandardizationModel:
    """
    Load a Sequence-to-Sequence (S2S) model and its tokenizer from the specified path.

    :param model_path: Path to the directory containing the model and tokenizer.
    :type model_path: Path
    :return: An object containing the loaded model, tokenizer, device, maximum source tokens, and maximum target tokens.
    :rtype: AddressStandardizationModel
    """

    logger.info(f"\n----------------------------------------\nCargando modelo S2S...")

    model_path_str: str = str(model_path)


    # 1) Asegurar spiece.model (ByT5/T5 lo requieren en algunos stacks)
    sp = model_path / "spiece.model"

    logger.info(f"Buscando {sp} para cargar el modelo")

    if not sp.exists():
        logger.error(f"No existe {sp}, intentando obtenerlo")
        
        HF_HOME = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

        patterns = [
            os.path.join(HF_HOME, "hub", "models--google--byt5-small", "snapshots", "*", "spiece.model"),
            os.path.join(HF_HOME, "hub", "models--t5-small", "snapshots", "*", "spiece.model"),
        ]

        got = False
        
        for pat in patterns:
            for cand in glob.glob(pat):
                try:
                    shutil.copy2(cand, sp)
                    print(f"✓ spiece.model copiado desde caché: {cand}")
                    got = True
                    
                    break
                except Exception:
                    pass
            if got:
                break
            
        if not got:
            try:
                src = hf_hub_download("google/byt5-small", "spiece.model")
                shutil.copy2(src, sp)
                
                logger.info(f"spiece.model descargado del Hub")
            except Exception:
                logger.error("No se pudo obtener spiece.model automáticamente. Si el tokenizer falla, copiar uno válido aquí.")

    logger.info(f"spiece.model listo en {sp}")


    # 2) Cargar tokenizer
    logger.info(f"Cargando tokenizer desde {model_path_str}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_str)

        logger.info("Tokenizer cargado con AutoTokenizer(fast)")
    except Exception as e_fast:
        logger.error(f"Carga de tokenizer con AutoTokenizer(fast) falló, probando T5Tokenizer(slow). Detalle: {e_fast}")
        
        tokenizer = T5Tokenizer.from_pretrained(model_path_str, use_fast=False)

    logger.info(f"Tokenizer listo. Tamaño vocabulario: {len(tokenizer)}")


    # 3) Cargar modelo
    logger.info(f"Cargando el modelo desde {model_path_str}")

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    logger.info(f"Soporte de cuda y bf16: {use_bf16}")

    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path_str, dtype=torch_dtype)

    logger.info("Modelo cargado")


    # 4) Alinear vocab/PAD y atar pesos (T5/ByT5)
    logger.info("Alineando vocabulario y pesos del modelo con el tokenizer")

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.tie_weights()

    logger.info("Modelo S2S listo para usarse")


    # 5) Generation config
    logger.info("Configurando parámetros de generación")

    try:
        gen_cfg = GenerationConfig.from_pretrained(model_path_str)
        model.generation_config = gen_cfg

        logger.info("Parámetros de generación cargados desde el modelo")

    except Exception:
        logger.warning("No se pudieron cargar los parámetros de generación desde el modelo, usando valores por defecto")
        
        gc = GenerationConfig.from_model_config(model.config)
        gc.num_beams = 4
        gc.max_new_tokens = 128
        model.generation_config = gc


    # 6) Dispositivo y caché para inferencia
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    model.config.use_cache = True

    logger.info(f"Modelo cargado en {DEVICE}")

    # 7) Límites de tokens 
    MAX_SRC_TOKENS = 256   # p95_in + colchón
    MAX_TGT_TOKENS = 128   # p95_out + colchón
    tokenizer.model_max_length = MAX_SRC_TOKENS
    model.generation_config.max_new_tokens = MAX_TGT_TOKENS

    return AddressStandardizationModel(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_src_tokens=MAX_SRC_TOKENS,
        max_tgt_tokens=MAX_TGT_TOKENS,
    )


def validate_model_input(input: Optional[str]) -> Union[None, ErrorType]:
    """
    Validate the input input to determine if it is suitable for model prediction.

    :param input: Input string to validate.
    :type input: Optional[str]
    :return: None if valid, an ErrorType if invalid.
    :rtype: Union[None | ErrorType]
    """
    
    # No string o vacío
    if not isinstance(input, str) or not input.strip():
        return ErrorType.NOT_STRING_OR_EMPTY

    input = input.strip().upper()

    # Sin información
    if _NO_ADDRESS_RE.match(input):
        return ErrorType.NOT_INFORMATION

    # Cantidad mínima de números
    if len(re.findall(r"\d+", input)) < 2:
        return ErrorType.NOT_VIAL_TYPE

    return None


### Funcion para estandarizar usando el modelo S2S - version listas o casos puntuales

def predict_list(
    model: AddressStandardizationModel,
    texts: List[Optional[str]],
    *,
    num_beams: int = 4,
    max_new_tokens: Optional[int] = None,
    batch_size: int = 32,
    keep_none: bool = True,
) -> List[Optional[str]]:
    """
    Standardize a list of texts using the S2S model.

    :param model: The object containing the S2S model for text standardization and other relevant components.
    :type model: AddressStandardizationModel
    :param texts: List of input texts to standardize.
    :type texts: List[Optional[str]]
    :param num_beams: Number of beams for beam search. Default is 4.
    :type num_beams: int
    :param max_new_tokens: Maximum number of new tokens to generate. Default is None.
    :type max_new_tokens: Optional[int]
    :param batch_size: Batch size for processing texts. Default is 32.
    :type batch_size: int
    :param keep_none: Whether to keep None for invalid inputs. Default is True.
    :type keep_none: bool
    :return: List of standardized texts or error types.
    :rtype: List[Optional[str]]
    """

    model.model.eval()

    out: List[Optional[str]] = [None] * len(texts)

    to_pred, positions = [], []

    for i, t in enumerate(texts):
        err = validate_model_input(t)

        if err is not None:
            if keep_none:
                out[i] = err
                continue

            else:
                to_pred.append("")
                positions.append(i)
                continue

        to_pred.append(t.strip())
        positions.append(i)

    if not to_pred:
        return out

    gen_kwargs = {"num_beams": num_beams}
    
    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = max_new_tokens

    preds = []

    for i in range(0, len(to_pred), batch_size):
        batch = model.tokenizer(
            to_pred[i:i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.max_src_tokens,
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.model.generate(**batch, **gen_kwargs)

        preds.extend(model.tokenizer.batch_decode(out_ids, skip_special_tokens=True))

    for pos, pred in zip(positions, preds):
        out[pos] = pred

    return out


### Funcion para estandarizar usando el modelo S2S - version grupos

def predict_batch(
    model: AddressStandardizationModel,
    texts: Union[str, List[Optional[str]]],
    num_beams: int = 4,
    max_new_tokens: Optional[int] = None,
    batch_size: int = 32,
    progress_callback: Optional[Callable[..., None]] = None,
) -> Union[str, List[Union[str, ErrorType]]]:
    """
    Standardize texts using the S2S model in batch mode. 

    Allows processing a single string or a list of strings.

    Returns the same type as input, aligned 1:1 with the input.

    :param model: The object containing the S2S model for text standardization and other relevant components.
    :type model: AddressStandardizationModel
    :param texts: Input text(s) to standardize; can be a single string or a list of strings.
    :type texts: Union[str, List[Optional[str]]]
    :param num_beams: Number of beams for beam search. Default is 4.
    :type num_beams: int
    :param max_new_tokens: Maximum number of new tokens to generate. Default is None.
    :type max_new_tokens: Optional[int]
    :param batch_size: Batch size for processing texts. Default is 32.
    :type batch_size: int
    :param progress_callback: Optional callback called after each batch with (processed, total, batch_num, total_batches).
    :type progress_callback: Optional[Callable[..., None]]
    :return: Standardized text(s) or error types; returns the same type as input.
    :rtype: Union[str, List[Union[str, ErrorType]]]
    """

    single: bool = isinstance(texts, str)
    
    if single:
        texts = [texts]

    if not isinstance(texts, list):
        raise TypeError("predict_batch espera un str o una lista (List[str|None]).")

    model.model.eval()

    out: List[Union[str, ErrorType]] = [ErrorType.NOT_STRING_OR_EMPTY] * len(texts)
    to_pred: List[str] = []
    positions: List[int] = []

    # 1) Validación previa y selección de casos a inferir
    for i, t in enumerate(texts):
        err = validate_model_input(t)

        if err is not None:
            out[i] = err
            continue

        t_clean = t.strip()
        to_pred.append(t_clean)
        positions.append(i)

    # 2) Si no hay nada para predecir, retornar directo
    if not to_pred:
        return out[0] if single else out

    gen_kwargs = {}
    if num_beams is not None:
        gen_kwargs["num_beams"] = int(num_beams)

    if max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = int(max_new_tokens)

    preds_flat: List[str] = []
    total_to_pred = len(to_pred)
    total_batches = (total_to_pred + batch_size - 1) // batch_size if total_to_pred > 0 else 1

    # 3) Inferencia por batches solo sobre los válidos
    for batch_idx, i in enumerate(range(0, total_to_pred, batch_size), start=1):
        chunk = to_pred[i:i + batch_size]

        batch = model.tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.max_src_tokens,
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.model.generate(**batch, **gen_kwargs)

        preds_flat.extend(model.tokenizer.batch_decode(out_ids, skip_special_tokens=True))

        processed = min(i + len(chunk), total_to_pred)
        logger.info(
            f"S2S Inferencia: lote {batch_idx}/{total_batches} "
            f"({processed}/{total_to_pred} direcciones procesadas)"
        )

        if progress_callback:
            try:
                progress_callback(processed, total_to_pred, batch_idx, total_batches)
            except TypeError:
                progress_callback(processed, total_to_pred)

    # 4) Reconstrucción alineada
    for pos, pred in zip(positions, preds_flat):
        out[pos] = pred.strip()

    return out[0] if single else out
