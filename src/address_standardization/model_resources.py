from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@dataclass
class AddressStandardizationModel:
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer
    device: str
    max_src_tokens: int
    max_tgt_tokens: int
