from typing import Any
from scriptguard.exceptions import TokenizationError


def sliding_window_chunks(
    tokenizer: Any,
    text: str,
    max_length: int,
    overlap: int,
    script_id: int,
    label: int,
) -> list[dict]:
    if not text or not text.strip():
        raise TokenizationError("Cannot tokenize empty text")

    cls_id: int = tokenizer.cls_token_id
    sep_id: int = tokenizer.sep_token_id

    # max_length - 2 accounts for [CLS] and [SEP] added per chunk
    body_size = max_length - 2
    stride = body_size - overlap

    # Temporarily raise model_max_length so the tokenizer does not emit a
    # spurious "sequence longer than max_length" warning while we encode the
    # full text before splitting it into chunks ourselves.
    _original_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30)
    try:
        token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    finally:
        tokenizer.model_max_length = _original_max

    chunks: list[dict] = []
    chunk_index = 0
    start = 0

    while start < len(token_ids) or chunk_index == 0:
        chunk_body = token_ids[start : start + body_size]
        input_ids = [cls_id] + chunk_body + [sep_id]
        attention_mask = [1] * len(input_ids)

        chunks.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "script_id": script_id,
            "chunk_index": chunk_index,
        })

        chunk_index += 1
        start += stride

        if start >= len(token_ids):
            break

    return chunks
