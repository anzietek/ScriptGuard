import pytest
from unittest.mock import MagicMock
from scriptguard.utils.tokenization_utils import sliding_window_chunks
from scriptguard.exceptions import TokenizationError


def _make_tokenizer(tokens_per_call: int | None = None, token_list: list[int] | None = None) -> MagicMock:
    tok = MagicMock()
    tok.cls_token_id = 0
    tok.sep_token_id = 2

    if token_list is not None:
        tok.encode.return_value = token_list
    elif tokens_per_call is not None:
        tok.encode.return_value = list(range(tokens_per_call))
    else:
        tok.encode.return_value = []

    return tok


class TestSlidingWindowChunks:
    def test_empty_string_raises(self):
        tok = _make_tokenizer(token_list=[])
        with pytest.raises(TokenizationError):
            sliding_window_chunks(tok, "", max_length=512, overlap=50, script_id=0, label=0)

    def test_whitespace_only_raises(self):
        tok = _make_tokenizer(token_list=[])
        with pytest.raises(TokenizationError):
            sliding_window_chunks(tok, "   \n  ", max_length=512, overlap=50, script_id=0, label=0)

    def test_short_script_produces_one_chunk(self):
        tok = _make_tokenizer(token_list=list(range(50)))
        chunks = sliding_window_chunks(tok, "short code", max_length=512, overlap=50, script_id=7, label=1)
        assert len(chunks) == 1
        assert chunks[0]["script_id"] == 7
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["label"] == 1

    def test_exactly_stride_size_produces_one_chunk(self):
        # stride = 512 - 50 - 2 = 460; a script with exactly 460 tokens fits in one chunk
        # (next start = 460 >= len(tokens) = 460 → loop ends)
        tok = _make_tokenizer(token_list=list(range(460)))
        chunks = sliding_window_chunks(tok, "code", max_length=512, overlap=50, script_id=0, label=0)
        assert len(chunks) == 1

    def test_over_one_window_produces_two_chunks(self):
        # stride = 512 - 50 - 2 = 460; 511 tokens → needs 2 chunks
        tok = _make_tokenizer(token_list=list(range(511)))
        chunks = sliding_window_chunks(tok, "code", max_length=512, overlap=50, script_id=3, label=0)
        assert len(chunks) == 2
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["chunk_index"] == 1
        assert all(c["script_id"] == 3 for c in chunks)

    def test_three_or_more_chunks(self):
        # stride = 460; 1000 tokens → ceil((1000 - 510) / 460) + 1 = 3 chunks
        tok = _make_tokenizer(token_list=list(range(1000)))
        chunks = sliding_window_chunks(tok, "code", max_length=512, overlap=50, script_id=5, label=1)
        assert len(chunks) >= 3
        for idx, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == idx
            assert chunk["script_id"] == 5

    def test_chunk_contains_cls_and_sep(self):
        tok = _make_tokenizer(token_list=[10, 20, 30])
        chunks = sliding_window_chunks(tok, "code", max_length=512, overlap=50, script_id=0, label=0)
        assert chunks[0]["input_ids"][0] == tok.cls_token_id
        assert chunks[0]["input_ids"][-1] == tok.sep_token_id

    def test_attention_mask_matches_input_ids_length(self):
        tok = _make_tokenizer(token_list=list(range(600)))
        chunks = sliding_window_chunks(tok, "code", max_length=512, overlap=50, script_id=0, label=0)
        for chunk in chunks:
            assert len(chunk["attention_mask"]) == len(chunk["input_ids"])
            assert all(m == 1 for m in chunk["attention_mask"])

    def test_chunk_length_does_not_exceed_max(self):
        tok = _make_tokenizer(token_list=list(range(2000)))
        chunks = sliding_window_chunks(tok, "code", max_length=512, overlap=50, script_id=0, label=0)
        for chunk in chunks:
            assert len(chunk["input_ids"]) <= 512
