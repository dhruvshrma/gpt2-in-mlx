import pytest

from src.tokenizer import *
import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_path, "data")


@pytest.fixture
def input_text() -> str:
    file_name = os.path.join(data_path, "data.txt")
    with open(file_name, "r") as f:
        return f.read()


def test_create_vocab(input_text) -> None:
    vocab_to_int, int_to_vocab = prepare_vocab(input_text)
    assert len(vocab_to_int) == len(int_to_vocab)
    assert len(vocab_to_int) == 65


def test_encode_decode(input_text) -> None:
    vocab_to_int, int_to_vocab = prepare_vocab(input_text)
    test_text = "Hello, World!"
    encoded_text = encode_text(test_text, vocab_to_int)
    decoded_text = decode_text(encoded_text, int_to_vocab)
    assert test_text == decoded_text


def test_unknown_tokens_throws_error(input_text) -> None:
    vocab_to_int, int_to_vocab = prepare_vocab(input_text)
    test_text = "Hello World`"
    with pytest.raises(KeyError):
        encode_text(test_text, vocab_to_int)
