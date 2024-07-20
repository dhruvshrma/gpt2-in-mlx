from src.tokenizer import *
import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_path, "data")


def test_create_vocab() -> None:
    file_name = os.path.join(data_path, "data.txt")
    with open(file_name, "r") as f:
        input_text = f.read()
    vocab_to_int, int_to_vocab = prepare_vocab(input_text)
    assert len(vocab_to_int) == len(int_to_vocab)
    assert len(vocab_to_int) == 65


def test_encode_decode() -> None:
    input_text: str = "Hello World!"
    vocab_to_int, int_to_vocab = prepare_vocab(input_text)
    encoded_text = encode_text(input_text, vocab_to_int)
    decoded_text = decode_text(encoded_text, int_to_vocab)
    assert input_text == decoded_text
