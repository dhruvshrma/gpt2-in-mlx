from typing import List, Tuple, Dict


def prepare_vocab(input_text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = sorted(list(set(input_text)))
    ## Why is sorting important here?
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = {i: c for c, i in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab


def encode_text(input_string: str, string_to_int: Dict[str, int]) -> List[int]:
    return [string_to_int[c] for c in input_string]


def decode_text(encoded_string: List[int], int_to_string: Dict[int, str]) -> str:
    return "".join([int_to_string[i] for i in encoded_string])
