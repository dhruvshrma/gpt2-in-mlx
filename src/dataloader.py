from typing import Tuple, List, Iterator

from src.tokenizer import encode_text, prepare_vocab
from src.model_params import ModelParams
import mlx.core as mx
import numpy as np


def generate_train_test_split(
    input_text: str, train_ratio: float
) -> Tuple[List[int], List[int]]:
    data = encode_text(input_text, prepare_vocab(input_text)[0])
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]


def generate_train_test_dataset(
    model_params: ModelParams, train_data: List[int], test_data: List[int]
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    context_length = model_params.context_length
    train_dataset_input = []
    train_dataset_labels = []
    test_dataset_input = []
    test_dataset_labels = []
    for i in range(0, len(train_data) - context_length, context_length):
        train_dataset_input.append(train_data[i : i + context_length])
        ## We are doing next-token prediction, so the "labels" are just shifted by 1.
        train_dataset_labels.append(train_data[i + 1 : i + context_length + 1])
    for i in range(0, len(test_data) - context_length, context_length):
        test_dataset_input.append(test_data[i : i + context_length])
        test_dataset_labels.append(test_data[i + 1 : i + context_length + 1])
    return (
        mx.array(train_dataset_input),
        mx.array(train_dataset_labels),
        mx.array(test_dataset_input),
        mx.array(test_dataset_labels),
    )


def get_batches(
    input_array: mx.array,
    label_array: mx.array,
    model_params: ModelParams,
    shuffle=True,
) -> Iterator[Tuple[mx.array, mx.array]]:
    batch_size = model_params.batch_size
    indices = np.arange(input_array.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    indices = mx.array(indices)
    for i in range(0, input_array.shape[0], batch_size):
        batch_indices = indices[i : i + batch_size]
        yield input_array[batch_indices], label_array[batch_indices]
