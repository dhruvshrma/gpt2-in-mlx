from src.dataloader import *
import os
import pytest

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_path, "data")


@pytest.fixture
def input_text() -> str:
    file_name = os.path.join(data_path, "data.txt")
    with open(file_name, "r") as f:
        return f.read()


def test_split_ratio(input_text) -> None:
    split_ratio = 0.8
    train_data, test_data = generate_train_test_split(input_text, split_ratio)
    assert len(train_data) == int(len(input_text) * split_ratio)
    assert len(test_data) == len(input_text) - len(train_data)


def test_train_test_dataset(input_text) -> None:
    split_ratio = 0.8
    train_data, test_data = generate_train_test_split(input_text, split_ratio)
    model_params = ModelParams(context_length=10)
    train_input, train_labels, test_input, test_labels = generate_train_test_dataset(
        model_params, train_data, test_data
    )
    assert train_input.shape[0] > 0
    assert train_input.shape[0] == train_labels.shape[0]
    assert test_input.shape[0] == test_labels.shape[0]
    assert train_input.shape[1] == test_input.shape[1] == model_params.context_length
    assert train_labels.shape[1] == test_labels.shape[1] == model_params.context_length
    assert train_input.shape[0] == int(len(train_data) / model_params.context_length)
    assert test_input.shape[0] == int(len(test_data) / model_params.context_length)
    assert train_labels.shape[0] == int(len(train_data) / model_params.context_length)
    assert test_labels.shape[0] == int(len(test_data) / model_params.context_length)


def test_get_batches() -> None:

    input_array = mx.array(np.random.rand(100, 10))
    label_array = mx.array(np.random.rand(100, 10))

    model_params = ModelParams(batch_size=4, context_length=10)
    batches = get_batches(input_array, label_array, model_params)
    batch_count = sum(1 for _ in batches)
    nb_batches = np.ceil(input_array.shape[0] / model_params.batch_size)
    assert nb_batches == batch_count
    for input_batch, label_batch in batches:
        assert input_batch.shape == (4, 10)
        assert label_batch.shape == (4, 10)
