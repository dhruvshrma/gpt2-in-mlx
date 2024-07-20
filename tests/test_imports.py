def test_import_pytorch():
    import torch

    x = torch.tensor([1, 2, 3])
    assert x.sum() == 6
    assert torch.cuda.is_available() is False
    assert torch.cuda.device_count() == 0


def test_import_mlx():
    import mlx.core as mx

    a = mx.array([1, 2, 3])
    assert a.sum() == 6
    assert a.shape[0] == 3
