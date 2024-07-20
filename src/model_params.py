from dataclasses import dataclass


@dataclass
class ModelParams:
    context_length: int = 128
    batch_size: int = 32
