from dataclasses import dataclass


@dataclass
class ModelParams:
    context_length: int = 128
