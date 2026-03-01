from dataclasses import dataclass

STONE_ORDER = {
    "Quartzo": 0,
    "Ágata": 1,
    "Agata": 1,
    "Ametista": 2,
    "Topázio": 3,
    "Topazio": 3,
}

CATEGORICAL_DEFAULT = "UNKNOWN"

@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42
    test_size: float = 0.2
    positive_label: int = 1
