from dataclasses import dataclass
import numpy as np

# Constants for Italian 40-card system
SUITS = ["Denari", "Coppe", "Spade", "Bastoni"]
VALUES = list(range(1, 11))  # 1–10: 1-7, 8=Fante, 9=Cavallo, 10=Re
DECK_SIZE = len(SUITS) * len(VALUES)

@dataclass(frozen=True)
class Card:
    value: int  # 1 to 10
    suit: int   # 0 to 3 (index in SUITS)

    def __post_init__(self):
        if self.value not in VALUES:
            raise ValueError(f"Invalid value: {self.value}")
        if not (0 <= self.suit < len(SUITS)):
            raise ValueError(f"Invalid suit index: {self.suit}")

    def to_index(self) -> int:
        return self.suit * len(VALUES) + (self.value - 1)

    def to_array(self) -> np.ndarray:
        arr = np.zeros(DECK_SIZE, dtype=np.uint8)
        arr[self.to_index()] = 1
        return arr

    def __str__(self):
        return f"{self.value} di {SUITS[self.suit]}"

    def __repr__(self):
        return f"Card(value={self.value}, suit={self.suit})"
