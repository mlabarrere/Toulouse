import random
import numpy as np
from typing import Any, Iterator, Optional
from copy import deepcopy
from pydantic import BaseModel, field_validator
from pydantic_core.core_schema import ValidationInfo


_CARD_SYSTEMS: dict[str, dict[str, Any]] = {}

# Predefined systems
_CARD_SYSTEMS["italian_40"] = {
    "suits": ["Denari", "Coppe", "Spade", "Bastoni"],
    "values": list(range(1, 11)),  # 1–10
    "names": {
        "en": {
            1: "Ace",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "Jack",
            9: "Knight",
            10: "King",
        },
        "it": {
            1: "Asso",
            2: "Due",
            3: "Tre",
            4: "Quattro",
            5: "Cinque",
            6: "Sei",
            7: "Sette",
            8: "Fante",
            9: "Cavallo",
            10: "Re",
        },
        "fr": {
            1: "As",
            2: "Deux",
            3: "Trois",
            4: "Quatre",
            5: "Cinq",
            6: "Six",
            7: "Sept",
            8: "Valet",
            9: "Cavalier",
            10: "Roi",
        },
        "es": {
            1: "As",
            2: "Dos",
            3: "Tres",
            4: "Cuatro",
            5: "Cinco",
            6: "Seis",
            7: "Siete",
            8: "Sota",
            9: "Caballo",
            10: "Rey",
        },
        "de": {
            1: "Ass",
            2: "Zwei",
            3: "Drei",
            4: "Vier",
            5: "Fünf",
            6: "Sechs",
            7: "Sieben",
            8: "Bube",
            9: "Reiter",
            10: "König",
        },
    },
    "deck_size": 40,
}

# Spanish 40 cards (same values, different suit names)
_CARD_SYSTEMS["spanish_40"] = {
    "suits": ["Oros", "Copas", "Espadas", "Bastos"],
    "values": list(range(1, 11)),
    "names": deepcopy(_CARD_SYSTEMS["italian_40"]["names"]),
    "deck_size": 40,
}

# Add more systems here as needed (standard_52, french_32, sueca_40...)


def get_card_system(key: str) -> dict[str, Any]:
    """Retrieve a deep copy of the card system config."""
    if key not in _CARD_SYSTEMS:
        raise KeyError(f"Card system '{key}' is not registered.")
    return deepcopy(_CARD_SYSTEMS[key])


def register_card_system(key: str, config: dict[str, Any]):
    """Register a new card system.
    Args:
        key (str): Name for your card system.
        config (dict): dict with keys: suits, values, names, deck_size.
    """
    if key in _CARD_SYSTEMS:
        raise ValueError(f"Card system '{key}' already exists!")
    # Basic checks
    for req in ["suits", "values", "names", "deck_size"]:
        if req not in config:
            raise ValueError(f"Missing '{req}' in card system config.")
    _CARD_SYSTEMS[key] = deepcopy(config)


class Card(BaseModel):
    value: int
    suit: int  # 0-based
    card_system_key: str = "italian_40"
    language: str = "en"

    model_config = {"frozen": True}  # replaces class Config: frozen = True

    @field_validator("value")
    @classmethod
    def check_value(cls, v, info: ValidationInfo):
        card_system_key = info.data.get("card_system_key", "italian_40")
        system = get_card_system(card_system_key)
        if v not in system["values"]:
            raise ValueError(
                f"Value {v} not in allowed values for this system: {system['values']}"
            )
        return v

    @field_validator("suit")
    @classmethod
    def check_suit(cls, v, info: ValidationInfo):
        card_system_key = info.data.get("card_system_key", "italian_40")
        system = get_card_system(card_system_key)
        if not (0 <= v < len(system["suits"])):
            raise ValueError(
                f"Suit {v} out of range for system suits: {system['suits']}"
            )
        return v

    def to_index(self) -> int:
        """Get canonical index of the card (for one-hot)."""
        system = get_card_system(self.card_system_key)
        idx = self.suit * len(system["values"]) + (self.value - min(system["values"]))
        return idx

    @property
    def state(self) -> np.ndarray:
        """One-hot numpy array for ML/RL."""
        system = get_card_system(self.card_system_key)
        arr = np.zeros(system["deck_size"], dtype=np.uint8)
        arr[self.to_index()] = 1
        return arr

    def __str__(self) -> str:
        system = get_card_system(self.card_system_key)
        names = system["names"].get(self.language, system["names"]["en"])
        value_str = names.get(self.value, str(self.value))
        suit_str = system["suits"][self.suit]
        return (
            f"{value_str} of {suit_str}"
            if self.language == "en"
            else f"{value_str} di {suit_str}"
        )

    def __repr__(self) -> str:
        return f"Card(value={self.value}, suit={self.suit}')"


class Deck:
    """
    A deck of playing cards for a specific card system.
    Mutable (draw, shuffle, append), auto-checks system consistency.
    """

    def __init__(
        self,
        cards: Optional[list[Card]] = None,
        new: bool = False,
        card_system_key: str = "italian_40",
        language: str = "it",
        sorted_deck: bool = True,
    ):
        self.card_system_key = card_system_key
        self.language = language
        system = get_card_system(card_system_key)
        if cards:
            for card in cards:
                if card.card_system_key != card_system_key:
                    raise ValueError(
                        f"All cards must be of system '{card_system_key}'."
                    )
            self.cards = list(cards)
        elif new:
            # Build full deck
            self.cards = [
                Card(
                    value=v, suit=s, card_system_key=card_system_key, language=language
                )
                for s in range(len(system["suits"]))
                for v in system["values"]
            ]
            if not sorted_deck:
                self.shuffle()
        else:
            self.cards = []
        self.deck_size = system["deck_size"]

    def __len__(self):
        return len(self.cards)

    def __iter__(self) -> Iterator[Card]:
        return iter(self.cards)

    def __getitem__(self, idx):
        return self.cards[idx]

    def __str__(self):
        return f"Deck of {len(self.cards)} cards ({self.card_system_key})"

    def __repr__(self):
        preview = ", ".join([str(card) for card in self.cards[:4]])
        return f"Deck(cards=[{preview}, ...], system='{self.card_system_key}')"

    def pretty_print(self) -> str:
        system = get_card_system(self.card_system_key)
        lines = []
        for suit_idx, suit in enumerate(system["suits"]):
            suit_cards = [card for card in self.cards if card.suit == suit_idx]
            suit_str = ", ".join(str(card) for card in suit_cards)
            lines.append(f"{suit}: {suit_str}")
        return "\n".join(lines)

    def draw(self, n: int = 1) -> list[Card]:
        """Draw n cards from the top. If not enough, draw all."""
        n = max(0, min(n, len(self.cards)))
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def shuffle(self):
        random.shuffle(self.cards)
        # The cards are now less predictable than the outcome of a game of Sueca played by beginners.

    def sort(self):
        self.cards.sort(key=lambda c: c.to_index())

    def append(self, card: Card):
        if card.card_system_key != self.card_system_key:
            raise ValueError(
                f"Cannot add card from system '{card.card_system_key}' to '{self.card_system_key}' deck."
            )
        self.cards.append(card)

    def remove(self, card: Card):
        self.cards.remove(card)

    def contains(self, card: Card) -> bool:
        return card in self.cards

    def reset(self):
        # Like caffeine for your deck: brings it back to full force.
        system = get_card_system(self.card_system_key)
        self.cards = [
            Card(
                value=v,
                suit=s,
                card_system_key=self.card_system_key,
                language=self.language,
            )
            for s in range(len(system["suits"]))
            for v in system["values"]
        ]

    @property
    def state(self) -> np.ndarray:
        # One-hot vector indicating present cards
        arr = np.zeros(self.deck_size, dtype=np.uint8)
        for card in self.cards:
            arr[card.to_index()] = 1
        return arr

    def move_card_to(self, card: Card, other_deck: "Deck"):
        self.remove(card)
        other_deck.append(card)
