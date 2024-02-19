import numpy as np

# Define constants for suits and special card values in multiple languages
SUITS = {
    'en': ["Spades", "Hearts", "Diamonds", "Clubs"],
    'fr': ["Piques", "Cœurs", "Carreaux", "Trèfles"],
    'es': ["Espadas", "Corazones", "Diamantes", "Tréboles"],
    'it': ["Picche", "Cuori", "Quadri", "Fiori"],
    'de': ["Pik", "Herz", "Karo", "Kreuz"]
}
SPECIAL_VALUES = {
    'en': {1: "Ace", 11: "Jack", 12: "Queen", 13: "King"},
    'fr': {1: "As", 11: "Valet", 12: "Dame", 13: "Roi"},
    'es': {1: "As", 11: "Jota", 12: "Reina", 13: "Rey"},
    'it': {1: "Asso", 11: "Fante", 12: "Regina", 13: "Re"},
    'de': {1: "Ass", 11: "Bube", 12: "Dame", 13: "König"}
}

class Card:
    def __init__(self, value: int, suit: int, deck_size: int = 52, language: str = 'en') -> None:
        """Initialize a card with value, suit, deck size, and language.
        
        Args:
            value: The value of the card (1-13).
            suit: The index of the suit (0-3).
            deck_size: The size of the deck, default is 52.
            language: The language for card representation, default is English ('en').
        """
        self.value = value
        self.suit = suit
        self.deck_size = deck_size
        self.language = language
        self._index = None
        self._state = None

    @property
    def state(self) -> np.ndarray:
        """Lazy-load and return the card's binary representation as a numpy array."""
        if self._index is None:
            self._index = self.calculate_index()
        if self._state is None:
            self._state = self.to_numpy()
        return self._state

    def calculate_index(self) -> int:
        """Calculate the card's index based on its value and suit, considering the deck size."""
        suit_size = self.deck_size // len(SUITS[self.language])  # Calculate suit size based on deck size
        return (self.value - 1) + self.suit * suit_size

    def to_numpy(self) -> np.ndarray:
        """Convert the card to a binary numpy array representation."""
        representation = np.zeros(self.deck_size, dtype=int)
        representation[self._index] = 1
        return representation

    def __str__(self) -> str:
        """Return a human-readable string representation of the card in the specified language."""
        value_str = SPECIAL_VALUES[self.language].get(self.value, str(self.value))
        suit_str = SUITS[self.language][self.suit] if self.suit < len(SUITS[self.language]) else "Unknown Suit"
        return f"{value_str} of {suit_str}"

    def __repr__(self) -> str:
        """Return an official string representation of the card."""
        return self.__str__()

    def __eq__(self, other) -> bool:
        """Check if two cards are equal based on their value and suit."""
        return isinstance(other, Card) and self.value == other.value and self.suit == other.suit and self.language == other.language

    def __add__(self, other) -> int:
        """Allow adding the values of two cards."""
        return self.value + other.value if isinstance(other, Card) else NotImplemented

    def __lt__(self, other) -> bool:
        """Check if this card is less than another card."""
        return self.value < other.value if isinstance(other, Card) else NotImplemented

    def __gt__(self, other) -> bool:
        """Check if this card is greater than another card."""
        return self.value > other.value if isinstance(other, Card) else NotImplemented

    def __hash__(self) -> int:
        """Return a hash value for the card."""
        return hash((self.value, self.suit, self.language))
