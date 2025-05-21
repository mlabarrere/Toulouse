import numpy as np

# CARD_SYSTEMS defines the properties of different types of card decks.
# Each key in CARD_SYSTEMS is a string identifier for a card system (e.g., "standard_52", "italian_40").
# Each card system is a dictionary with the following keys:
#   - "suits": A dictionary mapping language codes (e.g., 'en', 'it') to a list of suit names.
#   - "special_values": A dictionary mapping language codes to a dictionary where keys are numerical card values
#                       and values are their string representations (e.g., {1: "Ace", 11: "Jack"}).
#   - "values_map": A list of all valid numerical values for a card in this system.
#                   This map is crucial for determining a card's index and for validation.
#   - "deck_size": The total number of unique cards in a full deck of this system.
#   - "num_suits": The number of suits in this card system.
CARD_SYSTEMS = {
    "standard_52": {
        "suits": {
            'en': ["Spades", "Hearts", "Diamonds", "Clubs"],
            'fr': ["Piques", "Cœurs", "Carreaux", "Trèfles"],
            'es': ["Espadas", "Corazones", "Diamantes", "Tréboles"],
            'it': ["Picche", "Cuori", "Quadri", "Fiori"], # Standard Italian names for 52-card deck suits
            'de': ["Pik", "Herz", "Karo", "Kreuz"]
        },
        "special_values": {
            'en': {1: "Ace", 11: "Jack", 12: "Queen", 13: "King"},
            'fr': {1: "As", 11: "Valet", 12: "Dame", 13: "Roi"},
            'es': {1: "As", 11: "Jota", 12: "Reina", 13: "Rey"},
            'it': {1: "Asso", 11: "Fante", 12: "Regina", 13: "Re"}, # Standard Italian names for 52-card deck special values
            'de': {1: "Ass", 11: "Bube", 12: "Dame", 13: "König"}
        },
        "values_map": list(range(1, 14)),  # Ace (1) to King (13)
        "deck_size": 52,
        "num_suits": 4,
    },
    "italian_40": {
        "suits": { # For Italian 40-card system, only Italian names are most relevant
            'it': ["Denari", "Coppe", "Spade", "Bastoni"],
            # Providing English for completeness, though not traditional for this system
            'en': ["Coins", "Cups", "Swords", "Batons"],
        },
        "special_values": { # For Italian 40-card system
            'it': {1: "Asso", 8: "Fante", 9: "Cavallo", 10: "Re"}, # Values are 1-7, Fante (8), Cavallo (9), Re (10)
            'en': {1: "Ace", 8: "Jack", 9: "Knight", 10: "King"}, # English approximations
        },
        "values_map": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # Ace, 2-7, then Fante, Cavallo, Re
        "deck_size": 40,
        "num_suits": 4,
    }
}

# Language conjunctions used for constructing human-readable card strings (e.g., "Ace OF Spades").
LANGUAGE_CONJUNCTIONS = {
    'en': "of",
    'fr': "de",
    'es': "de",
    'it': "di",
    'de': "von"
}

class Card:
    def __init__(self, value: int, suit: int, card_system_key: str = 'standard_52', language: str = 'en') -> None:
        """Initialize a card with value, suit, card system, and language.
        
        Args:
            value: The numerical value of the card. This value's meaning is determined by the
                   `values_map` of the selected `card_system_key`. For example, in a standard
                   deck, 1 is Ace, 13 is King. In an Italian 40-card deck, values might be 1-7, 8 (Fante), 
                   9 (Cavallo), 10 (Re).
            suit: The index of the suit (e.g., 0 for Spades in a standard deck, or 0 for Denari
                  in an Italian deck if it's the first suit defined in its system).
            card_system_key: A string key (e.g., 'standard_52', 'italian_40') that determines the
                             card's fundamental properties such as its range of possible values and suits,
                             and how it fits into a deck (e.g., total deck size for binary representation).
                             This is the primary way to define the type of card.
            language: A string code (e.g., 'en', 'it') that primarily affects the card's string
                      representation (e.g., "Ace of Spades" vs "As de Piques"). It does not
                      change the card's functional value or suit index.
        """
        self.value = value  # Numerical value of the card
        self.suit = suit    # Numerical suit index of the card
        self.card_system_key = card_system_key # Identifier for the card system
        self.language = language # Language for string representation

        # Validate and load the card system properties
        if self.card_system_key not in CARD_SYSTEMS:
            raise ValueError(f"Unknown card system key: {self.card_system_key}")
        self.card_system = CARD_SYSTEMS[self.card_system_key]

        # Validate value and suit against the selected card system
        if self.value not in self.card_system["values_map"]:
            raise ValueError(f"Value {self.value} is not valid for card system {self.card_system_key}. Valid values: {self.card_system['values_map']}")
        if not (0 <= self.suit < self.card_system["num_suits"]):
            raise ValueError(f"Suit index {self.suit} is not valid for card system {self.card_system_key}. Valid range: 0-{self.card_system['num_suits']-1}")
        
        # Determine the display language for suits and special values, with fallback.
        # This allows a card to have a primary language (self.language) for conjunctions,
        # but use a fallback language if its names are not defined for self.language in CARD_SYSTEMS.
        if self.language not in self.card_system["suits"]:
            print(f"Warning: Language '{self.language}' not defined for suits in '{self.card_system_key}'. Falling back.")
            self.display_language_suits = list(self.card_system["suits"].keys())[0] # Fallback to first defined language
        else:
            self.display_language_suits = self.language

        if self.language not in self.card_system["special_values"]:
            print(f"Warning: Language '{self.language}' not defined for special values in '{self.card_system_key}'. Falling back.")
            self.display_language_special_values = list(self.card_system["special_values"].keys())[0] # Fallback
        else:
            self.display_language_special_values = self.language

        self._index = None  # Cached index of the card within its system's deck
        self._state = None  # Cached binary (numpy array) representation of the card

    @property
    def state(self) -> np.ndarray:
        """Lazy-load and return the card's binary representation as a numpy array.
        The array's size is `deck_size` from the card's system, with a '1' at the card's unique index
        and '0's elsewhere. This is useful for machine learning or state-based game representations.
        The index is calculated by `calculate_index()`.
        """
        if self._index is None:
            self._index = self.calculate_index()
        if self._state is None:
            self._state = self.to_numpy()
        return self._state

    def calculate_index(self) -> int:
        """Calculate the card's unique, zero-based index within a full deck of its system.
        This index is determined by the card's suit and its value's position in the system's `values_map`.
        For example, in a system with 10 values per suit:
        - A card with suit 0 and the first value in `values_map` gets index 0.
        - A card with suit 1 and the first value in `values_map` gets index 10 (num_values_per_suit * suit_index).
        """
        # Number of unique values per suit (e.g., 13 for standard, 10 for Italian 40-card)
        num_values_per_suit = len(self.card_system["values_map"])
        # Find the 0-based index of the card's value within its system's defined sequence of values.
        # E.g., if values_map is [1, 2, ..., 7, 8, 9, 10] and card value is 8 (Fante), its value_idx is 7.
        value_idx = self.card_system["values_map"].index(self.value)
        # The final index is calculated by: (value's position within suit) + (suit index * number of values per suit)
        return value_idx + self.suit * num_values_per_suit

    def to_numpy(self) -> np.ndarray:
        """Convert the card to a binary numpy array representation based on its system's deck size."""
        representation = np.zeros(self.card_system["deck_size"], dtype=int)
        # Set the bit at the card's calculated unique index to 1.
        representation[self.calculate_index()] = 1 
        return representation

    def __str__(self) -> str:
        """Return a human-readable string representation of the card.
        It uses the card's `value` and `suit` to look up names in the `special_values` and `suits`
        dictionaries from its `card_system`, using the `display_language_special_values` and
        `display_language_suits` (which may be fallbacks). The `language` attribute of the card
        is used to select the correct conjunction (e.g., "of", "di").
        """
        # Get special value name (e.g., "Ace", "Fante") if applicable, otherwise use numerical value.
        special_values_for_lang = self.card_system["special_values"].get(self.display_language_special_values, {})
        value_str = special_values_for_lang.get(self.value, str(self.value))
        
        # Get suit name (e.g., "Spades", "Denari").
        suits_for_lang = self.card_system["suits"].get(self.display_language_suits, [])
        suit_str = suits_for_lang[self.suit] if self.suit < len(suits_for_lang) else "Unknown Suit"
        
        # Get conjunction based on the card's primary language setting.
        conjunction = LANGUAGE_CONJUNCTIONS.get(self.language, "of") # Default to "of"
        
        return f"{value_str} {conjunction} {suit_str}"

    def __repr__(self) -> str:
        """Return an official, unambiguous string representation of the card, useful for debugging."""
        return f"Card(value={self.value}, suit={self.suit}, card_system_key='{self.card_system_key}', language='{self.language}')"

    def __eq__(self, other) -> bool:
        """Check if two cards are equal.
        Equality is based on having the same value, suit, AND being part of the same card system.
        A card from a 'standard_52' system is never equal to a card from an 'italian_40' system,
        even if their numerical value and suit index happen to be the same.
        """
        return isinstance(other, Card) and \
               self.value == other.value and \
               self.suit == other.suit and \
               self.card_system_key == other.card_system_key

    def __add__(self, other) -> int:
        """Allow adding the numerical values of two cards.
        Note: This assumes values are directly comparable numerically, which is typical for many card games.
        It does not consider the card system for this operation, only the `value` attribute.
        """
        return self.value + other.value if isinstance(other, Card) else NotImplemented

    def __lt__(self, other) -> bool:
        """Check if this card is less than another card.
        Comparison is primarily based on numerical value. If values are equal, suit index acts as a tie-breaker.
        Cards must be from the same card system to be comparable.
        """
        if not isinstance(other, Card) or self.card_system_key != other.card_system_key:
            return NotImplemented # Cannot compare cards from different systems
        if self.value == other.value:
            return self.suit < other.suit # Suit index as tie-breaker
        return self.value < other.value

    def __gt__(self, other) -> bool:
        """Check if this card is greater than another card.
        Comparison is primarily based on numerical value. If values are equal, suit index acts as a tie-breaker.
        Cards must be from the same card system to be comparable.
        """
        if not isinstance(other, Card) or self.card_system_key != other.card_system_key:
            return NotImplemented # Cannot compare cards from different systems
        if self.value == other.value:
            return self.suit > other.suit # Suit index as tie-breaker
        return self.value > other.value

    def __hash__(self) -> int:
        """Return a hash value for the card.
        The hash is based on the card's value, suit, and its card system key, ensuring that
        cards that are considered equal (by `__eq__`) have the same hash.
        """
        return hash((self.value, self.suit, self.card_system_key))

# Example usage section for quick testing or demonstration.
# This part is typically not run when the module is imported elsewhere.
if __name__ == '__main__':
    # Standard 52-card examples
    ace_of_spades_en = Card(value=1, suit=0, card_system_key='standard_52', language='en')
    print(f"English: {ace_of_spades_en} (Index: {ace_of_spades_en.calculate_index()})") # Expected: Ace of Spades, Index: 0

    as_de_pique_fr = Card(value=1, suit=0, card_system_key='standard_52', language='fr')
    print(f"French: {as_de_pique_fr} (Index: {as_de_pique_fr.calculate_index()})") # Expected: As de Piques, Index: 0

    # Italian 40-card examples
    asso_di_denari_it = Card(value=1, suit=0, card_system_key='italian_40', language='it')
    print(f"Italian 40-card: {asso_di_denari_it} (Index: {asso_di_denari_it.calculate_index()})") # Expected: Asso di Denari, Index: 0 (value 1 is 1st in values_map)

    fante_di_coppe_it = Card(value=8, suit=1, card_system_key='italian_40', language='it') # Fante (value 8) di Coppe (suit 1)
    # Italian values_map: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. Index of 8 is 7.
    # Index = 7 (value_idx) + 1 (suit_idx) * 10 (num_values_per_suit) = 17
    print(f"Italian 40-card: {fante_di_coppe_it} (Index: {fante_di_coppe_it.calculate_index()})") # Expected: Fante di Coppe, Index: 17

    # Test state property
    print(f"State for {ace_of_spades_en} (len {len(ace_of_spades_en.state)}): {ace_of_spades_en.state.argmax()}") # Should be 1 at index 0
    print(f"State for {asso_di_denari_it} (len {len(asso_di_denari_it.state)}): {asso_di_denari_it.state.argmax()}") # Should be 1 at index 0
    
    # Test equality
    ace_spades_en_copy = Card(value=1, suit=0, card_system_key='standard_52', language='en')
    print(f"Equality (same system, same card): {ace_of_spades_en == ace_spades_en_copy}") # Expected: True
    print(f"Equality (different system): {ace_of_spades_en == asso_di_denari_it}") # Expected: False
    
    # Test comparison
    king_of_hearts_std = Card(value=13, suit=1, card_system_key='standard_52', language='en')
    print(f"{ace_of_spades_en} < {king_of_hearts_std}: {ace_of_spades_en < king_of_hearts_std}") # Expected: True

    # Example of language fallback warning
    print("\nTesting language fallback:")
    card_de_fallback_suits = Card(value=1, suit=0, card_system_key='italian_40', language='de') # German ('de') not in italian_40 suits/specials
    # Expect warnings to be printed here from __init__
    print(f"German (fallback for Italian system card): {card_de_fallback_suits}") # String output will use fallback language names

    # Test invalid card scenarios
    print("\nTesting invalid card scenarios:")
    try:
        invalid_card_value = Card(value=15, suit=0, card_system_key='standard_52') # 15 is not in standard_52 values_map
    except ValueError as e:
        print(f"Error for invalid card value: {e}")

    try:
        invalid_suit_index = Card(value=1, suit=4, card_system_key='standard_52') # Standard deck has 4 suits (indices 0-3)
    except ValueError as e:
        print(f"Error for invalid suit index: {e}")

    try:
        invalid_system_key = Card(value=1, suit=0, card_system_key='non_existent_system')
    except ValueError as e:
        print(f"Error for invalid system key: {e}")

    # Detailed index calculation check for Italian deck
    print("\nDetailed index calculation for Italian deck:")
    # Denari (suit 0): values 1-10 -> indices 0-9
    # Coppe (suit 1): values 1-10 -> indices 10-19
    # Spade (suit 2): values 1-10 -> indices 20-29
    # Bastoni (suit 3): values 1-10 -> indices 30-39
    re_di_bastoni = Card(value=10, suit=3, card_system_key='italian_40', language='it') # Re (value 10) di Bastoni (suit 3)
    # values_map.index(10) = 9. Index = 9 + 3 * 10 = 39.
    print(f"{re_di_bastoni} (Index: {re_di_bastoni.calculate_index()})") # Expected: Re di Bastoni, Index: 39

    asso_di_coppe = Card(value=1, suit=1, card_system_key='italian_40', language='it') # Asso (value 1) di Coppe (suit 1)
    # values_map.index(1) = 0. Index = 0 + 1 * 10 = 10.
    print(f"{asso_di_coppe} (Index: {asso_di_coppe.calculate_index()})") # Expected: Asso di Coppe, Index: 10
```
