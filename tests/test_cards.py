import pytest
from toulouse.cards import Card
import numpy as np

def test_card_initialization():
    """Test if a Card object is initialized with the correct attributes."""
    card = Card(1, 0, 52, 'en')
    assert card.value == 1
    assert card.suit == 0
    assert card.deck_size == 52
    assert card.language == 'en'

def test_card_string_representation():
    """Test the string representation of a Card object."""
    card = Card(1, 0)  # Ace of Spades in English
    assert str(card) == "Ace of Spades"

def test_card_equality():
    """Test if two Card objects with the same value and suit are considered equal."""
    card1 = Card(10, 2)  # 10 of Diamonds
    card2 = Card(10, 2)  # 10 of Diamonds
    assert card1 == card2

def test_card_binary_representation():
    """Test the binary numpy array representation of a Card object."""
    card = Card(1, 0)  # Ace of Spades
    state = card.state
    assert isinstance(state, np.ndarray)
    assert state.sum() == 1  # Ensure only one element is set to 1
    assert state[0] == 1  # Ace of Spades should correspond to the first position

def test_card_addition():
    """Test adding the values of two cards."""
    card1 = Card(10, 2)  # 10 of Diamonds
    card2 = Card(3, 1)   # 3 of Hearts
    assert card1 + card2 == 13

def test_card_inequality():
    """Test inequality between two Card objects."""
    card1 = Card(2, 1)  # 2 of Hearts
    card2 = Card(3, 1)  # 3 of Hearts
    assert card1 != card2

def test_card_less_than():
    """Test if one card is less than another."""
    card1 = Card(4, 1)  # 4 of Hearts
    card2 = Card(7, 2)  # 7 of Diamonds
    assert card1 < card2

def test_card_greater_than():
    """Test if one card is greater than another."""
    card1 = Card(9, 0)  # 9 of Spades
    card2 = Card(5, 3)  # 5 of Clubs
    assert card1 > card2

@pytest.mark.parametrize("value, suit, deck_size, language, expected_str", [
    (1, 0, 52, 'en', "Ace of Spades"),
    (11, 1, 52, 'fr', "Valet de Cœurs"),
    (12, 2, 40, 'es', "Reina de Diamantes"),
    (13, 3, 32, 'it', "Re di Fiori"),
    (10, 0, 52, 'de', "10 von Pik"),
])
def test_card_initialization_extended(value, suit, deck_size, language, expected_str):
    """Test if a Card object is initialized with the correct attributes and string representation in various languages."""
    card = Card(value, suit, deck_size, language)
    assert card.value == value
    assert card.suit == suit
    assert card.deck_size == deck_size
    assert card.language == language
    assert str(card) == expected_str

@pytest.mark.parametrize("values, suits, result", [
    ((10, 10), (2, 2), True),
    ((1, 13), (0, 0), False),
    ((11, 11), (3, 1), False),
    ((7, 7), (2, 2), True),
])
def test_card_equality_extended(values, suits, result):
    """Test equality between various Card objects."""
    card1 = Card(values[0], suits[0])
    card2 = Card(values[1], suits[1])
    assert (card1 == card2) is result

@pytest.mark.parametrize("card1_details, card2_details, expected_result", [
    ((10, 2), (3, 1), 13),
    ((5, 0), (5, 3), 10),
    ((11, 1), (2, 2), 13),
    ((8, 3), (4, 0), 12),
])
def test_card_addition_extended(card1_details, card2_details, expected_result):
    """Test adding the values of two cards with varied values."""
    card1 = Card(*card1_details)
    card2 = Card(*card2_details)
    assert card1 + card2 == expected_result
