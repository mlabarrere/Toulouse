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
