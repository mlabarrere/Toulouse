import pytest
import numpy as np
from toulouse import Card, Deck, get_card

# --- Test Card Creation and Core Properties ---

def test_card_creation_valid():
    """Basic creation: check field assignments, index, and one-hot state."""
    card = get_card(value=1, suit=0, card_system_key="italian_40")
    assert card.value == 1
    assert card.suit == 0
    assert card.card_system_key == "italian_40"
    assert card.to_index() == 0
    arr = card.state
    assert isinstance(arr, np.ndarray)
    assert arr.sum() == 1
    assert arr[0] == 1
    assert arr.shape == (40,)

# --- Test String Representations in Multiple Languages ---

def test_card_to_string_languages():
    """Check card string representations in en, fr, it, es."""
    # English
    card_en = get_card(value=10, suit=2, card_system_key="italian_40")
    assert card_en.to_string("en") == "King of Swords"

    # French
    card_fr = get_card(value=8, suit=1, card_system_key="italian_40")
    assert card_fr.to_string("fr") == "Valet de Coupes"

    # Italian
    card_it = get_card(value=1, suit=0, card_system_key="italian_40")
    assert card_it.to_string("it") == "Asso di Denari"

    # Spanish
    card_es = get_card(value=9, suit=3, card_system_key="spanish_40")
    assert card_es.to_string("es") == "Caballo de Bastos"

# --- Test __repr__, Equality, and Hashability ---

def test_card_repr_and_hash():
    """Check __repr__, equality, and hashability."""
    card1 = get_card(value=7, suit=2)
    card2 = get_card(value=7, suit=2)
    assert card1 == card2
    assert hash(card1) == hash(card2)
    r = repr(card1)
    assert "value=7" in r and "suit=2" in r

# --- Test Invalid Card Creation ---

def test_card_creation_invalid_value():
    """Creating a card with an invalid value should raise ValueError."""
    with pytest.raises(ValueError):
        Card(value=99, suit=0)


def test_card_creation_invalid_suit():
    """Creating a card with an invalid suit should raise ValueError."""
    with pytest.raises(ValueError):
        Card(value=2, suit=44)

# --- Test Deck Functionality ---

def test_deck_creation_and_reset():
    """Test creating a new deck and resetting it."""
    deck = Deck.new_deck(language="fr")
    assert len(deck) == 40
    assert deck.language == "fr"
    deck.draw(5)
    assert len(deck) == 35
    deck.reset()
    assert len(deck) == 40

def test_deck_pretty_print_languages():
    """Check the pretty_print output in different languages."""
    cards = [get_card(1, 0), get_card(2, 0), get_card(1, 1)]
    deck_fr = Deck.from_cards(cards, language="fr")
    output_fr = deck_fr.pretty_print()
    assert "Deniers: As de Deniers, Deux de Deniers" in output_fr
    assert "Coupes: As de Coupes" in output_fr

    deck_en = Deck.from_cards(cards, language="en")
    output_en = deck_en.pretty_print()
    assert "Coins: Ace of Coins, Two of Coins" in output_en
    assert "Cups: Ace of Cups" in output_en