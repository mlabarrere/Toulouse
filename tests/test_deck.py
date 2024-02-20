import pytest
from toulouse.cards import Card
from toulouse.deck import Deck
import numpy as np

@pytest.mark.parametrize("deck_size, expected_length", [(52, 52), (40, 40), (20, 20), (32, 32)])
def test_deck_initialization_varied_sizes(deck_size, expected_length):
    """Test the initialization of Deck objects with varied sizes."""
    deck = Deck(new=True, deck_size=deck_size)
    assert len(deck.cards) == expected_length


@pytest.mark.parametrize("language, first_card", [
    ('en', 'Ace of Spades'),
    ('fr', 'As de Piques'),
    ('es', 'As de Espadas'),
    ('it', 'Asso di Picche'),
    ('de', 'Ass von Pik')])
def test_deck_custom_initialization_languages(language, first_card):
    """Test the initialization of a Deck object with different languages."""
    deck = Deck(new=True, language=language)
    assert str(deck.cards[0]) == first_card


@pytest.mark.parametrize("num_cards_to_draw", [1, 5, 10, 15])
def test_deck_draw_varied_numbers(num_cards_to_draw):
    """Test drawing a varied number of cards from the Deck."""
    deck = Deck(new=True)
    drawn_cards = deck.draw(num_cards_to_draw)
    assert len(drawn_cards) == num_cards_to_draw
    assert len(deck.cards) == 52 - num_cards_to_draw


@pytest.mark.parametrize("draw_amount", [53, 60, 70, 100])
def test_deck_draw_error_varied_amounts(draw_amount):
    """Test drawing more cards than are in the Deck raises an error with varied amounts."""
    deck = Deck(new=True)
    with pytest.raises(ValueError):
        deck.draw(draw_amount)


@pytest.mark.parametrize("card_values", [(1, 0), (13, 3), (5, 2), (8, 1)])
def test_deck_append_varied_cards(card_values):
    """Test appending varied cards to the Deck."""
    value, suit = card_values
    deck = Deck()
    card_to_add = Card(value, suit)
    deck.append(card_to_add)
    assert card_to_add in deck.cards


@pytest.mark.parametrize("card_index", [0, 10, 20, 30])
def test_deck_append_error_duplicate(card_index):
    """Test appending duplicate cards to the Deck raises an error."""
    deck = Deck(new=True)
    card_to_add = deck.cards[card_index]
    with pytest.raises(ValueError, match="Card already in deck."):
        deck.append(card_to_add)


@pytest.mark.parametrize("card_index", [0, 10, 20, 30])
def test_deck_remove_varied_cards(card_index):
    """Test removing varied cards from the Deck."""
    deck = Deck(new=True)
    card_to_remove = deck.cards[card_index]
    deck.remove(card_to_remove)
    assert card_to_remove not in deck.cards


@pytest.mark.parametrize("value, suit", [(14, 0), (0, 3), (20, 2), (-1, 1)])
def test_deck_remove_error_invalid_card(value, suit):
    """Test removing a card not in the Deck raises an error with invalid cards."""
    deck = Deck(new=True)
    card_to_remove = Card(value, suit)
    with pytest.raises(ValueError):
        deck.remove(card_to_remove)


def test_deck_shuffle_multiple_times():
    """Test shuffling the cards in the Deck multiple times."""
    deck = Deck(new=True, sorted=True)
    for _ in range(4):  # Shuffle 4 times
        deck.sorted = False
        deck.update_sort()
        not_sorted = False
        for i in range(len(deck.cards) - 1):
            if deck.cards[i].value > deck.cards[i + 1].value:
                not_sorted = True
                break
        assert not_sorted, "Deck should be shuffled and not in sorted order each time."


@pytest.mark.parametrize("invalid_state_size", [10, 30, 60, 100])
def test_deck_from_state_error_varied_sizes(invalid_state_size):
    """Test creating a Deck from invalid state arrays of varied sizes raises an error."""
    invalid_state = np.zeros(invalid_state_size)
    with pytest.raises(ValueError):
        Deck.from_state(invalid_state)


def test_deck_length_after_operations():
    """Test the length magic method of the Deck after various operations."""
    deck = Deck(new=True)
    _ = deck.draw(5)
    deck.append(Card(0, 0))
    deck.remove(deck.cards[0])
    assert len(deck) == 52 - 5, "Length of deck should be equal to the number of cards after operations."
