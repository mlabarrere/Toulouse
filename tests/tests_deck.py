import pytest
from toulouse.cards import Card, DECK_SIZE
from toulouse.deck import Deck

def test_card_index_and_array():
    card = Card(value=1, suit=0)  # Asso di Denari
    idx = card.to_index()
    assert idx == 0
    arr = card.to_array()
    assert arr.sum() == 1
    assert arr[idx] == 1

def test_deck_initialization():
    deck = Deck()
    assert len(deck) == 40
    arr = deck.to_array()
    assert arr.sum() == 40
    assert (arr == 1).all()

def test_draw_and_add_card():
    deck = Deck()
    card = deck.draw()[0]
    assert card not in deck.cards
    deck.add_card(card)
    assert card in deck.cards

def test_remove_and_contains():
    deck = Deck()
    card = deck.cards[10]
    assert deck.contains(card)
    deck.remove_card(card)
    assert not deck.contains(card)

def test_move_card_to():
    deck1 = Deck()
    deck2 = Deck(new=False)
    card = deck1.cards[5]
    deck1.move_card_to(card, deck2)
    assert card in deck2.cards
    assert card not in deck1.cards

def test_deck_reset():
    deck = Deck()
    deck.draw(10)
    assert len(deck) == 30
    deck.reset()
    assert len(deck) == 40

def test_draw_more_than_deck():
    deck = Deck()
    drawn = deck.draw(45)
    assert len(drawn) == 40
    assert len(deck) == 0
