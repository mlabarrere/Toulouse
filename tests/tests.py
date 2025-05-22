import pytest
import numpy as np
from toulouse.cards import Card, DECK_SIZE
from toulouse.deck import Deck


def test_card_index_and_array(self):
    card = Card(value=1, suit=0)  # Asso di Denari
    idx = card.to_index()
    self.assertEqual(idx, 0)
    arr = card.to_array()
    self.assertEqual(arr.sum(), 1)
    self.assertEqual(arr[idx], 1)

def test_deck_initialization(self):
    deck = Deck()
    self.assertEqual(len(deck), 40)
    arr = deck.to_array()
    self.assertEqual(arr.sum(), 40)
    self.assertTrue((arr == 1).all())

def test_draw_and_add_card(self):
    deck = Deck()
    card = deck.draw()[0]
    self.assertNotIn(card, deck.cards)
    deck.add_card(card)
    self.assertIn(card, deck.cards)

def test_remove_and_contains(self):
    deck = Deck()
    card = deck.cards[10]
    self.assertTrue(deck.contains(card))
    deck.remove_card(card)
    self.assertFalse(deck.contains(card))

def test_move_card_to(self):
    deck1 = Deck()
    deck2 = Deck(new=False)
    card = deck1.cards[5]
    deck1.move_card_to(card, deck2)
    self.assertIn(card, deck2.cards)
    self.assertNotIn(card, deck1.cards)

def test_deck_reset(self):
    deck = Deck()
    deck.draw(10)
    self.assertEqual(len(deck), 30)
    deck.reset()
    self.assertEqual(len(deck), 40)

def test_draw_more_than_deck(self):
    deck = Deck()
    drawn = deck.draw(45)
    self.assertEqual(len(drawn), 40)
    self.assertEqual(len(deck), 0)
