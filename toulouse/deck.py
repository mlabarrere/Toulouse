import random
import numpy as np
from typing import List
from cards import Card, SUITS, VALUES, DECK_SIZE

class Deck:
    def __init__(self, new: bool = True, shuffle: bool = False):
        self.cards: List[Card] = []
        if new:
            self.cards = [Card(value=v, suit=s) for s in range(len(SUITS)) for v in VALUES]
            if shuffle:
                random.shuffle(self.cards)

    def draw(self, n: int = 1) -> List[Card]:
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def shuffle(self):
        random.shuffle(self.cards)

    def add_card(self, card: Card):
        self.cards.append(card)

    def remove_card(self, card: Card):
        self.cards.remove(card)

    def contains(self, card: Card) -> bool:
        return card in self.cards

    def move_card_to(self, card: Card, other_deck: 'Deck'):
        self.remove_card(card)
        other_deck.add_card(card)

    def reset(self):
        self.cards = [Card(value=v, suit=s) for s in range(len(SUITS)) for v in VALUES]

    def to_array(self) -> np.ndarray:
        arr = np.zeros(DECK_SIZE, dtype=np.uint8)
        for card in self.cards:
            arr[card.to_index()] = 1
        return arr

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return f"Deck({len(self.cards)} cards)"

    def __repr__(self):
        return f"Deck(cards={self.cards})"
