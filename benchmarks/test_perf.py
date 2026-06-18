"""Performance benchmarks for Toulouse hot paths.

Run with:
    uv run -m pytest benchmarks/ --benchmark-only

These benchmarks target the operations called most frequently in RL/MCTS
loops: state vectorisation, drawing, deck cloning and membership checks.
"""

import pytest

from toulouse import Deck, get_card


@pytest.fixture
def full_deck() -> Deck:
    return Deck.new_deck()


def test_new_deck(benchmark):
    benchmark(Deck.new_deck)


def test_shuffle(benchmark, full_deck):
    benchmark(full_deck.shuffle)


def test_draw_one(benchmark):
    def run():
        deck = Deck.new_deck()
        while len(deck):
            deck.draw(1)
    benchmark(run)


def test_deck_state_repeated_access(benchmark, full_deck):
    # Cache hit path: this is the dominant ML hot path.
    _ = full_deck.state  # warm cache
    benchmark(lambda: full_deck.state)


def test_card_state_repeated_access(benchmark):
    card = get_card(value=1, suit=0)
    benchmark(lambda: card.state)


def test_to_index(benchmark):
    card = get_card(value=7, suit=2)
    benchmark(card.to_index)


def test_contains(benchmark, full_deck):
    card = get_card(value=5, suit=1)
    benchmark(lambda: full_deck.contains(card))


def test_mcts_clone(benchmark, full_deck):
    # Simulate MCTS branch exploration: clone the deck many times.
    _ = full_deck.state  # warm cache so copy() exercises the shared-state path
    def run():
        for _ in range(100):
            full_deck.copy()
    benchmark(run)


def test_to_string(benchmark):
    card = get_card(value=10, suit=3)
    benchmark(lambda: card.to_string("fr"))
