# Toulouse: A High-Performance Card Game Library for Scientific Computing

**Toulouse** is a Python library engineered for the high-performance simulation and vectorisation of card games, with a primary focus on applications in reinforcement learning (RL) and Monte Carlo Tree Search (MCTS).

It provides a robust, type-annotated, and efficient foundation for researchers and developers who require rapid state manipulation and observation generation for card-based environments.

---

## Key Features

- **Performance-Oriented Design**: Implements object pooling, LRU caching, and pre-computed state vectors to minimise computational overhead in large-scale simulations.
- **Vectorised State Representation**: Natively generates NumPy array representations for `Card` and `Deck` objects, suitable for direct integration with machine learning frameworks.
- **Extensible Card Systems**: Supports multiple card game configurations (e.g., Italian 40-card, Spanish 40-card) and allows users to register custom systems dynamically.
- **Internationalisation**: Built-in support for multiple languages (en, fr, it, es) via a centralised translation module.
- **Type-Safe and Tested**: A fully type-annotated codebase with a comprehensive `pytest` test suite to ensure reliability.

---

## Core Design & Architecture

Toulouse achieves its performance through several key architectural decisions:

1.  **`Card` Object Pooling**: The `get_card()` factory function ensures that each unique `Card` instance is created only once. Subsequent requests for the same card return a cached reference from a global pool, significantly reducing object creation overhead and memory footprint.

2.  **LRU-Cached System Lookups**: The `get_card_system()` function is decorated with `@lru_cache`, ensuring that card system configuration data is retrieved from memory after the initial lookup.

3.  **Lazy State Vectorisation**: Each `Deck` instance maintains a private `_state_cache` (a NumPy array). This cache is only recomputed when the deck's composition changes (tracked by a `_state_dirty` flag), making repeated access to the `.state` property exceptionally fast.

---

## Installation

Install the library using `pip` or `uv`:

```bash
# Using pip
pip install toulouse

# Using uv
uv add toulouse
```

---

## Quick Start

```python
from toulouse import Deck, get_card

# 1. Initialise a new 40-card Italian deck.
# The default language is Italian ('it').
deck = Deck.new_deck(card_system_key="italian_40", language="fr")
print(deck)  # Output: Deck of 40 cards (italian_40)

# 2. Draw the top card from the deck.
deck.shuffle()
hand = deck.draw(1)
drawn_card = hand[0]
print(f"Carte piochée: {drawn_card.to_string('fr')}") # Output: Carte piochée: [Card Name]

# 3. Use the factory function to get a specific card instance.
ace_of_spades = get_card(value=1, suit=2, card_system_key="italian_40")

# 4. Check for card presence (O(1) complexity).
print(f"Le deck contient-il l'As d'Épées? {deck.contains(ace_of_spades)}")

# 5. Get the deck's state as a NumPy vector for ML applications.
state_vector = deck.state
print(f"Shape of state vector: {state_vector.shape}") # Output: Shape of state vector: (40,)

# 6. Display the contents of the deck, grouped by suit.
deck.sort()
print(deck.pretty_print())
```

---

## API Reference

### `get_card()` Factory Function

This is the recommended method for obtaining `Card` instances.

`get_card(value: int, suit: int, card_system_key: str = "italian_40") -> Card`

### `Card` Class

An immutable, hashable data class representing a single card.

- `card.value: int`
- `card.suit: int`
- `card.to_index() -> int`: Returns the card's unique integer index within its system.
- `card.state -> np.ndarray`: Returns a one-hot encoded NumPy vector of the card's state.
- `card.to_string(language: str) -> str`: Returns a localised string representation.

### `Deck` Class

A mutable container for a collection of `Card` objects.

- `Deck.new_deck(card_system_key, language, sorted_deck)`: Class method to create a new, full deck.
- `Deck.from_cards(cards, card_system_key, language)`: Class method to create a deck from an existing list of cards.
- `deck.draw(n: int)`: Removes and returns `n` cards from the top of the deck.
- `deck.append(card: Card)`: Adds a card to the bottom of the deck.
- `deck.contains(card: Card) -> bool`: Checks for the presence of a card (O(1) complexity).
- `deck.state -> np.ndarray`: Returns a binary NumPy vector representing the current state of the deck.
- `deck.shuffle()`: Shuffles the deck in-place.
- `deck.sort()`: Sorts the deck in-place based on card index.
- `deck.reset()`: Restores the deck to its full, sorted state.
- `deck.pretty_print() -> str`: Returns a formatted string of the deck's contents, grouped by suit.

### Card System Management

- `register_card_system(key: str, config: dict)`: Registers a new card system configuration.
- `get_card_system(key: str) -> dict`: Retrieves a card system's configuration dictionary.

---

## Performance Benchmarks

The following benchmarks were recorded on an Apple M-series CPU with Python 3.11.

- **Deck Instantiation (1,000 iterations)**: ~6.2 ms
- **Shuffle, Draw & Reset (1,000 iterations)**: ~9.9 ms
- **Card Lookup (10,000 iterations)**: ~0.6 ms
- **State Vectorisation (10,000 iterations)**: ~4.2 ms

These results demonstrate the library's suitability for performance-critical applications.

---

## Testing

To run the test suite, execute `pytest` from the project root:

```bash
pytest
```

---

## Licence

This project is licensed under the MIT Licence.