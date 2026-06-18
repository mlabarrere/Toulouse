# Toulouse — High-Performance Python Card Game Library for Reinforcement Learning & MCTS

[![PyPI version](https://img.shields.io/pypi/v/toulouse.svg)](https://pypi.org/project/toulouse/)
[![Python versions](https://img.shields.io/pypi/pyversions/toulouse.svg)](https://pypi.org/project/toulouse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/mlabarrere/Toulouse/actions/workflows/pytest.yml/badge.svg)](https://github.com/mlabarrere/Toulouse/actions/workflows/pytest.yml)
[![Typed](https://img.shields.io/badge/typing-typed-blue.svg)](https://peps.python.org/pep-0561/)

**Toulouse is a fast, type-safe Python library for simulating and vectorizing card games**, built for
reinforcement learning (RL), Monte Carlo Tree Search (MCTS), and other machine-learning workloads.
It turns decks and cards into ready-to-use **NumPy observation vectors** with **zero-copy state
access**, **object pooling**, and **O(1) membership tests** — so you can run millions of game steps
per training loop without the memory churn and overhead of a naive implementation.

```bash
pip install toulouse
```

---

## Why Toulouse?

Card-game environments for RL and MCTS hammer the same operations millions of times: building
observation vectors, drawing cards, copying game states for tree search, and checking membership.
A naive card library allocates and copies NumPy arrays on every access, which destroys throughput
and creates relentless garbage-collection pressure.

Toulouse is engineered to eliminate that overhead:

- **Zero-copy state vectors** — `card.state` and `deck.state` return cached, read-only NumPy views.
  Repeated access is O(1) with **no allocation**, the single biggest win in tight RL/MCTS loops.
- **Precomputed card indices** — each card's integer index and one-hot vector are computed once at
  creation, so `to_index()` and `.state` are plain attribute reads.
- **Object pooling** — `get_card()` returns a shared, immutable instance per unique card, keeping
  the memory footprint tiny even across millions of references.
- **Copy-on-write deck cloning** — `deck.copy()` shares the immutable cached state instead of
  copying it, making MCTS branch exploration cheap.
- **O(1) membership** — `deck.contains(card)` is set-backed, not a linear scan.

The result: a library you can drop straight into a Gym-style environment or an MCTS rollout without
it becoming your bottleneck.

---

## Key Features

- ⚡ **Performance-first design**: object pooling, LRU-cached configs, lazy + cached vectorization,
  and zero-copy observations.
- 🧮 **NumPy-native observations**: one-hot `Card` vectors and binary `Deck` composition vectors
  (`np.uint8`) ready for PyTorch, JAX, TensorFlow, or Gymnasium.
- 🃏 **Extensible card systems**: ships with Italian 40-card and Spanish 40-card decks; register your
  own with `register_card_system()`.
- 🌍 **Internationalization**: built-in English, French, Italian, and Spanish names.
- ✅ **Type-safe & tested**: fully type-annotated (`py.typed`), with a `pytest` suite and a
  `pytest-benchmark` performance suite.
- 🪶 **Tiny footprint**: a single runtime dependency (NumPy).

---

## Installation

```bash
# Using pip
pip install toulouse

# Using uv
uv add toulouse
```

Toulouse supports **Python 3.9+** and requires only **NumPy**.

---

## Quick Start

```python
from toulouse import Deck, get_card

# 1. Create a new, shuffled 40-card Italian deck (names rendered in French).
deck = Deck.new_deck(card_system_key="italian_40", language="fr", sorted_deck=False)
print(deck)  # Deck of 40 cards (italian_40)

# 2. Draw the top card.
drawn_card = deck.draw(1)[0]
print(f"Drawn card: {drawn_card.to_string('fr')}")

# 3. Get a specific, pooled card instance.
ace_of_swords = get_card(value=1, suit=2, card_system_key="italian_40")

# 4. O(1) membership check.
print(f"Deck still contains the Ace of Swords? {deck.contains(ace_of_swords)}")

# 5. Get the deck observation vector for your ML model (zero-copy, read-only).
state_vector = deck.state            # np.ndarray, shape (40,), dtype uint8
model_input = state_vector           # use directly, or np.array(...) for a mutable copy
print(f"State vector shape: {state_vector.shape}")

# 6. Pretty-print the deck grouped by suit.
deck.sort()
print(deck.pretty_print())
```

### Using Toulouse in a reinforcement-learning environment

```python
import numpy as np
from toulouse import Deck

class CardEnv:
    def reset(self):
        self.deck = Deck.new_deck(sorted_deck=False)
        return self.observation()

    def step(self, action_n=1):
        hand = self.deck.draw(action_n)
        return self.observation(), len(hand), len(self.deck) == 0

    def observation(self) -> np.ndarray:
        # Zero-copy, read-only view — wrap with np.array(...) only if you must mutate.
        return self.deck.state
```

---

## API Reference

### `get_card(value, suit, card_system_key="italian_40") -> Card`

The recommended way to obtain `Card` instances. Returns a shared, immutable, pooled instance.

### `Card`

An immutable, hashable dataclass representing a single card.

| Member | Description |
| --- | --- |
| `card.value: int` | Card value (e.g. 1–10). |
| `card.suit: int` | Suit index. |
| `card.to_index() -> int` | Unique integer index within the card system (O(1), precomputed). |
| `card.state -> np.ndarray` | One-hot `uint8` vector. **Cached & read-only (zero-copy)**; use `np.array(card.state)` for a mutable copy. |
| `card.to_string(language="it") -> str` | Localized name (en/fr/it/es). |

### `Deck`

A mutable container of `Card` objects.

| Member | Description |
| --- | --- |
| `Deck.new_deck(card_system_key, language, sorted_deck)` | Create a full deck. |
| `Deck.from_cards(cards, card_system_key, language)` | Build a deck from existing cards. |
| `deck.draw(n) -> List[Card]` | Remove and return `n` cards from the top. |
| `deck.append(card)` | Add a card to the bottom. |
| `deck.remove(card)` | Remove a specific card. |
| `deck.contains(card) -> bool` | Membership test (**O(1)**). |
| `deck.state -> np.ndarray` | Binary `uint8` composition vector. **Cached & read-only (zero-copy)**; use `np.array(deck.state)` for a mutable copy. |
| `deck.shuffle()` / `deck.sort()` | In-place shuffle / sort. |
| `deck.reset()` | Restore the full, sorted deck. |
| `deck.copy() -> Deck` | Cheap copy-on-write clone (ideal for MCTS). |
| `deck.pretty_print() -> str` | Formatted contents grouped by suit. |

### Card system management

- `register_card_system(key, config)` — register a custom system (`suits`, `values`, `deck_size`).
- `get_card_system(key) -> dict` — retrieve a system configuration (LRU-cached).

---

## Performance

Toulouse's hot paths are designed for **zero per-call allocation**. Measured speedups versus a
naive (allocate/copy-every-call) implementation, 1,000,000 iterations on a single CPU core:

| Hot path | Naive | Toulouse | Speedup |
| --- | --- | --- | --- |
| `Card.state` (observation) | allocate new array each call | cached read-only view | **~6×** |
| `Card.to_index()` | recompute from config each call | precomputed attribute | **~4.6×** |
| `Deck.state` (observation) | rebuild + copy each call | cached read-only view | **~2.2×** |

Beyond raw speed, returning **read-only views instead of copies** removes a per-call heap allocation,
which dramatically reduces garbage-collection pressure in long-running training loops.

### Reproduce the benchmarks

```bash
uv run --extra dev -m pytest benchmarks/ --benchmark-only
```

---

## FAQ

### What is Toulouse used for?

Toulouse simulates card games and converts their state into NumPy vectors for machine learning. It is
designed for reinforcement learning agents, Monte Carlo Tree Search, and any pipeline that needs fast,
repeatable card-state observations.

### Why are `card.state` and `deck.state` read-only?

They return cached NumPy arrays shared across calls, so allowing mutation would corrupt the cache.
Returning a read-only view avoids an expensive copy on every access. If you need a writable array,
wrap it: `np.array(deck.state)`.

### How does Toulouse stay fast in MCTS?

`deck.copy()` uses copy-on-write: it shares the immutable, cached state array rather than copying it.
Any mutation rebuilds a fresh array, so clones stay correct while cloning stays cheap.

### Can I add my own card game?

Yes. Call `register_card_system("my_game", {"suits": [...], "values": [...], "deck_size": N})`, then
pass `card_system_key="my_game"` to `Deck` and `get_card`.

### Which Python versions and frameworks are supported?

Python 3.9 through 3.12. The observation vectors are plain NumPy arrays, so they work directly with
PyTorch, JAX, TensorFlow, and Gymnasium/Gym environments.

### Is Toulouse type-checked?

Yes — the package is fully type-annotated and ships a `py.typed` marker, so type checkers like mypy
and pyright pick up its types automatically.

---

## Development & Testing

```bash
# Install with dev tooling
uv sync --extra dev

# Run the unit tests
uv run -m pytest

# Run the performance benchmarks
uv run --extra dev -m pytest benchmarks/ --benchmark-only

# Lint
uv run -m pylint $(git ls-files '*.py')
```

See [PUBLISHING.md](PUBLISHING.md) for how releases are published to PyPI via Trusted Publishing.

---

## License

Toulouse is released under the [MIT License](LICENSE).
