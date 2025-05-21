# Toulouse: A Python Package for Card Games and Machine Learning 🃏

Toulouse is a Python package designed for creating and manipulating playing cards and decks. It supports various card game systems, including standard 52-card decks and Italian 40-card decks, with a focus on applications in machine learning. It offers a simple yet powerful interface for card game simulation and analysis, providing binary representations of cards and decks suitable for machine learning models.

## Installation

Toulouse can be installed easily via pip. Open your terminal and run the following command:

```bash
pip install toulouse
```

Ensure you have Python 3.8 or higher installed on your system to use Toulouse.

## Key Concepts

### Card Systems

Toulouse now supports different types of card games through `card_system_key`. Each system defines its own set of suits, card values, and deck size. Currently supported systems include:
-   `"standard_52"`: For traditional 52-card games.
-   `"italian_40"`: For games using a 40-card Italian deck (e.g., Scopa, Briscola).

## Classes

### Card

The `Card` class represents a single playing card, adaptable to different card systems.

**Inputs:**

-   `value` (int): The numerical value of the card according to its system's `values_map`. For example:
    -   In a `"standard_52"` deck: 1 (Ace) to 13 (King).
    -   In an `"italian_40"` deck: 1 (Asso) to 7, then 8 (Fante/Jack), 9 (Cavallo/Knight), 10 (Re/King).
-   `suit` (int): The suit index of the card (e.g., 0-3 for a 4-suit system). The meaning of the index (e.g., 0 = Spades or 0 = Denari) is defined by the `card_system_key`.
-   `card_system_key` (str, optional): Determines the fundamental properties of the card, such as its valid value range, suit names, and how it fits into a deck's structure (e.g., total deck size for binary state representation). Defaults to `"standard_52"`.
    -   Example values: `"standard_52"`, `"italian_40"`.
-   `language` (str, optional): The language for the card's string representation (e.g., "Ace of Spades" vs "As de Piques"). Defaults to `'en'` (English). Supported languages vary by card system but generally include 'en' 🇬🇧, 'fr' 🇫🇷, 'it' 🇮🇹, 'es' 🇪🇸, and 'de' 🇩🇪 where applicable.

**Example:**

```python
from toulouse import Card

# Standard 52-card deck examples
ace_of_spades = Card(value=1, suit=0, card_system_key="standard_52", language='en')
print(ace_of_spades)  # Output: Ace of Spades

king_of_hearts_fr = Card(value=13, suit=1, card_system_key="standard_52", language='fr')
print(king_of_hearts_fr) # Output: Roi de Cœurs

# Italian 40-card deck example
asso_di_denari = Card(value=1, suit=0, card_system_key="italian_40", language='it')
print(asso_di_denari)  # Output: Asso di Denari

fante_di_coppe_en = Card(value=8, suit=1, card_system_key="italian_40", language='en') # Fante (Jack) of Cups (Coppe)
print(fante_di_coppe_en) # Output: Jack of Cups
```

### Deck

The `Deck` class represents a collection of `Card` objects, also adaptable to different card systems.

**Inputs:**

-   `cards` (List[Card], optional): A list of `Card` objects to initialize the deck. If provided, these cards should ideally belong to the specified `card_system_key`. Defaults to `None`.
-   `new` (bool, optional): If `True`, creates a full new deck based on the `card_system_key`. Defaults to `False`.
-   `sorted_deck` (bool, optional): Determines if the deck is initially sorted (by canonical card index) or shuffled. Defaults to `True` (sorted).
-   `card_system_key` (str, optional): Defines the type of deck to be created (e.g., a 52-card standard deck or a 40-card Italian deck). This determines the kinds of cards, total number of cards, etc. Defaults to `"standard_52"`.
-   `language` (str, optional): The default language for cards created by this deck (if `new=True`) and for some deck-level representations. Defaults to `'en'`.

**Example:**

```python
from toulouse import Deck, Card

# Creating a new standard 52-card deck, sorted
standard_deck = Deck(new=True, card_system_key="standard_52", language='en')
print(f"Standard deck size: {len(standard_deck.cards)}") # Output: 52
print(f"First card: {standard_deck.cards[0]}") # Output: Ace of Spades

# Creating a new Italian 40-card deck, shuffled, in Italian
italian_deck_shuffled = Deck(new=True, card_system_key="italian_40", language='it', sorted_deck=False)
print(f"Italian deck size: {len(italian_deck_shuffled.cards)}") # Output: 40
print(f"First 5 cards (shuffled): {[str(c) for c in italian_deck_shuffled.cards[:5]]}")
```

## Use Cases

### Manipulating Cards and Decks

-   **Creating a New Deck:** `deck = Deck(new=True, card_system_key="standard_52")` or `deck_it = Deck(new=True, card_system_key="italian_40")`
-   **Drawing Cards from the Deck:** `drawn_cards = deck.draw(5)`
-   **Adding a Card to the Deck:** `deck.append(Card(value=1, suit=0, card_system_key="standard_52"))` (Adds an Ace of Spades to a standard deck)
-   **Removing a Card from the Deck:** `deck.remove(some_card)`
-   **Shuffling/Sorting the Deck:**
    -   To shuffle: `deck.sorted = False; deck.update_sort()`
    -   To sort: `deck.sorted = True; deck.update_sort()`

### Machine Learning Use Case

For machine learning applications, especially in game simulation and strategy analysis, the binary representation of cards and decks can be utilized as features for models.

**Binary Representation:**

Each card can be represented as a binary vector (`card.state`) where only one element is set to 1, and the rest are 0. The length of this vector is determined by the `deck_size` of the card's system (e.g., 52 for `"standard_52"`, 40 for `"italian_40"`).
The deck also has a `deck.state` which is a binary vector indicating the presence of cards in the deck.

```python
import numpy as np
from toulouse import Card, Deck

# Standard card state
ace_spades = Card(1, 0, card_system_key="standard_52")
print(f"Ace of Spades state (len {len(ace_spades.state)}): {ace_spades.state.argmax()}") # Index 0 in a 52-element array

# Italian card state
asso_denari = Card(1, 0, card_system_key="italian_40", language='it')
print(f"Asso di Denari state (len {len(asso_denari.state)}): {asso_denari.state.argmax()}") # Index 0 in a 40-element array

# Deck state
standard_deck = Deck(new=True, card_system_key="standard_52")
print(f"Standard deck state sum (all cards present): {standard_deck.state.sum()}") # Output: 52

italian_deck = Deck(new=True, card_system_key="italian_40")
print(f"Italian deck state sum (all cards present): {italian_deck.state.sum()}") # Output: 40
```

This one-hot encoding allows models to easily process card information.

### Example: Training a Model

Imagine you're building a model to predict the outcome of a card game. You could use the binary representations of drawn cards as features:

```python
# Assume drawn_cards is a list of Card objects from a specific deck system
features = np.array([card.state for card in drawn_cards])
# Assume labels are the outcomes you want to predict (e.g., win/loss)
# model.fit(features, labels) # Example placeholder for model training
```

This simplistic example shows how you might begin to incorporate card data into a machine learning model. For more complex games or analysis, you might combine features from multiple cards, include game state information, or use embeddings.

---

This README provides an overview of the Toulouse package, its installation, class inputs, and use cases, including its new multi-card system capabilities.