# Toulouse: A Python Package for Card Games and Machine Learning 🃏

Toulouse is a Python package designed for creating and manipulating standard playing cards and decks, with a focus on applications in machine learning. It offers a simple yet powerful interface for card game simulation and analysis, providing binary representations of cards suitable for machine learning models.

## Installation

Toulouse can be installed easily via pip. Open your terminal and run the following command:

```bash
pip install toulouse
```

Ensure you have Python 3.8 or higher installed on your system to use Toulouse.

## Classes

### Card

The `Card` class represents a single playing card.

**Inputs:**

- `value` (int): The value of the card (1-13, where 1 is Ace, 11 is Jack, 12 is Queen, and 13 is King in a case of 52 deck size).
- `suit` (int): The suit of the card (0-3, representing Spades ♠️, Hearts ♥️, Diamonds ♦️, and Clubs ♣️, respectively).
- `deck_size` (int, optional): The size of the deck, default is 52.
- `language` (str, optional): The language for card representation, default is 'en' (English). 

You can choose between 'en' 🇬🇧, 'fr' 🇫🇷, 'it' 🇮🇹, 'es' 🇪🇸 and 'de' 🇩🇪

### Deck

The `Deck` class represents a collection of `Card` objects.

**Inputs:**

- `cards` (List[Card], optional): A list of `Card` objects to initialize the deck. Default is `None`, which initializes a standard deck.
- `new` (bool, optional): If True, creates a new standard deck. Default is False.
- `sort_type` (str, optional): Determines how the deck is sorted or shuffled ('sort' for sorted, 'shuffle' for shuffled). Default is 'sort'.
- `deck_size` (int, optional): The size of the deck, affects card distribution if `new` is True. Default is 52.
- `language` (str, optional): The language for card representations within the deck. Default is 'en'.

## Use Cases

### Manipulating Cards and Decks

- **Creating a New Deck:** `deck = Deck(new=True)`
- **Drawing Cards from the Deck:** `drawn_cards = deck.draw(5)`
- **Adding a Card to the Deck:** `deck.append(Card(1, 0))` (Adds an Ace of Spades)
- **Removing a Card from the Deck:** `deck.remove(some_card)`
- **Shuffling the Deck:** Set `deck.sort_type = 'shuffle'` then `deck.update_sort()`

### Machine Learning Use Case

For machine learning applications, especially in game simulation and strategy analysis, the binary representation of cards can be utilized as features for models.

**Binary Representation:**

Each card can be represented as a binary vector where only one element is set to 1, and the rest are 0. This one-hot encoding allows the model to easily process card information.

```python
card = Card(1, 0)  # Ace of Spades
binary_representation = card.state
```

This representation can be used as input for neural networks, decision trees, or any other machine learning model to analyze card games or simulate strategies.

### Example: Training a Model

Imagine you're building a model to predict the outcome of a card game. You could use the binary representations of drawn cards as features:

```python
features = np.array([card.state for card in drawn_cards])
# Assume labels are the outcomes you want to predict
model.fit(features, labels)
```

This simplistic example shows how you might begin to incorporate card data into a machine learning model. For more complex games or analysis, you might combine features from multiple cards, include game state information, or use embeddings.

---

This README provides a basic overview of the Toulouse package, its installation, class inputs, and use cases.