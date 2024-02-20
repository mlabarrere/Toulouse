import random
from typing import List, Union
import numpy as np
from .cards import Card, SUITS

class Deck:
    """
    Represents a deck of cards with customizable size and language.
    
    Attributes:
        cards (List[Card]): A list of Card objects in the deck.
        sort_type (str): The sorting type of the deck, 'sort' or 'shuffle'.
        deck_size (int): The size of the deck, default is 52.
        language (str): The language for card representations.
    """
    def __init__(self, 
                 cards: Union[List[Card], None] = None,
                 new: bool = False,
                 sorted: bool = True,
                 deck_size: int = 52, 
                 language: str = 'en'
                 ) -> None:
        self.sorted = sorted
        self.deck_size = deck_size
        self.language = language
        if new:
            self.cards = [Card(value, suit % 4, deck_size, language) for suit in range(4) for value in range(1, (deck_size // 4) + 1)]
        else:
            self.cards = cards if cards is not None else []
        self.state = self.calculate_state()
        self.update_sort()

    def calculate_state(self) -> np.ndarray:
        """
        Calculates the state of the deck based on the cards. 
        
        Returns:
            np.ndarray: A numpy array representing the state of the deck.
        """
        state = np.zeros(self.deck_size, dtype=int)
        for card in self.cards:
            index = (card.suit * (self.deck_size // 4)) + (card.value - 1)
            state[index] = 1
        return state

    def update_state(func):
        """
        Decorator that updates the state and sort of the deck after a method call.
        """
        def wrapper_update_state(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.state = self.calculate_state()
            self.update_sort()
            return result
        return wrapper_update_state

    def __repr__(self) -> str:
        """
        Returns an official string representation of the deck.
        """
        return f"Deck({self.cards})"

    def __iter__(self):
        """
        Allows iteration over the cards in the deck.
        """
        return iter(self.cards)

    def __str__(self) -> str:
        """
        Returns a string representation of the deck.
        """
        return str(self.cards)

    def _get_cards(self, cards: Union[Card, 'Deck', List[Card]]) -> List[Card]:
        """
        Returns a list of cards from the given input.
        
        Args:
            cards: A Card, Deck, or list of Card objects.
            
        Returns:
            List[Card]: A list of Card objects.
        """
        if isinstance(cards, (Card, Deck)):
            return cards.cards if isinstance(cards, Deck) else [cards]
        return cards

    @update_state
    def draw(self, num_cards: int) -> List[Card]:
        """
        Draws a specified number of cards from the deck.
        
        Args:
            num_cards (int): The number of cards to draw.
            
        Returns:
            List[Card]: The drawn cards.
        
        Raises:
            ValueError: If there are not enough cards in the deck to draw.
        """
        if num_cards > len(self.cards):
            raise ValueError("Not enough cards in the deck to draw.")
        
        drawn_cards = self.cards[-num_cards:]
        self.cards = self.cards[:-num_cards]
        return drawn_cards

    @update_state
    def remove(self, cards: Union[Card, List[Card]]) -> None:
        """
        Removes specified cards from the deck.
        
        Args:
            cards: A single Card object or a list of Card objects to remove.
        """
        cards_to_remove = self._get_cards(cards)
        for card in cards_to_remove:
            if card in self.cards:
                self.cards.remove(card)
            else:
                raise ValueError("Card not in deck.")

    @update_state
    def append(self, cards: Union[Card, List[Card]]) -> None:
        """
        Appends specified cards to the deck.
        
        Args:
            cards: A single Card object or a list of Card objects to append.
        """
        cards_to_add = self._get_cards(cards)
        for card in cards_to_add:
            if card not in self.cards:
                self.cards.append(card)
            else:
                raise ValueError("Card already in deck.")

    def update_sort(self) -> None:
        """
        Sorts or shuffles the cards in the deck based on the sort_type attribute.
        """
        if self.sorted:
            self.cards.sort(key=lambda card: (card.suit, card.value))
        else:
            random.shuffle(self.cards)

    def __len__(self)-> int:
        """
        Returns the number of cards in the deck.
        """
        return len(self.cards)

    @classmethod
    def from_state(cls, state_array: np.ndarray, deck_size: int = 52, language: str = 'en') -> 'Deck':
        """
        Creates a deck from a state represented by a NumPy array.
        
        Args:
            state_array (np.ndarray): The state array of the deck.
            deck_size (int): The size of the deck.
            language (str): The language for card representations.
            
        Returns:
            Deck: A new Deck instance.
            
        Raises:
            ValueError: If the state array does not match the expected length.
        """
        if state_array.shape[0] != deck_size:
            raise ValueError(f"State array must be of length {deck_size}.")

        cards = []
        for i in range(state_array.shape[0]):
            if state_array[i] == 1:
                suit = i // (deck_size // 4)
                value = (i % (deck_size // 4)) + 1
                cards.append(Card(value, suit, deck_size, language))

        return cls(cards=cards, deck_size=deck_size, language=language)
