import random
from typing import List, Union
import numpy as np
from .cards import Card, CARD_SYSTEMS # Import Card class and CARD_SYSTEMS definitions

class Deck:
    """
    Represents a deck of cards, primarily designed for the "italian_40" card system.
    
    The Deck class handles card creation for a full deck, shuffling, drawing,
    and state representation based on the "italian_40" system defined in CARD_SYSTEMS.
    While the underlying structure could support other systems, the current focus is singular.

    Attributes:
        cards (List[Card]): A list of Card objects currently in the deck.
        card_system_key (str): Identifier for the card system this deck adheres to 
                               (e.g., 'standard_52', 'italian_40'). This determines the type
                               of cards in the deck and its total size.
        card_system (dict): The configuration dictionary for the current card system, loaded
                            from CARD_SYSTEMS.
        language (str): The language used for string representations of cards created by/for this deck,
                        if not overridden by individual cards.
        sorted (bool): If True, the deck is kept sorted (e.g., after card removal/addition). 
                       If False, the deck is shuffled after such operations (or remains shuffled).
        state (np.ndarray): A binary numpy array representing the presence of cards in the deck.
                            The size of this array is determined by the `deck_size` of the `card_system`.
    """
    def __init__(self, 
                 cards: Union[List[Card], None] = None,
                 new: bool = False,
                 sorted_deck: bool = True, # Parameter name changed from 'sorted'
                 language: str = 'en'
                 ) -> None:
        """Initializes a Deck instance. The card system is fixed to "italian_40".

        Args:
            cards (Union[List[Card], None], optional): A list of Card objects to populate the deck.
                If None, the deck is empty unless `new` is True. Defaults to None.
            new (bool, optional): If True, a full new deck of "italian_40" cards is generated.
                Defaults to False.
            sorted_deck (bool, optional): Determines if the deck should be sorted initially (True)
                or shuffled (False). Defaults to True.
            language (str, optional): The language for card string representations, primarily for
                cards created when `new=True`. Defaults to 'en'.
        """
        self.card_system_key = "italian_40" # Hardcoded card system
        # Load the card system properties (always "italian_40")
        self.card_system = CARD_SYSTEMS[self.card_system_key]
        
        self.language = language # Sets the default language for cards in this deck.
        self.sorted = sorted_deck # Determines if the deck is kept sorted or shuffled.

        if new:
            # Generate a new, full set of cards based on the selected card system.
            self.cards = []
            num_suits = self.card_system["num_suits"]
            values_map = self.card_system["values_map"] # Use the system's defined card values.
            # Iterate through each suit and each value defined in the system's values_map.
            for suit_idx in range(num_suits):
                for value in values_map:
                    # Card() no longer takes card_system_key
                    self.cards.append(Card(value=value, suit=suit_idx, language=self.language))
        else:
            # Use provided cards or initialize an empty deck.
            self.cards = cards if cards is not None else []
            if self.cards:
                # Ensure all provided cards are compatible with the deck's system.
                # (Current implementation detail: This check is basic. More robust validation could be added.)
                first_card = self.cards[0]
                if first_card.card_system_key != self.card_system_key:
                    # This warning or error could be more stringent. For now, it implies that the
                    # deck's `card_system_key` is authoritative.
                    print(f"Warning: Deck initialized with card_system_key '{self.card_system_key}' but first card provided is from '{first_card.card_system_key}'. Mismatches may occur.")
                
                # Standardize the language of provided cards to the deck's language.
                # This ensures consistency in string representations if cards are later printed via the deck.
                for card in self.cards:
                    card.language = self.language

        self.state = self.calculate_state() # Initialize the binary state representation of the deck.
        self.update_sort() # Sort or shuffle the deck based on the `sorted` attribute.

    def calculate_state(self) -> np.ndarray:
        """
        Calculates the binary state vector of the deck.
        The vector's length is the `deck_size` of the current `card_system`.
        A '1' at a position indicates the card with that `calculate_index()` is in the deck.
        
        Returns:
            np.ndarray: A numpy array representing the binary state of the deck.
        """
        # Initialize a zero vector of the size of the full deck for the current system.
        state = np.zeros(self.card_system["deck_size"], dtype=int)
        for card in self.cards:
            # It's crucial that cards being added to the state belong to the same system
            # as the deck itself, otherwise, their indices might be out of bounds or incorrect.
            if card.card_system_key != self.card_system_key:
                raise ValueError(f"Card {card} with system '{card.card_system_key}' does not match deck system '{self.card_system_key}'. Cannot calculate state.")
            # Mark the presence of the card by setting the bit at its unique index to 1.
            state[card.calculate_index()] = 1
        return state

    def update_state(func):
        """
        Decorator to automatically update the deck's state and sorting after method calls
        that modify the deck's contents (e.g., draw, remove, append).
        """
        def wrapper_update_state(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.state = self.calculate_state() # Recalculate the binary state.
            self.update_sort() # Re-apply sorting or shuffling.
            return result
        return wrapper_update_state

    def __repr__(self) -> str:
        """
        Returns an official, unambiguous string representation of the deck, useful for debugging.
        """
        return f"Deck(cards={self.cards}, card_system_key='{self.card_system_key}', language='{self.language}')"

    def __iter__(self):
        """
        Allows iteration over the cards currently in the deck.
        """
        return iter(self.cards)

    def __str__(self) -> str:
        """
        Returns a user-friendly string representation of the deck, showing string versions of all cards.
        """
        return str([str(card) for card in self.cards])


    def _get_cards(self, cards_input: Union[Card, 'Deck', List[Card]]) -> List[Card]:
        """
        Internal helper method to consistently convert input (single Card, another Deck, or a list)
        into a list of Card objects.
        
        Args:
            cards_input: A Card, Deck object, or a list of Card objects.
            
        Returns:
            List[Card]: A flat list of Card objects.
        """
        if isinstance(cards_input, Card):
            return [cards_input]
        elif isinstance(cards_input, Deck):
            return cards_input.cards
        elif isinstance(cards_input, list):
            return cards_input
        else:
            raise TypeError(f"Input must be Card, Deck, or List[Card], not {type(cards_input)}")

    @update_state # Decorator ensures state and sort are updated after drawing.
    def draw(self, num_cards: int) -> List[Card]:
        """
        Draws a specified number of cards from the "top" of the deck (end of the list).
        
        Args:
            num_cards (int): The number of cards to draw.
            
        Returns:
            List[Card]: The list of drawn cards.
        
        Raises:
            ValueError: If `num_cards` is negative or more cards are requested than available.
        """
        if num_cards < 0:
            raise ValueError("Number of cards to draw cannot be negative.")
        if num_cards > len(self.cards):
            raise ValueError(f"Not enough cards in the deck to draw {num_cards}. Deck has {len(self.cards)} cards.")
        
        # Cards are drawn from the end of the list (self.cards).
        # If the deck is sorted (e.g., Ace low, King high), this means higher value cards are drawn first.
        # If shuffled, the order is random.
        drawn_cards = self.cards[-num_cards:]
        self.cards = self.cards[:-num_cards]
        return drawn_cards

    @update_state # Decorator ensures state and sort are updated after removal.
    def remove(self, cards_to_remove: Union[Card, List[Card]]) -> None:
        """
        Removes specified cards from the deck.
        
        Args:
            cards_to_remove: A single Card object or a list of Card objects to be removed.
        
        Raises:
            ValueError: If a card to remove is not found or if a card is from a different system.
        """
        actual_cards_to_remove = self._get_cards(cards_to_remove)
        
        for card_to_remove in actual_cards_to_remove:
            # Critical check: Ensure the card being removed matches the deck's card system.
            # This relies on Card.__eq__ which checks value, suit, AND card_system_key.
            if card_to_remove.card_system_key != self.card_system_key:
                # This error indicates an attempt to remove a card that fundamentally
                # cannot belong to this deck type (e.g., removing a standard Ace of Spades
                # from an Italian 40-card deck).
                raise ValueError(f"Cannot remove card {card_to_remove} with system '{card_to_remove.card_system_key}' from deck with system '{self.card_system_key}'.")
            try:
                self.cards.remove(card_to_remove) # list.remove() uses Card.__eq__()
            except ValueError:
                # This means the card (matching value, suit, and system) was not found.
                raise ValueError(f"Card {card_to_remove} not found in deck.")

    @update_state # Decorator ensures state and sort are updated after appending.
    def append(self, cards_to_add: Union[Card, List[Card]]) -> None:
        """
        Appends specified cards to the "bottom" of the deck (end of the list).
        
        Args:
            cards_to_add: A single Card object or a list of Card objects to append.

        Raises:
            ValueError: If a card to add is already in the deck or is from a different card system.
        """
        actual_cards_to_add = self._get_cards(cards_to_add)
        for card_to_add in actual_cards_to_add:
            # Critical check: Ensure the card being added is compatible with the deck's card system.
            if card_to_add.card_system_key != self.card_system_key:
                 raise ValueError(f"Cannot add card {card_to_add} with system '{card_to_add.card_system_key}' to deck with system '{self.card_system_key}'.")
            # Check if an identical card (same value, suit, and system) is already present.
            if card_to_add in self.cards: # Relies on Card.__eq__
                raise ValueError(f"Card {card_to_add} already in deck.")
            self.cards.append(card_to_add)

    def update_sort(self) -> None:
        """
        Sorts the deck if `self.sorted` is True, or shuffles it if `self.sorted` is False.
        Sorting is based on the canonical index of each card, ensuring a consistent order
        (e.g., by suit, then by value within the suit according to `values_map`).
        """
        if self.sorted:
            # Sorts cards based on their unique index returned by card.calculate_index().
            # This provides a canonical ordering for any given card system.
            self.cards.sort(key=lambda card: card.calculate_index())
        else:
            # Shuffles the cards randomly.
            random.shuffle(self.cards)

    def __len__(self)-> int:
        """
        Returns the number of cards currently in the deck.
        """
        return len(self.cards)

    @classmethod
    def from_state(cls, state_array: np.ndarray, language: str = 'en') -> 'Deck':
        """
        Class method to create a Deck instance from a binary state array.
        The state array represents a full deck of the "italian_40" system, where '1' indicates
        a card's presence. The card system is fixed to "italian_40".
        
        Args:
            state_array (np.ndarray): The binary state array. Its length must match the
                                      `deck_size` of the "italian_40" system (40).
            language (str, optional): The language for the cards in the created deck. Defaults to 'en'.
            
        Returns:
            Deck: A new Deck instance populated with cards indicated by the state array.
            
        Raises:
            ValueError: If `state_array` length doesn't match the system's `deck_size`,
                        or if an index in the state array implies an invalid card for the system.
        """
        card_system_key = "italian_40" # Hardcoded
        card_system = CARD_SYSTEMS[card_system_key]
        expected_deck_size = card_system["deck_size"]

        # Validate state array length against the "italian_40" system's deck size.
        if state_array.shape[0] != expected_deck_size:
            raise ValueError(f"State array length {state_array.shape[0]} must match deck size {expected_deck_size} for system '{card_system_key}'.")

        cards = []
        num_values_per_suit = len(card_system["values_map"])

        for i in range(state_array.shape[0]):
            if state_array[i] == 1:
                suit_idx = i // num_values_per_suit
                value_map_idx = i % num_values_per_suit
                
                if suit_idx >= card_system["num_suits"]:
                     raise ValueError(f"Invalid suit index {suit_idx} derived from state for system '{card_system_key}' at overall index {i}.")
                if value_map_idx >= len(card_system["values_map"]):
                     raise ValueError(f"Invalid value map index {value_map_idx} derived from state for system '{card_system_key}' at overall index {i}.")

                value = card_system["values_map"][value_map_idx]
                # Card() no longer takes card_system_key
                cards.append(Card(value, suit_idx, language=language))
        
        # Cards created from state are inherently sorted by their canonical index.
        # Deck() no longer takes card_system_key
        return cls(cards=cards, sorted_deck=True, language=language)

# Example usage section for quick testing or demonstration.
if __name__ == '__main__':
    print("\n--- Italian 40-card Deck (Italian, Shuffled) ---")
    # Deck() no longer takes card_system_key
    deck_it_shuffled = Deck(new=True, language='it', sorted_deck=False) 
    print(f"Deck size: {len(deck_it_shuffled)}")
    print(f"First 5 cards (shuffled): {[str(c) for c in deck_it_shuffled.cards[:5]]}")
    
    # Test state calculation for a specific Italian card
    # Card() no longer takes card_system_key
    fante_di_denari_it = Card(value=8, suit=0, language='it') 
    print(f"Index of {fante_di_denari_it}: {fante_di_denari_it.calculate_index()}")

    # Test Deck.from_state for Italian deck
    print("\n--- Deck from State (Italian) ---")
    state_sample_it = np.zeros(40, dtype=int)
    state_sample_it[0] = 1 
    state_sample_it[17] = 1
    state_sample_it[39] = 1
    
    # Deck.from_state() no longer takes card_system_key
    deck_from_state_it = Deck.from_state(state_sample_it, language='it')
    print(f"Deck created from state (Italian): {deck_from_state_it}")
    print(f"Number of cards in deck from state: {len(deck_from_state_it.cards)}")
    for card in deck_from_state_it.cards:
        print(f"  Card: {card}, Index: {card.calculate_index()}")

    # Test adding a card from a different system (should fail if card_system_key was still a thing in Card)
    # This test is less relevant now as Card itself is hardcoded to "italian_40".
    # An attempt to create a Card that *would* be from another system will just be an "italian_40" card.
    # The check `if card_to_add.card_system_key != self.card_system_key:` in append will always pass.
    # print("\n--- Testing Error Handling for system mismatch (less relevant now) ---")
    # try:
    #     # This card will be created as an "italian_40" card due to hardcoding in Card.__init__
    #     # standard_card_to_add = Card(value=1, suit=0, language='en') 
    #     # deck_it_shuffled.append(standard_card_to_add) # This would only fail if Card could have a *different* system_key
    # except ValueError as e:
    #     print(f"Error (expected if Card could have other systems): {e}")


    # Test removing a card not present
    print("\n--- Testing Error Handling for removing non-existent card ---")
    try:
        # Card() no longer takes card_system_key
        card_not_present = Card(value=2, suit=0, language='it') 
        if card_not_present in deck_it_shuffled.cards: # Should not be removed if this is the test case
             pass # deck_it_shuffled.remove(card_not_present) # Don't remove if we want to test removing non-present
        
        # Ensure it is indeed not present before trying to remove for the test
        temp_deck_cards_str = {str(c) for c in deck_it_shuffled.cards}
        if str(card_not_present) in temp_deck_cards_str:
            # This means the card *is* in the deck, so the test for "not found" might not trigger as intended
            # unless we ensure it's a card that truly isn't in the deck (e.g. after drawing all cards)
            print(f"Warning: Card {card_not_present} is actually in the deck. Test might not behave as expected for 'not found'.")


        deck_it_shuffled.remove(card_not_present) # Try to remove it
    except ValueError as e:
        print(f"Error (expected if card truly not in deck): {e}")
