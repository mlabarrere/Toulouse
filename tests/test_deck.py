import pytest
from toulouse.cards import Card, CARD_SYSTEMS
from toulouse.deck import Deck
import numpy as np

# --- Tests for "italian_40" Deck (Default System) ---

def test_deck_initialization():
    """Test Deck initialization (system is "italian_40" by default)."""
    deck = Deck(new=True, language='it', sorted_deck=True)
    assert len(deck.cards) == 40
    assert deck.card_system_key == 'italian_40' # Verify hardcoded system
    assert deck.language == 'it'
    assert deck.card_system["deck_size"] == 40
    if deck.cards:
        assert deck.cards[0].card_system_key == 'italian_40' # Cards also use hardcoded system
        assert str(deck.cards[0]) == "Asso di Denari" 
        assert str(deck.cards[39]) == "Re di Bastoni" 

def test_deck_draw():
    """Test drawing cards from an Italian 40-card deck."""
    deck = Deck(new=True, language='it')
    num_to_draw = 3
    drawn_cards = deck.draw(num_to_draw)
    assert len(drawn_cards) == num_to_draw
    assert len(deck.cards) == 40 - num_to_draw
    for card in drawn_cards:
        assert card.card_system_key == 'italian_40'
        assert card.language == 'it'

def test_deck_state():
    """Test the state representation of an Italian 40-card deck."""
    deck = Deck(new=True, sorted_deck=True) # Default lang 'en'
    state = deck.state
    assert isinstance(state, np.ndarray)
    assert state.shape[0] == 40
    assert state.sum() == 40 # All cards present

    asso_denari = Card(value=1, suit=0) # Card system is "italian_40" by default
    assert state[asso_denari.calculate_index()] == 1

    re_bastoni = Card(value=10, suit=3)
    assert state[re_bastoni.calculate_index()] == 1
    
    fante_coppe = Card(value=8, suit=1)
    assert state[fante_coppe.calculate_index()] == 1


def test_deck_from_state():
    """Test creating an Italian 40-card deck from a state array."""
    state_array = np.zeros(40, dtype=int)
    state_array[0] = 1  # Asso di Denari
    state_array[17] = 1 # Fante di Coppe
    state_array[39] = 1 # Re di Bastoni
        
    # Deck.from_state no longer takes card_system_key
    deck = Deck.from_state(state_array, language='it')
    assert len(deck.cards) == 3
    assert deck.card_system_key == 'italian_40'
    
    # Card no longer takes card_system_key
    expected_card1 = Card(value=1, suit=0, language='it') 
    expected_card2 = Card(value=8, suit=1, language='it') 
    expected_card3 = Card(value=10, suit=3, language='it')

    assert expected_card1 in deck.cards
    assert expected_card2 in deck.cards
    assert expected_card3 in deck.cards


# --- General Deck Operation Tests ---

def test_deck_append_and_remove():
    """Test appending and removing cards for the Italian 40-card system."""
    # Deck() no longer takes card_system_key
    deck = Deck(new=True, sorted_deck=False) 
    
    # Test remove
    deck_for_remove = Deck(new=True, sorted_deck=True)
    card_present = deck_for_remove.cards[0] 
    len_before_remove = len(deck_for_remove)
    deck_for_remove.remove(card_present)
    assert len(deck_for_remove.cards) == len_before_remove - 1
    assert card_present not in deck_for_remove.cards

    # Test append
    deck_for_append = Deck(cards=[]) 
    # Card() no longer takes card_system_key
    value_to_add = CARD_SYSTEMS["italian_40"]["values_map"][0] 
    card_to_add = Card(value=value_to_add, suit=0) 
    
    deck_for_append.append(card_to_add)
    assert len(deck_for_append.cards) == 1
    assert card_to_add in deck_for_append.cards


def test_deck_shuffle():
    """Test shuffling the cards in an Italian 40-card deck."""
    # Deck() no longer takes card_system_key
    deck = Deck(new=True, sorted_deck=True)
    original_order = list(deck.cards) 
    
    deck.sorted = False 
    deck.update_sort() 
    
    shuffled_order = deck.cards
    assert len(original_order) == len(shuffled_order)
    assert original_order != shuffled_order 
    for card in original_order:
        assert card in shuffled_order


# --- Error Handling and Edge Cases ---

def test_deck_draw_too_many():
    """Test drawing more cards than are in the Deck raises ValueError (Italian 40)."""
    # Deck() no longer takes card_system_key
    deck = Deck(new=True) 
    with pytest.raises(ValueError, match="Not enough cards in the deck to draw 41"):
        deck.draw(41)

def test_deck_append_duplicate_card():
    """Test appending a card already in the deck raises ValueError (Italian 40)."""
    # Deck() no longer takes card_system_key
    deck = Deck(new=True, sorted_deck=True)
    card_to_add = deck.cards[0] 
    with pytest.raises(ValueError, match="already in deck"): 
        deck.append(card_to_add)

def test_deck_remove_card_not_in_deck():
    """Test removing a card not in the deck raises ValueError (Italian 40)."""
    # Deck() no longer takes card_system_key
    deck = Deck(cards=[]) 
    # Card() no longer takes card_system_key
    card_not_in_deck = Card(value=1, suit=0) 
    with pytest.raises(ValueError, match="not found in deck"):
        deck.remove(card_not_in_deck)

def test_deck_from_state_invalid_size():
    """Test Deck.from_state with state array of incorrect length for Italian 40."""
    invalid_state_size = 52 
    state_array = np.zeros(invalid_state_size, dtype=int)
    expected_size = CARD_SYSTEMS["italian_40"]["deck_size"] 
    # Deck.from_state no longer takes card_system_key
    with pytest.raises(ValueError, match=f"State array length {invalid_state_size} must match deck size {expected_size} for system 'italian_40'."):
        Deck.from_state(state_array)


def test_deck_len_magic_method():
    """Test the __len__ magic method of the Deck for Italian 40."""
    # Deck() no longer takes card_system_key
    deck_it = Deck(new=True)
    assert len(deck_it) == 40
    deck_it.draw(3)
    assert len(deck_it) == 37

def test_deck_old_parameters_compatibility():
    """
    Test backward compatibility, reflecting 'italian_40' as the default.
    """
    # Deck() no longer takes card_system_key
    deck_default = Deck(new=True) 
    assert len(deck_default.cards) == 40
    assert deck_default.card_system_key == 'italian_40'
    assert deck_default.language == 'en' # Default language in Deck constructor
    if deck_default.cards:
        assert str(deck_default.cards[0]) == "Ace of Coins"

    # Test language override for the deck and appending card with different language
    deck_shuffled_it_lang = Deck(new=True, language='it', sorted_deck=False) # Deck is shuffled
    assert deck_shuffled_it_lang.language == 'it'
    if deck_shuffled_it_lang.cards:
        # Check if Asso di Denari is present (it should be, with 'it' language)
        found_asso_denari = False
        for card in deck_shuffled_it_lang.cards:
            if card.value == 1 and card.suit == 0 and card.language == 'it':
                found_asso_denari = True
                break
        assert found_asso_denari, "Asso di Denari with 'it' language should be in the new 'it' language deck."

    card_to_append_en = Card(value=1, suit=1, language='en') # Ace of Cups (value 1, suit 1)
    
    # Find and remove the card with value 1, suit 1 (Asso di Coppe, lang 'it') from the deck
    # to ensure we can append our 'en' version without a duplicate error.
    card_to_remove_val = 1
    card_to_remove_suit = 1
    original_card_in_deck = None
    for card in deck_shuffled_it_lang.cards:
        if card.value == card_to_remove_val and card.suit == card_to_remove_suit:
            original_card_in_deck = card
            break
    
    if original_card_in_deck:
        assert original_card_in_deck.language == 'it' # Should be 'it' from deck init
        deck_shuffled_it_lang.remove(original_card_in_deck)
        assert original_card_in_deck not in deck_shuffled_it_lang.cards
    else:
        # This case should not be reached if the deck is full and card value/suit are valid
        pytest.fail(f"Card with value {card_to_remove_val}, suit {card_to_remove_suit} not found for removal.")

    # Append the 'en' version of the card
    deck_shuffled_it_lang.append(card_to_append_en)
    
    # Verify the appended card is present and retains its 'en' language
    found_appended_card_en = False
    for card in deck_shuffled_it_lang.cards:
        if card is card_to_append_en: # Check for object identity
            assert card.language == 'en'
            assert str(card) == "Ace of Cups"
            found_appended_card_en = True
            break
    assert found_appended_card_en, "Appended 'en' language card not found in deck or its properties are incorrect."
