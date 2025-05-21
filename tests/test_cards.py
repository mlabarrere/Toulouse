import pytest
from toulouse.cards import Card, CARD_SYSTEMS
import numpy as np

# --- Tests for Standard 52-card System ---

def test_card_initialization_standard():
    """Test Card initialization with standard 52-card system."""
    card = Card(value=1, suit=0, card_system_key='standard_52', language='en')
    assert card.value == 1
    assert card.suit == 0
    assert card.card_system_key == 'standard_52'
    assert card.language == 'en'
    assert card.card_system["deck_size"] == 52

def test_card_string_representation_standard_en():
    """Test string representation for standard system in English."""
    card = Card(value=1, suit=0, card_system_key='standard_52', language='en') # Ace of Spades
    assert str(card) == "Ace of Spades"

def test_card_string_representation_standard_fr():
    """Test string representation for standard system in French."""
    card = Card(value=11, suit=1, card_system_key='standard_52', language='fr') # Jack of Hearts
    assert str(card) == "Valet de Cœurs"

def test_card_equality_standard():
    """Test equality for standard cards."""
    card1 = Card(value=10, suit=2, card_system_key='standard_52')
    card2 = Card(value=10, suit=2, card_system_key='standard_52')
    card3 = Card(value=11, suit=2, card_system_key='standard_52')
    assert card1 == card2
    assert card1 != card3

def test_card_state_standard():
    """Test state (binary representation) for a standard card."""
    card = Card(value=1, suit=0, card_system_key='standard_52') # Ace of Spades
    state = card.state
    assert isinstance(state, np.ndarray)
    assert state.shape[0] == 52
    assert state.sum() == 1
    # Index for Ace of Spades (value 1, suit 0) in standard_52: values_map.index(1)=0. 0 + 0 * 13 = 0
    assert state[0] == 1 

    card_king_clubs = Card(value=13, suit=3, card_system_key='standard_52') # King of Clubs
    state_kc = card_king_clubs.state
    # Index for King of Clubs (value 13, suit 3): values_map.index(13)=12. 12 + 3 * 13 = 12 + 39 = 51
    assert state_kc[51] == 1

def test_card_addition_standard():
    """Test adding values of two standard cards."""
    card1 = Card(value=10, suit=2, card_system_key='standard_52')
    card2 = Card(value=3, suit=1, card_system_key='standard_52')
    assert card1 + card2 == 13

def test_card_comparison_standard():
    """Test comparison (<, >) of standard cards."""
    ace_spades = Card(value=1, suit=0, card_system_key='standard_52')
    king_spades = Card(value=13, suit=0, card_system_key='standard_52')
    ace_hearts = Card(value=1, suit=1, card_system_key='standard_52')
    assert ace_spades < king_spades
    assert king_spades > ace_spades
    assert ace_spades < ace_hearts # Same value, suit 0 < suit 1

# --- Tests for Italian 40-card System ---

def test_card_initialization_italian():
    """Test Card initialization with Italian 40-card system."""
    card = Card(value=1, suit=0, card_system_key='italian_40', language='it') # Asso di Denari
    assert card.value == 1
    assert card.suit == 0
    assert card.card_system_key == 'italian_40'
    assert card.language == 'it'
    assert card.card_system["deck_size"] == 40

def test_card_string_representation_italian_it():
    """Test string representation for Italian system in Italian."""
    asso_denari = Card(value=1, suit=0, card_system_key='italian_40', language='it')
    assert str(asso_denari) == "Asso di Denari"
    
    fante_coppe = Card(value=8, suit=1, card_system_key='italian_40', language='it')
    assert str(fante_coppe) == "Fante di Coppe"

    re_bastoni = Card(value=10, suit=3, card_system_key='italian_40', language='it')
    assert str(re_bastoni) == "Re di Bastoni"

    # Test with English language for Italian cards (if names provided)
    asso_denari_en = Card(value=1, suit=0, card_system_key='italian_40', language='en')
    assert str(asso_denari_en) == "Ace of Coins" # Based on CARD_SYSTEMS definition

def test_card_state_italian():
    """Test state (binary representation) for an Italian card."""
    card_asso_denari = Card(value=1, suit=0, card_system_key='italian_40') # Asso di Denari
    state_ad = card_asso_denari.state
    assert isinstance(state_ad, np.ndarray)
    assert state_ad.shape[0] == 40
    assert state_ad.sum() == 1
    # Index for Asso di Denari (value 1, suit 0): values_map.index(1)=0. 0 + 0 * 10 = 0
    assert state_ad[0] == 1

    card_re_bastoni = Card(value=10, suit=3, card_system_key='italian_40') # Re di Bastoni
    state_rb = card_re_bastoni.state
    # Index for Re di Bastoni (value 10, suit 3): values_map.index(10)=9. 9 + 3 * 10 = 39
    assert state_rb[39] == 1
    
    card_fante_coppe = Card(value=8, suit=1, card_system_key='italian_40') # Fante di Coppe
    state_fc = card_fante_coppe.state
    # Index for Fante di Coppe (value 8, suit 1): values_map.index(8)=7. 7 + 1 * 10 = 17
    assert state_fc[card_fante_coppe.calculate_index()] == 1 
    assert card_fante_coppe.calculate_index() == 17


def test_card_equality_italian():
    """Test equality for Italian cards."""
    card1 = Card(value=7, suit=2, card_system_key='italian_40') # 7 di Spade
    card2 = Card(value=7, suit=2, card_system_key='italian_40') # 7 di Spade
    card3 = Card(value=8, suit=2, card_system_key='italian_40') # Fante di Spade
    assert card1 == card2
    assert card1 != card3

def test_card_equality_mixed_systems():
    """Test equality between cards from different systems."""
    standard_ace = Card(value=1, suit=0, card_system_key='standard_52')
    italian_ace = Card(value=1, suit=0, card_system_key='italian_40')
    assert standard_ace != italian_ace

def test_card_addition_italian():
    """Test adding values of two Italian cards."""
    card1 = Card(value=7, suit=2, card_system_key='italian_40')
    card2 = Card(value=3, suit=1, card_system_key='italian_40')
    assert card1 + card2 == 10

def test_card_comparison_italian():
    """Test comparison (<, >) of Italian cards."""
    asso_denari = Card(value=1, suit=0, card_system_key='italian_40')
    re_denari = Card(value=10, suit=0, card_system_key='italian_40')
    asso_coppe = Card(value=1, suit=1, card_system_key='italian_40')
    assert asso_denari < re_denari
    assert re_denari > asso_denari
    assert asso_denari < asso_coppe # Same value, suit 0 < suit 1

# --- Tests for Invalid Cases and Edge Cases ---

def test_invalid_card_value_standard():
    """Test initialization with invalid value for standard system."""
    with pytest.raises(ValueError, match="Value 14 is not valid for card system standard_52"):
        Card(value=14, suit=0, card_system_key='standard_52')

def test_invalid_card_suit_standard():
    """Test initialization with invalid suit for standard system."""
    with pytest.raises(ValueError, match="Suit index 4 is not valid for card system standard_52"):
        Card(value=1, suit=4, card_system_key='standard_52')

def test_invalid_card_value_italian():
    """Test initialization with invalid value for Italian system."""
    with pytest.raises(ValueError, match="Value 11 is not valid for card system italian_40"):
        Card(value=11, suit=0, card_system_key='italian_40')

def test_invalid_card_suit_italian(): # Though num_suits is same, good to check system context
    """Test initialization with invalid suit for Italian system."""
    with pytest.raises(ValueError, match="Suit index 4 is not valid for card system italian_40"):
        Card(value=1, suit=4, card_system_key='italian_40')

def test_unknown_card_system():
    """Test initialization with an unknown card system key."""
    with pytest.raises(ValueError, match="Unknown card system key: unknown_system"):
        Card(value=1, suit=0, card_system_key='unknown_system')

def test_language_fallback_warning(capsys):
    """Test warning for language not defined in a card system (suits)."""
    # italian_40 does not have 'de' for suits
    card = Card(value=1, suit=0, card_system_key='italian_40', language='de')
    captured = capsys.readouterr()
    assert "Warning: Language 'de' not defined for suits in 'italian_40'. Falling back." in captured.out
    # String representation will use the fallback language (first in system's suit dict, e.g. 'it' or 'en')
    # For italian_40, if 'it' is first: "Asso di Denari"
    # If 'en' is first: "Ace of Coins"
    # This depends on the order in CARD_SYSTEMS definition
    first_lang_suits = list(CARD_SYSTEMS['italian_40']['suits'].keys())[0]
    expected_suit_name = CARD_SYSTEMS['italian_40']['suits'][first_lang_suits][0] # Suit 0
    
    first_lang_special = list(CARD_SYSTEMS['italian_40']['special_values'].keys())[0]
    expected_val_name = CARD_SYSTEMS['italian_40']['special_values'][first_lang_special].get(1, "1")

    conjunction = "di" if first_lang_suits == 'it' else "of" # Simple guess for conjunction
    
    # This part of the assertion might be fragile if CARD_SYSTEMS definition order changes
    # A more robust test would mock CARD_SYSTEMS or check for specific fallback behavior
    assert str(card) == f"{expected_val_name} {conjunction} {expected_suit_name}"


@pytest.mark.parametrize("card_details_1, card_details_2, are_equal", [
    # Standard system
    ({'value': 5, 'suit': 2, 'system': 'standard_52'}, {'value': 5, 'suit': 2, 'system': 'standard_52'}, True),
    ({'value': 5, 'suit': 2, 'system': 'standard_52'}, {'value': 6, 'suit': 2, 'system': 'standard_52'}, False),
    # Italian system
    ({'value': 8, 'suit': 1, 'system': 'italian_40'}, {'value': 8, 'suit': 1, 'system': 'italian_40'}, True),
    ({'value': 8, 'suit': 1, 'system': 'italian_40'}, {'value': 9, 'suit': 1, 'system': 'italian_40'}, False),
    # Mixed systems
    ({'value': 1, 'suit': 0, 'system': 'standard_52'}, {'value': 1, 'suit': 0, 'system': 'italian_40'}, False),
])
def test_card_equality_parametrized(card_details_1, card_details_2, are_equal):
    card1 = Card(value=card_details_1['value'], suit=card_details_1['suit'], card_system_key=card_details_1['system'])
    card2 = Card(value=card_details_2['value'], suit=card_details_2['suit'], card_system_key=card_details_2['system'])
    assert (card1 == card2) is are_equal

@pytest.mark.parametrize("card1_val, card2_val, system, expected_lt", [
    (1, 13, 'standard_52', True), # Ace < King
    (7, 2, 'standard_52', False), # 7 not < 2
    (1, 10, 'italian_40', True),  # Asso < Re
    (8, 4, 'italian_40', False),  # Fante not < 4
])
def test_card_comparison_parametrized(card1_val, card2_val, system, expected_lt):
    card1 = Card(value=card1_val, suit=0, card_system_key=system)
    card2 = Card(value=card2_val, suit=0, card_system_key=system) # Same suit for simplicity
    assert (card1 < card2) is expected_lt
    assert (card2 > card1) is expected_lt # If c1 < c2, then c2 > c1

def test_card_hash():
    """Test hashing of card objects."""
    card1_std = Card(1, 0, 'standard_52')
    card2_std = Card(1, 0, 'standard_52')
    card3_std = Card(2, 0, 'standard_52')
    card1_it = Card(1, 0, 'italian_40')

    s = set()
    s.add(card1_std)
    s.add(card2_std) # Should not add, as it's equal to card1_std
    assert len(s) == 1
    s.add(card3_std)
    assert len(s) == 2
    s.add(card1_it)
    assert len(s) == 3
    assert card1_it in s

```
