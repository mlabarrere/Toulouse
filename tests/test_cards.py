import pytest
from toulouse.cards import Card, CARD_SYSTEMS
import numpy as np

# --- Tests for "italian_40" Card System (Default) ---

def test_card_initialization():
    """Test Card initialization (system is "italian_40" by default)."""
    card = Card(value=1, suit=0, language='it') # Asso di Denari
    assert card.value == 1
    assert card.suit == 0
    assert card.card_system_key == 'italian_40' # Verify it's correctly hardcoded
    assert card.language == 'it'
    assert card.card_system["deck_size"] == 40

def test_card_string_representation():
    """Test string representation for the card system."""
    asso_denari = Card(value=1, suit=0, language='it')
    assert str(asso_denari) == "Asso di Denari"
    
    fante_coppe = Card(value=8, suit=1, language='it')
    assert str(fante_coppe) == "Fante di Coppe"

    re_bastoni = Card(value=10, suit=3, language='it')
    assert str(re_bastoni) == "Re di Bastoni"

    # Test with English language for Italian cards
    asso_denari_en = Card(value=1, suit=0, language='en')
    assert str(asso_denari_en) == "Ace of Coins" # Based on CARD_SYSTEMS definition

def test_card_state():
    """Test state (binary representation) for an Italian card."""
    card_asso_denari = Card(value=1, suit=0) # Default lang 'en', system "italian_40"
    state_ad = card_asso_denari.state
    assert isinstance(state_ad, np.ndarray)
    assert state_ad.shape[0] == 40
    assert state_ad.sum() == 1
    assert state_ad[0] == 1 # Index for Asso di Denari

    card_re_bastoni = Card(value=10, suit=3) 
    state_rb = card_re_bastoni.state
    assert state_rb[39] == 1 # Index for Re di Bastoni
    
    card_fante_coppe = Card(value=8, suit=1) 
    state_fc = card_fante_coppe.state
    assert state_fc[card_fante_coppe.calculate_index()] == 1 
    assert card_fante_coppe.calculate_index() == 17


def test_card_equality():
    """Test equality for Italian cards."""
    card1 = Card(value=7, suit=2) # 7 di Spade
    card2 = Card(value=7, suit=2) # 7 di Spade
    card3 = Card(value=8, suit=2) # Fante di Spade
    assert card1 == card2
    assert card1 != card3

# test_card_equality_mixed_systems was removed as there's only one system now.

def test_card_addition():
    """Test adding values of two Italian cards."""
    card1 = Card(value=7, suit=2)
    card2 = Card(value=3, suit=1)
    assert card1 + card2 == 10

def test_card_comparison():
    """Test comparison (<, >) of Italian cards."""
    asso_denari = Card(value=1, suit=0)
    re_denari = Card(value=10, suit=0)
    asso_coppe = Card(value=1, suit=1)
    assert asso_denari < re_denari
    assert re_denari > asso_denari
    assert asso_denari < asso_coppe # Same value, suit 0 < suit 1

# --- Tests for Invalid Cases and Edge Cases ---

def test_invalid_card_value():
    """Test initialization with invalid value for Italian system."""
    with pytest.raises(ValueError, match="Value 11 is not valid for card system italian_40"):
        Card(value=11, suit=0)

def test_invalid_card_suit(): 
    """Test initialization with invalid suit for Italian system."""
    with pytest.raises(ValueError, match="Suit index 4 is not valid for card system italian_40"):
        Card(value=1, suit=4)

# test_unknown_card_system is removed as card_system_key is hardcoded in Card.__init__

def test_language_fallback_warning(capsys):
    """Test warning for language not defined in a card system (suits)."""
    # italian_40 does not have 'de' for suits in its definition in CARD_SYSTEMS
    card = Card(value=1, suit=0, language='de') # 'de' is for conjunction, fallback for names
    captured = capsys.readouterr()
    assert "Warning: Language 'de' not defined for suits in 'italian_40'. Falling back." in captured.out
    # String representation will use the fallback language for names (e.g., 'it' or 'en' from CARD_SYSTEMS)
    # and the specified language ('de') for the conjunction if available in LANGUAGE_CONJUNCTIONS.
    # For italian_40, if 'it' is the first language defined for suits in CARD_SYSTEMS: "Asso von Denari"
    # If 'en' is first: "Ace von Coins"
    # This depends on the order in CARD_SYSTEMS and conjunctions.
    # Let's determine the expected fallback name and conjunction more robustly.
    from toulouse.cards import LANGUAGE_CONJUNCTIONS as global_language_conjunctions # Import it
    fallback_lang_suits = list(CARD_SYSTEMS['italian_40']['suits'].keys())[0]
    expected_suit_name = CARD_SYSTEMS['italian_40']['suits'][fallback_lang_suits][0] # Suit 0
    
    fallback_lang_special = list(CARD_SYSTEMS['italian_40']['special_values'].keys())[0]
    expected_val_name = CARD_SYSTEMS['italian_40']['special_values'][fallback_lang_special].get(1, "1")

    # The conjunction should come from the card's actual language ('de') if defined in LANGUAGE_CONJUNCTIONS
    if 'de' not in global_language_conjunctions: 
        conjunction = "of" # Default if 'de' is not specifically handled for conjunctions
    else:
        conjunction = global_language_conjunctions['de']


    assert str(card) == f"{expected_val_name} {conjunction} {expected_suit_name}"


@pytest.mark.parametrize("card_details_1, card_details_2, are_equal", [
    # Italian system cases (system parameter is now implicit)
    ({'value': 8, 'suit': 1}, {'value': 8, 'suit': 1}, True),
    ({'value': 8, 'suit': 1}, {'value': 9, 'suit': 1}, False),
    ({'value': 1, 'suit': 0, 'lang': 'it'}, {'value': 1, 'suit': 0, 'lang': 'en'}, True), # Diff lang, same card
])
def test_card_equality_parametrized(card_details_1, card_details_2, are_equal):
    lang1 = card_details_1.get('lang', 'it')
    lang2 = card_details_2.get('lang', 'it')
    # Card() no longer takes card_system_key
    card1 = Card(value=card_details_1['value'], suit=card_details_1['suit'], language=lang1)
    card2 = Card(value=card_details_2['value'], suit=card_details_2['suit'], language=lang2)
    assert (card1 == card2) is are_equal

@pytest.mark.parametrize("card1_val, card2_val, expected_lt", [
    (1, 10, True),  # Asso < Re
    (8, 4, False),  # Fante not < 4
])
def test_card_comparison_parametrized(card1_val, card2_val, expected_lt):
    # Card() no longer takes card_system_key
    card1 = Card(value=card1_val, suit=0) # Same suit for simplicity
    card2 = Card(value=card2_val, suit=0)
    assert (card1 < card2) is expected_lt
    assert (card2 > card1) is expected_lt # If c1 < c2, then c2 > c1

def test_card_hash():
    """Test hashing of card objects (system is implicitly "italian_40")."""
    # Card() no longer takes card_system_key
    card1_it = Card(1, 0, language='it') # Asso di Denari
    card2_it = Card(1, 0, language='en') # Ace of Coins (same card by value/suit/system, diff language)
    card3_it = Card(8, 1) # Fante di Coppe (default lang 'en')

    s = set()
    s.add(card1_it)
    s.add(card2_it) # Should not add, as it's equal to card1_it by game logic (value, suit, system)
    assert len(s) == 1
    assert card1_it in s 
    assert card2_it in s # Equality means it's considered the same for set purposes

    s.add(card3_it) # Add a different card
    assert len(s) == 2
    assert card3_it in s
