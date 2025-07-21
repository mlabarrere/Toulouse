"""
Internationalization and Localization for Toulouse.

This module provides a centralized system for managing translations of card suits,
values, and other UI strings. It is designed to be easily extensible for new
languages and card systems.

Structure:
- TRANSLATIONS: A nested dictionary storing all language-specific data.
  - Key 1: Language code (e.g., "en", "fr", "it", "es").
  - Key 2: String category (e.g., "suits", "values", "connectors").
  - Key 3: Card system key (e.g., "italian_40", "spanish_40").

- get_translation(): A function to retrieve the appropriate translation data
  based on language and card system.
"""

from typing import Dict, Any

# Centralized dictionary for all translations
TRANSLATIONS: Dict[str, Dict[str, Any]] = {
    "en": {
        "suits": {
            "italian_40": ["Coins", "Cups", "Swords", "Clubs"],
            "spanish_40": ["Golds", "Cups", "Swords", "Clubs"],
        },
        "values": {
            "italian_40": {1: "Ace", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Jack", 9: "Knight", 10: "King"},
            "spanish_40": {1: "Ace", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Sota", 9: "Caballo", 10: "Rey"},
        },
        "connectors": {"of": "of"},
    },
    "fr": {
        "suits": {
            "italian_40": ["Deniers", "Coupes", "Épées", "Bâtons"],
            "spanish_40": ["Ors", "Coupes", "Épées", "Bâtons"],
        },
        "values": {
            "italian_40": {1: "As", 2: "Deux", 3: "Trois", 4: "Quatre", 5: "Cinq", 6: "Six", 7: "Sept", 8: "Valet", 9: "Cavalier", 10: "Roi"},
            "spanish_40": {1: "As", 2: "Deux", 3: "Trois", 4: "Quatre", 5: "Cinq", 6: "Six", 7: "Sept", 8: "Sota", 9: "Caballo", 10: "Rey"},
        },
        "connectors": {"of": "de"},
    },
    "it": {
        "suits": {
            "italian_40": ["Denari", "Coppe", "Spade", "Bastoni"],
            "spanish_40": ["Oros", "Coppe", "Spade", "Bastoni"],
        },
        "values": {
            "italian_40": {1: "Asso", 2: "Due", 3: "Tre", 4: "Quattro", 5: "Cinque", 6: "Sei", 7: "Sette", 8: "Fante", 9: "Cavallo", 10: "Re"},
            "spanish_40": {1: "As", 2: "Due", 3: "Tre", 4: "Quattro", 5: "Cinque", 6: "Sei", 7: "Sette", 8: "Sota", 9: "Caballo", 10: "Re"},
        },
        "connectors": {"of": "di"},
    },
    "es": {
        "suits": {
            "italian_40": ["Oros", "Copas", "Espadas", "Bastos"],
            "spanish_40": ["Oros", "Copas", "Espadas", "Bastos"],
        },
        "values": {
            "italian_40": {1: "As", 2: "Dos", 3: "Tres", 4: "Cuatro", 5: "Cinco", 6: "Seis", 7: "Siete", 8: "Sota", 9: "Caballo", 10: "Rey"},
            "spanish_40": {1: "As", 2: "Dos", 3: "Tres", 4: "Cuatro", 5: "Cinco", 6: "Seis", 7: "Siete", 8: "Sota", 9: "Caballo", 10: "Rey"},
        },
        "connectors": {"of": "de"},
    },
}


def get_translation(language: str, card_system: str) -> Dict[str, Any]:
    """
    Retrieves translation data for a given language and card system.

    Args:
        language: The language code (e.g., "en", "fr").
        card_system: The card system key (e.g., "italian_40").

    Returns:
        A dictionary containing "suits", "values", and "connector" strings.
        Falls back to English if the requested language is not found.
    """
    lang_data = TRANSLATIONS.get(language, TRANSLATIONS["en"])
    
    return {
        "suits": lang_data["suits"].get(card_system, {}),
        "values": lang_data["values"].get(card_system, {}),
        "connector": lang_data["connectors"].get("of", "of"),
    }
