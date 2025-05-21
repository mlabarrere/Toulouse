import pytest
from pettingzoo.test import api_test
from toulouse.games.scoppa import ScoppaEnv

def test_scoppa_aec_api():
    """Minimal API test."""
    env = ScoppaEnv(language='it') # Use a default if constructor requires it
    # Using a small number of cycles for faster feedback during debugging
    # verbose_progress=True will give more detailed error messages from api_test
    api_test(env, num_cycles=100, verbose_progress=True)
```
