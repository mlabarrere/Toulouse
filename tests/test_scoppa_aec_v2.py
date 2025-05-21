import pytest
from pettingzoo.test import api_test
from toulouse.games.scoppa import ScoppaEnv

def test_scoppa_aec_api_v2(): # Renamed test function as well
    """Minimal API test for ScoppaEnv - v2 filename."""
    env = ScoppaEnv(language='it') 
    # Using a small number of cycles for faster feedback during debugging
    # verbose_progress=True will give more detailed error messages from api_test
    api_test(env, num_cycles=100, verbose_progress=True)
```
