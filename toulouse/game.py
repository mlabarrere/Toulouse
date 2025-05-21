import abc
from typing import List, Dict, Any, Tuple, Optional, Union

import gymnasium.spaces as spaces
import numpy as np

from toulouse.cards import Card  # Corrected import
from toulouse.deck import Deck    # Corrected import

class BaseGameEnv(abc.ABC):
    """
    An abstract base class for card game environments, designed with consideration for
    PettingZoo's AEC (Agent-Environment Cycle) model.

    This class provides a common interface and structure for various card games.
    Subclasses must implement the abstract methods and properties defined herein
    to create a functional game environment.

    Key AEC Model Concepts to keep in mind for subclasses:
    - Multiple agents interact with the environment.
    - The game progresses in turns.
    - Each agent receives its own observations and performs actions.
    - The environment handles game termination and rewards for each agent.
    """

    def __init__(self, card_system_key: str, num_players: int):
        """
        Initializes the base game environment.

        Args:
            card_system_key (str): The key for the card system to be used (e.g., "standard_52", "italian_40").
                                   This determines the type of deck and cards used in the game.
            num_players (int): The number of players in the game.
        """
        self.card_system_key: str = card_system_key
        self._num_players: int = num_players # Internal storage for num_players, exposed via property

        # Initialize a list of agent IDs. These are typically strings like "player_0", "player_1", etc.
        # PettingZoo uses these agent IDs to manage turns, observations, and actions.
        self.possible_agents: list[Any] = [f"player_{i}" for i in range(num_players)]

        # Common game elements. Subclasses will manage these in detail.
        self.deck: Optional[Deck] = None
        self.players_hands: Dict[Any, List[Card]] = {agent: [] for agent in self.possible_agents}
        self.table_cards: List[Card] = []
        self.discard_pile: List[Card] = [] # May not be used by all games

        # Tracks the agent whose turn it is to act.
        self.current_agent_id: Any = None

    @abc.abstractmethod
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """
        Resets the environment to its initial state.

        This involves shuffling the deck, dealing cards to players, setting up any
        initial table cards, and selecting the first agent to act.

        Args:
            seed (int | None, optional): A seed for the random number generator, allowing for
                                         reproducible environment resets. Defaults to None.
            options (dict | None, optional): Additional options for resetting the environment.
                                             Defaults to None.

        Returns:
            tuple[Any, dict]: A tuple containing:
                - observation (Any): The initial observation for the first agent to act.
                - info (dict): Auxiliary information about the initial state.
        """
        pass

    @abc.abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """
        Processes an action taken by the current agent and advances the game state.

        This method updates the game based on the action, calculates rewards,
        checks for game termination or truncation, and determines the next agent to act.

        Args:
            action (Any): The action taken by the current agent. The format of the action
                          is defined by the agent's action space.

        Returns:
            tuple[Any, float, bool, bool, dict]: A tuple containing:
                - observation (Any): The observation for the *next* agent.
                - reward (float): The reward received by the *current* agent for the action taken.
                - terminated (bool): True if the game has ended for the *current* agent due to game rules
                                     (e.g., win/loss).
                - truncated (bool): True if the game has ended for the *current* agent due to external reasons
                                    (e.g., time limit), not a natural game conclusion.
                - info (dict): Auxiliary information about the step.
        """
        pass

    @abc.abstractmethod
    def observe(self, agent_id: Any) -> Any:
        """
        Returns the observation for the specified agent.

        The structure of the observation is game-specific and should be defined by
        the `observation_space` for that agent.

        Args:
            agent_id (Any): The ID of the agent for whom to retrieve the observation.

        Returns:
            Any: The observation for the specified agent.
        """
        pass

    @abc.abstractmethod
    def legal_moves(self, agent_id: Any) -> list[Any]:
        """
        Returns a list of legal actions for the specified agent in the current game state.

        This is crucial for implementing action masking, ensuring that agents only
        consider valid actions.

        Args:
            agent_id (Any): The ID of the agent for whom to get legal moves.

        Returns:
            list[Any]: A list of legal actions available to the agent.
        """
        pass

    @abc.abstractmethod
    def render(self) -> Union[None, str, dict]:
        """
        Provides a representation of the game's current state.

        This can be used for displaying the game in a human-readable format (e.g., CLI)
        or for other visualization purposes. The return type can vary based on the
        rendering mode (e.g., None for in-place rendering, str for text, dict for structured data).

        Returns:
            Union[None, str, dict]: The rendered state of the game.
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """
        Performs any necessary cleanup when the environment is no longer needed.

        This might include releasing resources, closing files, or other cleanup tasks.
        """
        pass

    @property
    @abc.abstractmethod
    def agents(self) -> list[Any]:
        """
        Returns a list of active agent IDs in the current game state.
        This list might change if agents can be added or removed during a game.
        Typically, for fixed-player card games, this would be similar to `possible_agents`
        unless an agent has finished playing (e.g., is 'done').
        """
        pass

    @property
    @abc.abstractmethod
    def num_agents(self) -> int:
        """
        Returns the current number of active agents in the game.
        This should correspond to `len(self.agents)`.
        """
        pass

    @property
    @abc.abstractmethod
    def observation_space(self, agent_id: Any) -> spaces.Space:
        """
        Returns the Gymnasium observation space for a given agent.

        This defines the structure and type of observations that an agent can receive.
        It must be an instance of `gymnasium.spaces.Space`.

        Args:
            agent_id (Any): The ID of the agent whose observation space is being queried.

        Returns:
            spaces.Space: The observation space for the specified agent.
        """
        pass

    @property
    @abc.abstractmethod
    def action_space(self, agent_id: Any) -> spaces.Space:
        """
        Returns the Gymnasium action space for a given agent.

        This defines the structure and type of actions that an agent can take.
        It must be an instance of `gymnasium.spaces.Space`.

        Args:
            agent_id (Any): The ID of the agent whose action space is being queried.

        Returns:
            spaces.Space: The action space for the specified agent.
        """
        pass

    # --- Optional Helper Methods (Concrete implementations or to be overridden) ---

    def _initialize_deck(self) -> None:
        """
        Initializes (or re-initializes) the deck for the game.
        This typically involves creating a new Deck instance based on the game's
        `card_system_key` and shuffling it.
        """
        self.deck = Deck(new=True, card_system_key=self.card_system_key, sorted_deck=False)
        # In PettingZoo, shuffling is often done in reset based on a seed.
        # For now, new=True with sorted_deck=False implicitly shuffles.
        # self.deck.update_sort() # Ensure it's shuffled if sorted_deck was True by default in Deck

    def _deal_cards(self, num_cards_per_player: int) -> None:
        """
        Deals a specified number of cards to each player from the deck.

        Args:
            num_cards_per_player (int): The number of cards to deal to each player.

        Raises:
            ValueError: If the deck is not initialized or if there are not enough cards
                        to deal to all players.
        """
        if self.deck is None:
            raise ValueError("Deck has not been initialized. Call _initialize_deck() first.")

        for agent_id in self.possible_agents: # Deals to all possible agents
            if len(self.deck.cards) < num_cards_per_player:
                raise ValueError(f"Not enough cards in the deck to deal {num_cards_per_player} cards to {agent_id}.")
            
            drawn_cards = self.deck.draw(num_cards_per_player)
            self.players_hands[agent_id].extend(drawn_cards)

```
