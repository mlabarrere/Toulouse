import abc
from typing import List, Dict, Any, Tuple, Optional, Union, cast
import itertools # For capture combinations

import gymnasium.spaces as spaces
import numpy as np
from pettingzoo.utils.agent_selector import agent_selector 
from pettingzoo.utils.env import AECEnv # Import AECEnv

from toulouse.cards import Card, CARD_SYSTEMS
from toulouse.deck import Deck

class ScoppaEnv(AECEnv): 
    """
    A PettingZoo AEC compliant implementation of the Italian card game Scoppa for two players.
    Observation space uses multi-binary representation for hand and table cards.
    """

    metadata = {
        "render_modes": ["text", "human"], 
        "name": "Scoppa-v0",
        "is_parallelizable": False, 
        "render_fps": 2, 
    }

    MAX_HAND_SIZE = 3
    INITIAL_TABLE_CARDS = 4
    # MAX_TABLE_CARDS_OBS is not strictly needed for multi-binary, as shape is fixed by deck size
    # NO_CARD_INDEX is also not needed for multi-binary representation of hand/table
    DECK_SIZE = CARD_SYSTEMS["italian_40"]["deck_size"] # Should be 40

    def __init__(self, language: str = 'it'):
        super().__init__() 
        self.language: str = language
        
        self.possible_agents: list[str] = [f"player_{i}" for i in range(2)]

        self._observation_spaces = {
            agent: self._def_observation_space() for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: self._def_action_space() for agent in self.possible_agents
        }
        
        self.agents: list[str] = []
        self._agent_selector: agent_selector = agent_selector(self.possible_agents)
        self.agent_selection: str = "" 
        
        self.rewards: Dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards: Dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        self.terminations: Dict[str, bool] = {agent: False for agent in self.possible_agents}
        self.truncations: Dict[str, bool] = {agent: False for agent in self.possible_agents}
        self.infos: Dict[str, Dict[str, Any]] = {agent: {} for agent in self.possible_agents}
        
        self.capture_pile: Dict[str, List[Card]] = {agent: [] for agent in self.possible_agents}
        self.last_capturer: Optional[str] = None
        self.round_over: bool = False 
        self.game_over: bool = False 
        self.scopa_counts: Dict[str, int] = {agent: 0 for agent in self.possible_agents}

        self.deck: Optional[Deck] = None
        self.players_hands: Dict[str, List[Card]] = {agent: [] for agent in self.possible_agents}
        self.table_cards: List[Card] = []
        
        self._was_done_step: bool = False


    def _def_observation_space(self) -> spaces.Space:
        """Defines the observation space for an agent using multi-binary for cards."""
        # Multi-binary representation for hand and table cards
        # Shape is (DECK_SIZE,) where each index corresponds to a card. 1 if present, 0 otherwise.
        cards_multi_binary_space = spaces.Box(low=0, high=1, shape=(self.DECK_SIZE,), dtype=np.int8)

        return spaces.Dict({
            "hand": cards_multi_binary_space,
            "table_cards": cards_multi_binary_space,
            "player_captures": spaces.Discrete(self.DECK_SIZE + 1), # Num cards captured
            "opponent_captures": spaces.Discrete(self.DECK_SIZE + 1),
            "last_capturer": spaces.Discrete(len(self.possible_agents) + 1), # agent_idx or num_agents for None
            "deck_size": spaces.Discrete(self.DECK_SIZE + 1), # Num cards in deck
            "action_mask": spaces.Box(low=0, high=1, shape=(self.MAX_HAND_SIZE,), dtype=np.int8),
        })

    def _def_action_space(self) -> spaces.Space:
        return spaces.Discrete(self.MAX_HAND_SIZE)

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def _initialize_deck(self) -> None:
        self.deck = Deck(new=True, card_system_key="italian_40", sorted_deck=False)

    def _deal_cards(self, num_cards_per_player: int) -> None:
        if self.deck is None: raise ValueError("Deck not initialized.")
        agents_to_deal = self.agents if self.agents else self.possible_agents
        for agent_id in agents_to_deal: 
            if len(self.deck.cards) < num_cards_per_player: break 
            self.players_hands[agent_id].extend(self.deck.draw(num_cards_per_player))
            
    def _clear_rewards(self):
        for agent in self.possible_agents: self.rewards[agent] = 0.0

    def _accumulate_rewards(self):
        agent_to_accumulate_for = self.agent_selection
        if agent_to_accumulate_for in self.rewards and agent_to_accumulate_for in self._cumulative_rewards:
            self._cumulative_rewards[agent_to_accumulate_for] += self.rewards[agent_to_accumulate_for]

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        if seed is not None:
            # For full reproducibility, Deck shuffling should be seeded.
            # np.random.seed(seed) # If using a global RNG for other game aspects
            pass

        self.agents = list(self.possible_agents)
        self._agent_selector.reinit(self.agents) 
        
        for agent_id in self.possible_agents:
            self.rewards[agent_id] = 0.0
            self._cumulative_rewards[agent_id] = 0.0
            self.terminations[agent_id] = False
            self.truncations[agent_id] = False
            self.infos[agent_id] = {}
            self.players_hands[agent_id] = []
            self.capture_pile[agent_id] = []
            self.scopa_counts[agent_id] = 0

        self._initialize_deck()
        if self.deck is None: raise RuntimeError("Deck init failed.")

        self.table_cards = []
        self.last_capturer = None
        self.round_over = False
        self.game_over = False 

        self._deal_cards(num_cards_per_player=self.MAX_HAND_SIZE)

        if len(self.deck.cards) < self.INITIAL_TABLE_CARDS:
            raise RuntimeError("Not enough cards for initial table deal.")
        self.table_cards.extend(self.deck.draw(self.INITIAL_TABLE_CARDS))
        
        self.agent_selection = self._agent_selector.reset() 
        self._was_done_step = False


    def step(self, action: int) -> None:
        agent_acted = self.agent_selection 
        self._accumulate_rewards() 

        if self.terminations[agent_acted] or self.truncations[agent_acted]:
            self._was_done_step = True 
            self._clear_rewards() 
            if self.agents: self.agent_selection = self._agent_selector.next()
            return

        self._was_done_step = False 
        self._clear_rewards() 

        agent_hand = self.players_hands[agent_acted]
        if not (0 <= action < len(agent_hand)):
            self.infos[agent_acted]['error'] = f"Invalid action {action} for hand {agent_hand}."
            self.rewards[agent_acted] = -1.0 
        else:
            played_card = agent_hand.pop(action)
            scopa_achieved = False
            immediate_reward = 0.0 

            cards_to_capture = self._find_best_capture_combination(played_card, self.table_cards)
            if cards_to_capture:
                self.capture_pile[agent_acted].append(played_card)
                for card_obj in cards_to_capture: self.capture_pile[agent_acted].append(card_obj); self.table_cards.remove(card_obj)
                self.last_capturer = agent_acted
                if not self.table_cards: 
                    scopa_achieved = True; self.scopa_counts[agent_acted] += 1; immediate_reward = 1.0 
            else:
                self.table_cards.append(played_card)
            
            self.rewards[agent_acted] = immediate_reward 
            self.infos[agent_acted].update({"scopa_achieved": scopa_achieved, "played_card": str(played_card)})
        
        all_hands_empty = all(not self.players_hands[agent] for agent in self.possible_agents)
        if self.deck and all_hands_empty and len(self.deck.cards) > 0:
            if len(self.deck.cards) >= len(self.possible_agents) * self.MAX_HAND_SIZE:
                self._deal_cards(num_cards_per_player=self.MAX_HAND_SIZE)

        game_just_ended = False
        if self.deck and len(self.deck.cards) == 0 and all(not self.players_hands[agent] for agent in self.possible_agents):
            self.round_over = True; self.game_over = True; game_just_ended = True
            if self.last_capturer and self.table_cards:
                self.capture_pile[self.last_capturer].extend(self.table_cards); self.table_cards.clear()
            
            final_scores = self._calculate_score() 
            for ag_id in self.possible_agents:
                self.terminations[ag_id] = True
                self.rewards[ag_id] += final_scores.get(ag_id, 0) 
                self.infos[ag_id]["final_score"] = final_scores.get(ag_id, 0)
        
        if game_just_ended: self.terminations[agent_acted] = True 

        if game_just_ended: self.agents = [] 
        elif self._agent_selector.is_last():
             self._agent_selector.reinit(self.agents) 
             self.agent_selection = self._agent_selector.next()
        else:
             self.agent_selection = self._agent_selector.next()


    def last(self, observe: bool = True) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        agent_for_last = self.agent_selection 
        observation = self.observe(agent_for_last) if observe else None
        reward = self._cumulative_rewards.get(agent_for_last, 0.0)
        termination = self.terminations.get(agent_for_last, False)
        truncation = self.truncations.get(agent_for_last, False)
        info = self.infos.get(agent_for_last, {})
        return observation, reward, termination, truncation, info

    def _cards_to_multi_binary(self, card_list: List[Card]) -> np.ndarray:
        """Converts a list of cards to a multi-binary vector of length DECK_SIZE."""
        vec = np.zeros(self.DECK_SIZE, dtype=np.int8)
        for card in card_list:
            vec[card.calculate_index()] = 1
        return vec

    def state(self) -> np.ndarray:
        """Returns a global state representation of the environment as a NumPy array."""
        p0_id, p1_id = self.possible_agents[0], self.possible_agents[1]
        
        p0_hand_vec = self._cards_to_multi_binary(self.players_hands.get(p0_id, []))
        p1_hand_vec = self._cards_to_multi_binary(self.players_hands.get(p1_id, []))
        table_vec = self._cards_to_multi_binary(self.table_cards)

        p0_capt_count = len(self.capture_pile.get(p0_id, []))
        p1_capt_count = len(self.capture_pile.get(p1_id, []))
        
        p0_scopa_count = self.scopa_counts.get(p0_id,0)
        p1_scopa_count = self.scopa_counts.get(p1_id,0)

        last_capt_idx = self.possible_agents.index(self.last_capturer) if self.last_capturer else len(self.possible_agents)
        deck_rem_count = len(self.deck.cards) if self.deck else 0
        
        current_player_turn_idx = -1 
        try:
            if self.agent_selection and self.agents: 
                 current_player_turn_idx = self.possible_agents.index(self.agent_selection)
        except ValueError: current_player_turn_idx = -1

        # Concatenate all parts into a single NumPy array
        # Order: P0_hand (40), P1_hand (40), table (40), scalars (7) = 127
        state_parts = np.concatenate([
            p0_hand_vec, p1_hand_vec, table_vec,
            np.array([p0_capt_count, p1_capt_count, p0_scopa_count, p1_scopa_count,
                      last_capt_idx, deck_rem_count, current_player_turn_idx], dtype=np.int32)
        ])
        return state_parts


    def observe(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self.possible_agents: raise ValueError(f"Unknown agent ID: {agent_id}.")

        hand_card_objects = self.players_hands.get(agent_id, [])
        hand_multibinary = self._cards_to_multi_binary(hand_card_objects)
        table_multibinary = self._cards_to_multi_binary(self.table_cards)

        last_capturer_idx = self.possible_agents.index(self.last_capturer) if self.last_capturer else len(self.possible_agents)
        
        action_mask = np.zeros(self.MAX_HAND_SIZE, dtype=np.int8)
        if not (self.terminations.get(agent_id, False) or self.truncations.get(agent_id, False)):
            num_cards_in_hand = len(hand_card_objects)
            action_mask[:num_cards_in_hand] = 1
        
        opponent_id_idx = (self.possible_agents.index(agent_id) + 1) % len(self.possible_agents)
        opponent_id = self.possible_agents[opponent_id_idx]

        return {
            "hand": hand_multibinary, 
            "table_cards": table_multibinary,
            "player_captures": len(self.capture_pile.get(agent_id, [])),
            "opponent_captures": len(self.capture_pile.get(opponent_id, [])),
            "last_capturer": last_capturer_idx, 
            "deck_size": len(self.deck.cards) if self.deck else 0,
            "action_mask": action_mask,
        }

    def agent_iter(self, max_iter: int = int(1e9)) -> Any:
        return iter(self._agent_selector)

    def render(self) -> None: 
        # Render can remain mostly the same, as it's for human readability
        # and can still access self.players_hands etc.
        print("\n--- Scoppa Game State (PettingZoo AEC - MultiBinary Obs) ---")
        active_player_display = self.agent_selection
        
        if not self.agents: active_player_display = "N/A (Game Over)"
        elif self.terminations.get(self.agent_selection) or self.truncations.get(self.agent_selection):
             active_player_display = f"{self.agent_selection} (Done)"

        print(f"Agent for next action: {active_player_display}")
        for ag_id in self.possible_agents:
            hand_str = ", ".join([str(c) for c in self.players_hands.get(ag_id, [])])
            capt_cnt = len(self.capture_pile.get(ag_id, []))
            scopa_cnt = self.scopa_counts.get(ag_id,0)
            term = self.terminations.get(ag_id, "N/A")
            trunc = self.truncations.get(ag_id, "N/A")
            cum_reward = self._cumulative_rewards.get(ag_id, "N/A") 
            info_dict = self.infos.get(ag_id, {})
            print(f"  Player {ag_id}: Hand=[{hand_str}], Capt={capt_cnt}, Scopas={scopa_cnt}")
            print(f"    Term={term}, Trunc={trunc}, CumReward={cum_reward}, Info={info_dict}")
        tbl_str = ", ".join([str(c) for c in self.table_cards])
        print(f"  Table ({len(self.table_cards)} cards): [{tbl_str}]")
        if self.deck: print(f"  Deck: {len(self.deck.cards)} cards remaining.")
        if self.last_capturer: print(f"  Last Capturer: {self.last_capturer}")
        if self.game_over : print("--- GAME OVER ---")
        print("------------------------------------------------------------\n")

    def close(self) -> None:
        self.deck = None; self.players_hands.clear(); self.table_cards.clear(); self.capture_pile.clear()
        self.agents = []; self.agent_selection = ""
        if hasattr(self, '_agent_selector') and self._agent_selector is not None: self._agent_selector.reinit([])
        for k_agent in self.possible_agents: 
            self.rewards[k_agent]=0.0; self._cumulative_rewards[k_agent]=0.0; 
            self.terminations[k_agent]=False; self.truncations[k_agent]=False; self.infos[k_agent]={}
        print("ScoppaEnv (PettingZoo AEC - MultiBinary Obs) closed.")

    def _find_best_capture_combination(self, played_card: Card, current_table_cards: list[Card]) -> Optional[List[Card]]:
        for table_card in current_table_cards:
            if table_card.value == played_card.value: return [table_card]
        if len(current_table_cards) >= 2:
            for card_combo in itertools.combinations(current_table_cards, 2):
                if sum(c.value for c in card_combo) == played_card.value: return list(card_combo)
        return None

    def _calculate_score(self) -> Dict[str, int]:
        scores = {}
        for agent_id_calc in self.possible_agents:
            card_score = len(self.capture_pile.get(agent_id_calc, []))
            scopa_score = self.scopa_counts.get(agent_id_calc, 0) 
            scores[agent_id_calc] = card_score + scopa_score 
        return scores

```
