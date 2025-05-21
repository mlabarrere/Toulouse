from toulouse.games.scoppa import ScoppaEnv
from toulouse.cards import Card # Needed for type hinting if we directly access hands
import numpy as np # For action_mask type

def display_hand_for_cli(player_id: str, hand: list[Card], action_mask: np.ndarray):
    """Displays the player's hand with indices for playable cards for the CLI."""
    print(f"\n--- Player {player_id}'s Turn ---")
    print("Your hand:")
    playable_cards_displayed = 0
    for i, card in enumerate(hand): # Iterate through actual cards in hand
        if action_mask[i] == 1:
            print(f"  Index {i}: {card}")
            playable_cards_displayed +=1
        # If action_mask[i] is 0, it means that slot in the hand is not playable
        # (e.g. hand has < MAX_HAND_SIZE cards, so higher indices are masked out)
    
    if playable_cards_displayed == 0 and hand: # Hand has cards, but mask says none are playable (should not happen in Scoppa)
        print("  (No playable cards in hand according to action mask - this may be an issue!)")
    elif not hand: # Hand is actually empty
        print("  (Hand is empty)")


def main_game_loop():
    """Main loop for playing Scoppa via CLI."""
    print("Welcome to Scoppa on the CLI!")
    env = ScoppaEnv(language='it') # Default language for Scoppa
    
    # Initial reset to get the first agent and observation
    # For PettingZoo-like environments, reset() often doesn't return the first agent_id directly.
    # We typically start with env.agent_selection or the first agent in env.possible_agents.
    # ScoppaEnv's reset sets self.current_agent_id to possible_agents[0].
    observation, info = env.reset()
    
    game_on = True
    while game_on:
        current_agent_id = env.current_agent_id 
        
        if current_agent_id is None: # Game might have ended after the last step
            # This check is important if the loop condition `game_on` isn't solely based on terminated/truncated
            # from the *previous* step, or if there's a state where no agent is current.
            # In our loop, `game_on = False` sets this, so this might be redundant if loop exits promptly.
            print("Game has ended (no current agent).")
            break

        # Display general game state using the environment's render method
        print("\n----------------------------------------")
        env.render() # Shows table, capture piles, deck size etc.
        print("----------------------------------------")

        # The observation is for the current_agent_id
        # It was either from env.reset() or the last env.step()
        
        current_agent_hand_cards = env.players_hands[current_agent_id] # List of Card objects
        action_mask = observation.get("action_mask")

        if action_mask is None:
            print(f"Error: Action mask not found in observation for {current_agent_id}. Exiting.")
            break
        
        if not any(action_mask):
            # This means the current player has no valid moves.
            # In Scoppa, this usually means their hand is empty and cards need to be re-dealt,
            # or the game/round is over. The step function should handle this state transition.
            # If step brings us here with no moves, it's likely an issue or game end.
            print(f"Player {current_agent_id} has no legal moves (empty hand or game state issue).")
            # The game's step function should ideally handle transitions that lead to empty hands
            # (e.g., by re-dealing or ending the game). If current_agent_id is still set
            # but they have no moves, it implies the game loop should not have continued to action selection for this agent.
            # For now, we assume if this happens, it's effectively game over or an error state.
            # This check might be more robustly handled by checking `terminated` from previous step.
            game_on = False # Should have been caught by terminated flag from previous step
            break

        display_hand_for_cli(current_agent_id, current_agent_hand_cards, action_mask)

        chosen_action_index = -1
        while True:
            try:
                action_input = input(f"Choose a card index to play (0 to {len(current_agent_hand_cards) - 1}): ")
                chosen_action_index = int(action_input)
                
                # Validate against hand size and action_mask
                if 0 <= chosen_action_index < len(current_agent_hand_cards): # Check if index is for an actual card
                    if action_mask[chosen_action_index] == 1:
                        break # Valid action
                    else:
                        # This specific index is masked out, though it's within overall MAX_HAND_SIZE
                        print(f"Invalid move: Card at index {chosen_action_index} is not playable according to the action mask (mask value: {action_mask[chosen_action_index]}).")
                        print(f"Action mask: {action_mask}. Playable indices for current hand: {[i for i,m in enumerate(action_mask) if m==1 and i < len(current_agent_hand_cards)]}")
                else: # Index out of bounds for current hand size
                     print(f"Invalid index. Please choose an index from 0 to {len(current_agent_hand_cards) - 1} for cards currently in your hand.")

            except ValueError:
                print("Invalid input. Please enter a number.")
            except IndexError: # Should be caught by "0 <= chosen_action_index < len(current_agent_hand_cards)"
                 print(f"Invalid index. Please choose from available cards in your hand.")


        # Execute the chosen action (which is an index for the hand)
        observation, reward, terminated, truncated, info = env.step(chosen_action_index)

        print(f"\nPlayer {current_agent_id} played. Reward for this step: {reward}")
        if info.get("scopa_achieved"):
            print("***** SCOPA! Player {current_agent_id} cleared the table! *****")

        if terminated or truncated:
            print("\n!!!!!!!!!!!!!!!! GAME OVER !!!!!!!!!!!!!!!!")
            # Render final state one last time
            print("Final Game State:")
            env.render() 
            
            final_scores = info.get("final_scores")
            if final_scores:
                print("\n--- Final Scores ---")
                for p_id, score in final_scores.items():
                    print(f"  Player {p_id}: {score} points")
            else:
                # Fallback if final_scores not in info (it should be as per ScoppaEnv.step)
                print("No final scores reported in info dict. Calculating from env...")
                calculated_scores = env._calculate_score() # Accessing protected for CLI convenience
                print("\n--- Final Scores (Calculated) ---")
                for p_id, score in calculated_scores.items():
                    print(f"  Player {p_id}: {score} points")
            
            game_on = False # End the loop
        else:
            # If not terminated, the observation is for the *next* agent.
            # The loop will correctly use env.current_agent_id which was updated in env.step().
            pass 
            # `observation` variable is already updated for the next agent for the top of the loop.

    env.close()
    print("\nThanks for playing Scoppa! Goodbye.")

if __name__ == "__main__":
    main_game_loop()
```
