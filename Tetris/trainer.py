import random
from pyboy import PyBoy
from memory_addresses import *
from skimage.transform import downscale_local_mean
import numpy as np

class Trainer:
    
    def __init__(self, pyboy: PyBoy, net, genome):
        self.pyboy = pyboy
        self.net = net
        self.genome = genome
        self.actions = ['','a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']
        self.rewards = {'seen_coord': 1, 'battle_win': 10, 'battle_lose': -10, 'badge_obtained': 1000}
        self.seen_coords = {}
        self.last_coords_before_battle = None
        self.wait_for_action = 150
        self.steps_without_reward = 2000
        self.current_steps_without_reward = 0
        self.max_recent_actions = 5
        self.recent_actions = [random.randint(0, 8) for _ in range(self.max_recent_actions)]
        self.recent_screens = np.zeros((72, 80, 3), dtype=np.uint8)
        self.all_actions = []
    
    def reward_function(self, action, state, next_state):
        final_reward = 0
        
        if not state['same_coords'] or not next_state['same_coords']:
            final_reward += self.rewards['seen_coord']
            
        # Check if the player has won or lost a battle
        if state['battle_flag'] == 1 and next_state['battle_flag'] == 0:
            # Battle finished, check if the player won or lost
            if self.last_coords_before_battle == next_state['coords_string']:
                final_reward += self.rewards['battle_win']
            else:
                final_reward += self.rewards['battle_lose']
        
        return final_reward
            
    def step(self):
        
        current_state = self.get_state()
        action = self.perform_action(current_state)
            
        next_state = self.get_state()
        reward = self.reward_function(action, current_state, next_state)
        
        if reward != 0:
            self.current_steps_without_reward = 0
        else:
            self.current_steps_without_reward += 1
            
        if self.current_steps_without_reward > self.steps_without_reward:
            return True, reward
        
        return False, reward
    
    def get_state(self):
        
        # Get player position
        same_coords, coords_string = self.check_seen_coords()

        return {"same_coords": same_coords, "coords_string": coords_string, "battle_flag": self.pyboy.memory[BATTLE_FLAG_ADDRESS]}
    
    def perform_action(self, current_state):
        # Perform action and wait for 1200 ticks
        action = self.get_action(current_state)

        if action != '':
            self.pyboy.button(action)
        self.pyboy.tick(self.wait_for_action)
        return action
    
    def get_action(self, current_state):
        
        # Add observation to input
        observation = self.get_observation(current_state)
        
        result = self.net.activate(observation)
        
        action_index = result.index(max(result))
        self.recent_actions.append(action_index)
        self.all_actions.append(self.actions[action_index])

        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions.pop(0)
            
        return self.actions[action_index]
    
    def get_observation(self, current_state):
        x_pos, y_pos, map_n = self.get_game_coords()
        current_badges = self.pyboy.memory[BADGE_COUNT_ADDRESS]
        screen = self.render()
        self.update_recent_screens(screen)
        
        # Flatten and normalize the screen data
        flattened_screens = self.recent_screens.flatten() / 255.0
        
        # Return scalar values and flattened screen data
        return [x_pos, y_pos, map_n, current_state['battle_flag']] + self.recent_actions + [current_badges] + flattened_screens.tolist()
    
    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:,:, 0]
    
    # Get the current player coordinates
    def get_game_coords(self):
        return (self.pyboy.memory[X_POS_ADDRESS], self.pyboy.memory[Y_POS_ADDRESS], self.pyboy.memory[MAP_N_ADDRESS])
    
    # Reward for checking a new coordinate
    def check_seen_coords(self):
        if self.pyboy.memory[BATTLE_FLAG_ADDRESS] == 0:
            coord_string = self.get_coords_string()
            self.last_coords_before_battle = coord_string
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
                return True, coord_string
            else:
                self.seen_coords[coord_string] = 1
                return False, coord_string  
        return True, None
    
    def get_coords_string(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return f"x:{x_pos} y:{y_pos} m:{map_n}"
    
    def manual_inputs(self):
        command = input("Enter command (one of: a, b, left, right, up, down, start, select): ")
        if command and command in self.actions:
            self.pyboy.button(command)
            print(f"Button pressed: {command}")
        else:
            print("Invalid input")
            
    def save_actions(self, final_reward):
        if final_reward > 1:
            with open(f"checkpoints/{self.genome.key}_actions.txt", "w") as f:
                for action in self.all_actions:
                    f.write(action + "\n")
    
    def render(self, reduce_res=True):
        game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
