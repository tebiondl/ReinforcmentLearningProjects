import random
from pyboy import PyBoy
from memory_addresses import *
from skimage.transform import downscale_local_mean
import numpy as np

class Player:
    
    def __init__(self, pyboy: PyBoy, net, genome):
        self.pyboy = pyboy
        self.net = net
        self.genome = genome
        self.actions = ['','a', 'b', 'left', 'right', 'up', 'down']
        self.rewards = {'score': 100, 'level': 1000, 'new_piece': 1, 'game_over': -10}
        self.wait_for_action = 1
        self.max_recent_actions = 5
        self.recent_actions = [random.randint(0, 8) for _ in range(self.max_recent_actions)]
        self.recent_screens = np.zeros((36, 40, 3), dtype=np.uint8)
        self.all_actions = []
        self.last_score = 0
        self.last_level = 0
    
    def reward_function(self, state):
        final_reward = 0
        
        # Reward for score
        if state["score"] > self.last_score:
            print("Score increased")
            final_reward =+ self.rewards["score"]
            self.last_score = state["score"]
        
        # Reward for level
        if state["level"] > self.last_level:
            print("Level increased")
            final_reward =+ self.rewards["level"]
            self.last_level = state["level"]
        
        # Reward for new piece
        if state["piece_change"] == 128:
            final_reward =+ self.rewards["new_piece"]
        
        return final_reward
            
    def step(self):
        
        self.perform_action()
        state = self.get_state()
        reward = self.reward_function(state)
        
        # Check if game is over
        if self.pyboy.memory[GAME_STATE] != 0:
            reward += self.rewards["game_over"]
            return True, reward
        
        return False, reward
    
    def get_state(self):
        return {"score": self.pyboy.memory[SCORE], "level": self.pyboy.memory[LEVEL], "piece_change": self.pyboy.memory[PIECE_CHANGE]}
    
    def perform_action(self):
        action = self.get_action()

        if action != '':
            self.pyboy.button(action)
        self.pyboy.tick(self.wait_for_action)
    
    def get_action(self):
        
        # Add observation to input
        observation = self.get_observation()
        
        result = self.net.activate(observation)
        
        action_index = result.index(max(result))
        self.recent_actions.append(action_index)
        self.all_actions.append(self.actions[action_index])

        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions.pop(0)
            
        return self.actions[action_index]
    
    def get_observation(self):
        
        screen = self.render()
        self.update_recent_screens(screen)
        
        # Flatten and normalize the screen data
        flattened_screens = self.recent_screens.flatten() / 255.0
        
        # Combine recent actions and flattened screen data
        observation = self.recent_actions + self.get_piece_position() + flattened_screens.tolist()
        
        return observation
    
    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:,:, 0]
    
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
    
    def render(self, reduce_res=True, scale_factor=4):
        game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (scale_factor, scale_factor, 1))
            ).astype(np.uint8)
        return game_pixels_render
    
    def get_map_positions(self):
        map_positions = []
        for i in range(160):
            for j in range(144):
                if self.pyboy.screen.ndarray[i,j,0] == 0:
                    map_positions.append((i,j))
        return map_positions
    
    def get_piece_position(self):
        return [self.pyboy.memory[CURRENT_PIECE_X], self.pyboy.memory[CURRENT_PIECE_Y]]
