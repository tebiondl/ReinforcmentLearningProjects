from pyboy import PyBoy
from player import Player

class PyBoyHandler:
    
    def __init__(self, game_path="ROM/Tetris.gb", state_path=None, net=None, genome=None):
        
        # For no window -> window="null"
        
        self.pyboy = PyBoy(game_path, sound_volume=0)
        self.state_path = state_path
        self.net = net
        self.genome = genome
        self.player = Player(self.pyboy, net, genome)
        self.final_reward = 0

        if self.state_path is not None:
            self.open_state()
            
        self.apply_configuration()
        
    def apply_configuration(self):
        self.pyboy.set_emulation_speed(10)
    
    def start_game(self):       
        # Normal game loop
        return self.game_loop()
        
    def game_loop(self):
        end_game = False
        
        while self.pyboy.tick():
            end_game, reward = self.player.step()
            self.final_reward += reward
            if end_game:
                break
            
        #self.player.save_actions(self.final_reward)
            
        return self.end_game()
        
    def reset_game(self):
        self.pyboy.stop()
        self.pyboy = PyBoy(self.game_path, state_path=self.state_path, sound_volume=0)
        self.apply_configuration()
        self.start_game()

    def end_game(self):
        self.pyboy.stop()
        return self.final_reward
        
    def open_state(self):
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
            
        
            
        
            
        
    