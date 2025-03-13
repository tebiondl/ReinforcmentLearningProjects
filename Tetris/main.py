from model_handler import ModelHandler
from pyboy import PyBoy

if __name__ == "__main__":
    #model_handler = ModelHandler()
    #model_handler.run()
    PyBoy("ROM/Tetris.gb", sound_volume=0)