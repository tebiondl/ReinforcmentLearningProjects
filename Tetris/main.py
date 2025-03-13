from model_handler import ModelHandler
from pyboy import PyBoy

if __name__ == "__main__":
    model_handler = ModelHandler(test=False, generations=100)
    model_handler.run()