from model_handler import ModelHandler

if __name__ == "__main__":
    model_handler = ModelHandler(test=False, generations=100, checkpoint=26)
    model_handler.run()