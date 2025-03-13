from pyboy_handler import PyBoyHandler
from multiprocessing import Process, Queue
import neat

class ModelHandler:
    
    def __init__(self, game_path="ROM/PokemonRed.gb", state_path="ROM/states/has_pokedex.state", config_file="config-neat.txt", generations=20):
        self.list_of_pb = []
        self.generations = generations
        self.population_size = 3
        self.game_path = game_path
        self.state_path = state_path
        self.config_file = config_file
        self.current_generation = 0

    def create_handler(self, net, genome, queue):
        handler = PyBoyHandler(self.game_path, self.state_path, net, genome)
        fitness = handler.start_game()
        queue.put((genome.key, fitness))
    
    def eval_genomes(self, genomes, config):
        
        self.current_generation += 1
        
        processes = []
        queue = Queue()
        
        for _, genome in genomes:
            genome.fitness = 0  # start with fitness level of 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            p = Process(target=self.create_handler, args=(net, genome, queue))
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            
        while not queue.empty():
            genome_id, fitness = queue.get()
            for _, genome in genomes:
                if genome.key == genome_id:
                    genome.fitness = fitness
                    break

    def run(self):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         self.config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        winner = p.run(self.eval_genomes, self.generations)

        # show final stats
        print('\nBest genome:\n{!s}'.format(winner))
