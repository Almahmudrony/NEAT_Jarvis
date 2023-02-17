import neat
import string
import random

# Define the fitness function
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0.0
        # Generate a random input string
        input_string = ''.join(random.choice(string.ascii_lowercase) for _ in range(200))
        # Generate the expected output string
        output_string = input_string
        # Activate the network with the input string
        output = net.activate(input_string)
        # Compare the output with the expected output and update the fitness
        genome.fitness = sum(1 for expected, actual in zip(output_string, output) if expected == actual)

# Load the NEAT configuration file
config_file = 'config-feedforward.txt'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

# Create the population
population = neat.Population(config)

# Add a reporter to track the progress of the evolution
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

# Run the evolution
winner = population.run(eval_genomes)

# Show the winner genome
print('\nBest genome:\n{!s}'.format(winner))
