import random
import string
import neat

input_size = 200
output_size = 200

input_strings = ["Hello, how are you?",
                 "What is your name?",
                 "How old are you?",
                 "What do you like to do?",
                 "Do you have any pets?",
                 "What's your favorite food?",
                 "Do you like to travel?",
                 "What is your favorite book?",
                 "What kind of music do you like?",
                 "What do you think of the weather today?"]

def generate_random_string(length):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    error = 0.0
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = net.activate(xi)
        error += (output[0] - xo[0]) ** 2
    genome.fitness = 1 - error
    return genome

def get_fitness_feedback():
    while True:
        feedback = input("Please rate the fitness of this response (1-10): ")
        if feedback.isdigit() and int(feedback) in range(1, 11):
            return int(feedback)
        else:
            print("Invalid input. Please enter a number between 1 and 10.")

def handle_turn(net):
    input_string = input("You: ")
    output = net.activate(list(input_string))
    response = ''.join([chr(int(round(x * 255))) for x in output])
    print(f"Bot: {response}")
    fitness = get_fitness_feedback()
    return input_string, output, fitness

# Define the main function for training the neural network
def train():
    # Load the NEAT configuration file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'neat_config.txt')
    # Create the initial population of neural networks
    population = neat.Population(config)
    # Add a reporter to output progress during training
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    # Run the NEAT algorithm for a specified number of generations
    for generation in range(10):
        print(f"Generation {generation+1}")
        best_genome, = population.run(evaluate_genome, 10)
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        # Keep talking with the player and actively training the neural network until the player exits
        while True:
            input_string, output, fitness = handle_turn(net)
            best_genome.fitness += fitness
            population.reproduction.reproduce([best_genome], config, 1, 9)
            net = neat.nn.FeedForwardNetwork.create(best_genome, config)

if __name__ == '__main__':
    train()
