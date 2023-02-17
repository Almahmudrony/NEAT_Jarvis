import random
import string
import neat
#import visualize

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

def clamp(minvalue, value, maxvalue):
    return max(minvalue, min(value, maxvalue))

def ascii_float_clamp(value):
    return float(clamp(0,value,0x10FFFF))

def string_to_float_ascii_arr(string):
    truncated_string = string[:200]
    ascii_arr = [float(ord(c)) for c in truncated_string] + [32.0] * (200 - len(truncated_string))
    return ascii_arr

def float_ascii_arr_to_string(float_ascii_arr):
    ascii_arr = []
    for f in float_ascii_arr:
        i = int(round(f))
        if i >= 0 and i <= 0x10FFFF:
            ascii_arr.append(i)
    return ''.join([chr(i) for i in ascii_arr])

def generate_random_string(length):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def evaluate_genome(genomes, config):
    for genome_id, genome in genomes:
        print("Training.... ",genome_id)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_error = 0.0
        for input_string in input_strings:
            output = float_ascii_arr_to_string(net.activate(string_to_float_ascii_arr(input_string)))
            target_output = generate_random_string(output_size)
            #print(target_output)
            total_error += sum((ord(x) - ord(y)) ** 2 for x, y in zip(output, target_output))
        genome.fitness = 1 / (total_error + 1)
    #return genome.fitness

def get_fitness_feedback():
    #while True:
    feedback = input("Please rate the fitness of this response (0-10): ")
    if feedback.isdigit() and int(feedback) in range(0, 11):
        return int(feedback)
    else:
        print("Invalid input. Please enter a number between 0 and 10.")
        return 0

def handle_turn(net):
    input_string = input("You: ")
    output = float_ascii_arr_to_string(net.activate(string_to_float_ascii_arr(input_string)))
    response = ''.join([chr(int(round(float(ord(x)) * 255))) for x in output])
    print(f"Bot: {response}")
    fitness = get_fitness_feedback()
    print(fitness)
    return input_string, output, fitness


def active_learning(genomes, config):
    for genome_id, genome in genomes:
        print("Training.... ",genome_id)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        input_string = input("You: ")
        output = float_ascii_arr_to_string(net.activate(string_to_float_ascii_arr(input_string)))
        response = ''.join([chr(int(round(float(ord(x)) * 255))) for x in output])
        print(f"Bot: {response}")
        fitness = get_fitness_feedback()
        print(fitness)
        genome.fitness += fitness

def train():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'neat_config.txt')
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    for generation in range(10):
        print(f"Generation {generation+1}")
        best_genome = population.run(evaluate_genome, 1)
        #visualize.draw_net(config, best_genome, True)
        #visualize.plot_stats(stats, ylog=False, view=True)
        #visualize.plot_species(stats, view=True)
        #net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        while True:
            winner = population.run(active_learning, 1)
            #visualize.draw_net(config, winner, True)
            #visualize.plot_stats(stats, ylog=False, view=True)
            #visualize.plot_species(stats, view=True)
            #input_string, output, fitness = handle_turn(net)
            #best_genome.fitness += fitness
            #population.reproduction.reproduce(config, population.species, 1, 9)
            #net = neat.nn.FeedForwardNetwork.create(best_genome, config)

if __name__ == '__main__':
    train()
