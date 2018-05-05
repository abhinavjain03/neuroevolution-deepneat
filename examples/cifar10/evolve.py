"""
This example produces networks that can remember a fixed-length sequence of bits. It is
intentionally very (overly?) simplistic just to show the usage of the NEAT library. However,
if you come up with a more interesting or impressive example, please submit a pull request!

This example also demonstrates the use of a custom activation function.
"""

from __future__ import division, print_function

import math
import os
import random
import dataPrep
import numpy as np
import shutil

import neat
import network
import visualize

DATA_ROOT_DIR_CIFAR="D:\Downloads\cifar-10-python\cifar-10-batches-py"
DATA_ROOT_DIR_TRAIN="DevanagariHandwrittenCharacterDataset/Train"
DATA_ROOT_DIR_TEST="DevanagariHandwrittenCharacterDataset/Test"
MAX_FITNESS=-10
MAX_VAL_ACCURACY=-1
MAX_TEST_ACCURACY=-1
NUM_OF_GENERATIONS=7


NP_SAVE_PATH="split/train_feat.npy"
if not os.path.exists(NP_SAVE_PATH):
    train_feat, train_class, val_feat, val_class, test_feat, test_class = dataPrep.prepDataCifar(DATA_ROOT_DIR_CIFAR)
    
    # train_feat, train_class, val_feat, val_class = dataPrep.prepData(DATA_ROOT_DIR_TRAIN)
    # test_feat, test_class = dataPrep.prepDataTest(DATA_ROOT_DIR_TEST)
    np.save("split/train_feat.npy",train_feat)
    np.save("split/train_class.npy",train_class)
    np.save("split/val_feat.npy",val_feat)
    np.save("split/val_class.npy",val_class)
    np.save("split/test_feat.npy",test_feat)
    np.save("split/test_class.npy",test_class)
else:
    print("##########Already exists###############")
    train_feat = np.load("split/train_feat.npy")
    train_class = np.load("split/train_class.npy")
    val_feat = np.load("split/val_feat.npy")
    val_class = np.load("split/val_class.npy")
    test_feat = np.load("split/test_feat.npy")
    test_class = np.load("split/test_class.npy")


def eval_genome(genome, config):
    
    global MAX_FITNESS
    global MAX_VAL_ACCURACY
    global MAX_TEST_ACCURACY

    eval_result, test_result = network.create_net_and_train(genome, train_feat, train_class, val_feat, val_class, test_feat, test_class)
    genome.accuracy=eval_result["accuracy"]
    genome.test_accuracy=test_result["accuracy"]
    fitness=((genome.accuracy)*100)-((genome.size())**(1/5))
    # fitness=genome.accuracy*100
    if genome.accuracy > MAX_VAL_ACCURACY:
        MAX_VAL_ACCURACY=genome.accuracy
    if genome.test_accuracy > MAX_TEST_ACCURACY:
        MAX_TEST_ACCURACY=genome.test_accuracy
    if fitness > MAX_FITNESS:
        MAX_FITNESS=fitness
        #save the model

    return fitness


def eval_genomes(genomes, config):
    
    for genome_id, genome in genomes:
        # if len(genome.sequence)==1:
        #     genome.fitness=-100
        #     genome.accuracy=-1
        # else:
        genome.fitness = eval_genome(genome, config)
        shutil.rmtree("F:/NeuroEvolutionTemp/convnet_model", ignore_errors=True)

        to_print=genome.__str__()
        with open("genome_configs","a") as f:
            print(to_print, file=f)

    to_print="##########################MAX_FITNESS:{0}###########################\n##########################MAX_VAL_ACCURACY:{1}###########################\n#####################MAX_TEST_ACCURACY:{2}######################\n".format(MAX_FITNESS,MAX_VAL_ACCURACY,MAX_TEST_ACCURACY)
    with open("genome_configs","a") as f:
        print(to_print, file=f)


def run():
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Demonstration of saving a configuration back to a text file.
   # config.save('test_save_config.txt')


    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))


    winner = pop.run(eval_genomes, NUM_OF_GENERATIONS)


    # Log statistics.
    stats.save()

    # Show output of the most fit genome against a random input.
    print('\nBest genome:\n{!s}'.format(winner))
    # print('\nOutput:')
    # #winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    # num_correct = 0
    # for n in range(num_tests):
    #     print('\nRun {0} output:'.format(n))
    #     seq = [random.choice((0.0, 1.0)) for _ in range(N)]
    #     winner_net.reset()
    #     for s in seq:
    #         inputs = [s, 0.0]
    #         winner_net.activate(inputs)
    #         print('\tseq {0}'.format(inputs))

    #     correct = True
    #     for s in seq:
    #         output = winner_net.activate([0, 1])
    #         print("\texpected {0:1.5f} got {1:1.5f}".format(s, output[0]))
    #         correct = correct and round(output[0]) == s
    #     print("OK" if correct else "FAIL")
    #     num_correct += 1 if correct else 0

    # print("{0} of {1} correct {2:.2f}%".format(num_correct, num_tests, num_correct/num_tests))

    # node_names = {-1: 'input', -2: 'gate', 0: 'output'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)


if __name__ == '__main__':
    run()