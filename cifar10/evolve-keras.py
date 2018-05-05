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
import numpy as np
import shutil
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range

import neat
import network
import visualize

MAX_FITNESS=-10
MAX_ACCURACY=-1
NUM_OF_GENERATIONS=50
pickle_file = 'devanagari.pickle'
nb_classes = 46

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  # print('Training set', train_dataset.shape, train_labels.shape)
  # print('Validation set', valid_dataset.shape, valid_labels.shape)
  # print('Test set', test_dataset.shape, test_labels.shape)

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(nb_classes) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_feat, train_class = reformat(train_dataset, train_labels)
val_feat, val_class = reformat(valid_dataset, valid_labels)
test_feat, test_class = reformat(test_dataset, test_labels)
# print('Training set', train_feat.shape, train_class.shape)
# print('Validation set', val_feat.shape, val_class.shape)
# print('Test set', test_feat.shape, test_class.shape)


def eval_genome(genome, config):
    
    global MAX_FITNESS
    global MAX_ACCURACY

    val_result, test_result = network.create_net_and_train(genome, train_feat, train_class, val_feat, val_class, test_feat, test_class)
    genome.accuracy=val_result
    genome.test_accuracy=test_result
    fitness=(genome.accuracy)*100-(genome.size())**(1/5)
    if genome.accuracy > MAX_ACCURACY:
        MAX_ACCURACY=genome.accuracy
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
        # shutil.rmtree("tmp")

        to_print=genome.__str__()
        with open("genome_configs","a") as f:
            print(to_print, file=f)

    to_print="##########################MAX_FITNESS:{0}###########################\n##########################MAX_ACCURACY:{1}###########################\n".format(MAX_FITNESS,MAX_ACCURACY)
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

if __name__ == '__main__':
    run()