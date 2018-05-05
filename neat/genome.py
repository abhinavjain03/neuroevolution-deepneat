"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function


from itertools import count
from random import choice, random, shuffle, randint

import sys

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.graphs import creates_cycle
from neat.six_util import iteritems, iterkeys


class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""

    def __init__(self, params):
        # Create full set of available activation functions.

        ################### change acc to the config file in the end ############3
        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_hidden', int),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default')]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1','yes','true','on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0','no','false','off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        write_pretty_params(f, self, [p for p in self._params
                                      if not 'initial_connection' in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        
        # Unique identifier for a genome instance.
        self.key = key
        self.nodes = {}

        # Fitness results.
        self.fitness = None
        self.accuracy = None
        self.test_accuracy = None

        # Number of layes
        self.layers = 0

        # Sequence of gene keys to define a network
        self.sequence = []



    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        #Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            for i in range(config.num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node
                self.layers += 1
                self.sequence.append(node_key)

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in iteritems(parent1_set):
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            self.layers += 1
            self.sequence.append(key)
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    def mutate(self, config):
        """ Mutates this genome. """


        if len(self.nodes) == 1:
            self.mutate_add_node(config)
        else:
            if config.single_structural_mutation:
                div = max(1,(config.node_add_prob + config.node_delete_prob))
                r = random()
                if r < (config.node_add_prob/div):
                    self.mutate_add_node(config)
                else:
                    self.mutate_delete_node(config)
            else:
                if random() < config.node_add_prob:
                    self.mutate_add_node(config)

                if random() < config.node_delete_prob:
                    self.mutate_delete_node(config)



        # Mutate node genes 
        # for ng in self.nodes.values():
        #     ng.mutate(config)
        for key in self.nodes:
            if key==0:
                continue
            self.nodes[key].mutate(config)

    def mutate_add_node(self, config):
        pos = randint(0, self.layers)
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng
        self.layers += 1
        self.sequence.insert(pos, new_node_id)



    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in iterkeys(self.nodes)]
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)


        del self.nodes[del_key]
        self.layers -= 1
        #del from seq
        self.sequence.remove(del_key)

        return del_key


    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if len(self.nodes)!=len(other.nodes):
            node_distance=50
        else:
            if self.nodes or other.nodes:
                disjoint_nodes = 0
                for k2 in iterkeys(other.nodes):
                    if k2 not in self.nodes:
                        disjoint_nodes += 1

                for k1, n1 in iteritems(self.nodes):
                    n2 = other.nodes.get(k1)
                    if n2 is None:
                        disjoint_nodes += 1
                    else:
                        # Homologous genes compute their own distance value.
                        node_distance += n1.distance(n2, config)

                max_nodes = max(len(self.nodes), len(other.nodes))
                node_distance = (node_distance +
                                 (config.compatibility_disjoint_coefficient *
                                  disjoint_nodes)) / max_nodes


        return node_distance

    def size(self):

        factors={
        "conv2d":1,
        "dense":1
        }

        size=0
        for x in self.sequence:
            gene=self.nodes[x]
            if gene.type_of_layer == "conv2d":
                size+=factors[gene.type_of_layer]*(float(gene.num_filters_conv)*float(gene.kernel_size_conv)**2)
            else:
                size+=factors[gene.type_of_layer]*float(gene.num_of_nodes)
        
        return size

    def __str__(self):
        s = "Key: {0}\nFitness: {1}\nValidation Accuracy: {2}\nTest Accuracy: {3}\nNodes:".format(self.key, self.fitness, self.accuracy, self.test_accuracy)
        for k, ng in iteritems(self.nodes):
            s += "\n\t{0} {1!s}".format(k, ng)

        return s

    @staticmethod
    def create_node(config, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        return node
