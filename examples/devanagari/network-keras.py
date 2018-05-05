from __future__ import division, print_function, absolute_import
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Conv2D, MaxPooling2D, Dropout, Reshape
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

batch_size = 128
nb_classes = 46
nb_epoch = 12


def create_dense_layer(list_of_nodes, node_index, node, model):
	if (len(list_of_nodes) == 0): #this means that this is the first layer
		#reshape input to 1D
		# model.add(Reshape((32*32*1), input_shape=(32, 32, 1)))
		model.add(Flatten(input_shape=(32, 32, 1)))
		#add dense layer
		model.add(Dense(int(node.num_of_nodes)))
		model.add(Activation(node.activation))

	else:  #this means there are more layers between input and current layer
		if list_of_nodes[node_index].type_of_layer=="conv2d": #if the previous layer is conv2d
			#flatten and add dense
			model.add(Flatten())
			model.add(Dense(int(node.num_of_nodes)))
			model.add(Activation(node.activation))

		else: #previous layer is a dense layer
			#just add a dense layer
			model.add(Dense(int(node.num_of_nodes)))
			model.add(Activation(node.activation))

	return model


def create_conv_maxpool_layer(list_of_nodes, node_index, node, model):

	if (len(list_of_nodes) == 0): #this means that the current layer is the first layer
		#so just add a convolution layer
		model.add(Conv2D(int(node.num_filters_conv), 
			kernel_size=(int(node.kernel_size_conv),int(node.kernel_size_conv)),
			strides=int(node.stride_conv), 
			padding='same', 
			input_shape=(32, 32, 1), 
			activation=node.activation,
			data_format='channels_last'))

		# add a max pooling layer
		model.add(MaxPooling2D(pool_size=(int(node.poolsize_pool), int(node.poolsize_pool)), strides=int(node.stride_pool),data_format='channels_last'))

		
	else: #this means that there are more layers between input and the current layer
		
		if list_of_nodes[node_index].type_of_layer=="conv2d": # if the previous layer is a convolution layer, i.e the maxpool layer
			#just add a convolution layer
			model.add(Conv2D(int(node.num_filters_conv), 
				kernel_size=(int(node.kernel_size_conv),int(node.kernel_size_conv)),
				strides=int(node.stride_conv), 
				padding='same',  
				activation=node.activation,
				data_format='channels_last'))
			
			#then add a maxpooling layer
			model.add(MaxPooling2D(pool_size=(int(node.poolsize_pool), int(node.poolsize_pool)), strides=int(node.stride_pool),data_format='channels_last'))
			
		else: #the previous layer is a dense layer
			#reshape the dense layer to the proper convolution size and add the layers (this makes no sense but still have to do it to make
			# this work, the fitness of the network will decrease, so this will be discarded at the end, so should not be a problem)
			
			if list_of_nodes[node_index].num_of_nodes == "512":
				model.add(Reshape((16, 16, 2)))
			elif list_of_nodes[node_index].num_of_nodes == "1024":
				model.add(Reshape((32, 32, 1)))
			elif list_of_nodes[node_index].num_of_nodes == "2048":
				model.add(Reshape((32, 32, 2)))

			
			#add a convolution layer
			model.add(Conv2D(int(node.num_filters_conv), 
				kernel_size=(int(node.kernel_size_conv),int(node.kernel_size_conv)),
				strides=int(node.stride_conv), 
				padding='same',  
				activation=node.activation,
				data_format='channels_last'))
			
			#add a maxpooling layer
			model.add(MaxPooling2D(pool_size=(int(node.poolsize_pool), int(node.poolsize_pool)), strides=int(node.stride_pool),data_format='channels_last'))
	
	return model


# ADDED TEST AS WELL
def create_net_and_train(genome, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
	list_of_nodes=[]
	node_index=-1

	model = Sequential()

	
	if(len(genome.sequence)>0):
		for x in genome.sequence:
			node=genome.nodes[x]
			if (x == 0 or x=="0"):
				continue
			if node.type_of_layer == "dense":
				model = create_dense_layer(list_of_nodes, node_index, node, model)

			elif node.type_of_layer == "conv2d":
				model = create_conv_maxpool_layer(list_of_nodes, node_index, node, model)

			list_of_nodes.append(node)
			node_index+=1


	# FLATTEN IF NO PREV LAYER
	if (node_index == -1):
		model.add(Flatten(input_shape=(32, 32, 1)))
	elif list_of_nodes[node_index].type_of_layer=="conv2d": #if the previous layer is conv2d
		#fist flatten it to 1d vector
		model.add(Flatten())


	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))


	model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

	model.fit(train_dataset, train_labels, batch_size=batch_size, nb_epoch=nb_epoch,
	          verbose=1)

	val_score = model.evaluate(valid_dataset, valid_labels, verbose=0)
	# print('Validation score:', score[0])
	# print('Validation accuracy:', score[1])

	test_score = model.evaluate(test_dataset, test_labels, verbose=0)
	# print('Test score:', score[0])
	# print('Test accuracy:', score[1])

	return (val_score[1], test_score[1])