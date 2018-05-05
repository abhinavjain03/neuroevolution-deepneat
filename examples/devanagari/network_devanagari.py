import tensorflow as tf

# if need to add dropout layer
# dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
# dropout = tf.layers.dropout(
#   inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

tf.logging.set_verbosity(tf.logging.WARN)

# NUMBER_OF_CLASSES=46 devnagari
NUMBER_OF_CLASSES=10  # CIFAR
INPUT_DIM=32
EPOCHS=12

def create_input_layer(features, channel):
	return tf.reshape(features, [-1, INPUT_DIM, INPUT_DIM, channel])

def create_dense_layer(list_of_layers, list_of_nodes, layer_index, node_index, node):

	if node.activation=="relu":
		activation=tf.nn.relu
	if node.activation=="sigmoid":
		activation=tf.nn.sigmoid
	if node.activation=="tanh":
		activation=tf.nn.tanh

	if len(list_of_nodes) == 0: #this means that this is the first layer after input
		#FIND OUT THE FLATTEN VARIABLES
		input_flat = tf.reshape(list_of_layers[layer_index], [-1, 32 * 32 * 1])
		list_of_layers.append(input_flat)
		layer_index+=1
		dense = tf.layers.dense(inputs=input_flat, units=node.num_of_nodes, activation=activation)
		list_of_layers.append(dense)
		layer_index+=1
	else:  #this means there are more layers between input and current layer
		if list_of_nodes[node_index].type_of_layer=="conv2d": #if the previous layer is conv2d
			#fist flatten it to 1d vector
			# pool_flat = tf.reshape(list_of_layers[layer_index],
			# 			 [-1, list_of_layers[layer_index].output_shape[1] * list_of_layers[layer_index].output_shape[1] * list_of_layers[layer_index].output_shape[3]])
			

			p, q, r, s = list_of_layers[layer_index].get_shape()
			# print (p, q, r, s)
			pool_flat = tf.reshape(list_of_layers[layer_index],[-1, q * r * s])
			
			
			layer_index+=1
			list_of_layers.append(pool_flat)

			#create a dense layer
			dense = tf.layers.dense(inputs=pool_flat, units=node.num_of_nodes, activation=activation)
			list_of_layers.append(dense)
			layer_index+=1
		else: #previous layer is a dense layer
			#just add a dense layer
			dense = tf.layers.dense(inputs=list_of_layers[layer_index], units=node.num_of_nodes, activation=activation)
			list_of_layers.append(dense)
			layer_index+=1

	return (list_of_layers, layer_index)

def create_conv_maxpool_layer(list_of_layers, list_of_nodes, layer_index, node_index, node):

	if node.activation=="relu":
		activation=tf.nn.relu
	if node.activation=="sigmoid":
		activation=tf.nn.sigmoid
	if node.activation=="tanh":
		activation=tf.nn.tanh

	if len(list_of_nodes) == 0: #this means that the current layer is the first layer after input
		#so just add a convolution layer

		conv = tf.layers.conv2d(
	      inputs=list_of_layers[layer_index],
	      filters=int(node.num_filters_conv),
	      kernel_size=[int(node.kernel_size_conv), int(node.kernel_size_conv)],
	      padding="same",
	      strides=int(node.stride_conv),
	      data_format="channels_last",
	      activation=activation)
		list_of_layers.append(conv)
		layer_index+=1
		#then add a maxpooling layer
		pool=tf.layers.max_pooling2d(inputs=conv, pool_size=[int(node.poolsize_pool), int(node.poolsize_pool)], padding='same', data_format="channels_last", strides=int(node.stride_pool))
		list_of_layers.append(pool)
		layer_index+=1
		
	else: #this means that there are more layers between input and the current layer
		
		if list_of_nodes[node_index].type_of_layer=="conv2d": # if the previous layer is a convolution layer, i.e the maxpool layer
			#just add a convolution layer
			conv = tf.layers.conv2d(
				inputs=list_of_layers[layer_index],
				filters=int(node.num_filters_conv),
				kernel_size=[int(node.kernel_size_conv), int(node.kernel_size_conv)],
				padding="same",
				strides=int(node.stride_conv),
				data_format="channels_last",
				activation=activation)
			list_of_layers.append(conv)
			layer_index+=1
			
			#then add a maxpooling layer
			pool=tf.layers.max_pooling2d(inputs=conv, pool_size=[int(node.poolsize_pool), int(node.poolsize_pool)], padding='same', data_format="channels_last", strides=int(node.stride_pool))
			list_of_layers.append(pool)
			layer_index+=1
			
		else: #the previous layer is a dense layer
			#reshape the dense layer to the proper convolution size and add the layers (this makes no sense but still have to do it to make
			# this work, the fitness of the network will decrease, so this will be discarded at the end, so should not be a problem)
			
			if list_of_nodes[node_index].num_of_nodes == "512":
				size=16
				channel=2
			elif list_of_nodes[node_index].num_of_nodes == "1024":
				size=32
				channel=1
			elif list_of_nodes[node_index].num_of_nodes == "2048":
				size=32
				channel=2
			dense_conv = tf.reshape(list_of_layers[layer_index], [-1, size, size, channel])
			layer_index+=1
			list_of_layers.append(dense_conv)
			
			#add a convolution layer
			conv = tf.layers.conv2d(
				inputs=dense_conv,
				filters=int(node.num_filters_conv),
				kernel_size=[int(node.kernel_size_conv), int(node.kernel_size_conv)],
				padding="same",
				strides=int(node.stride_conv),
				data_format="channels_last",
				activation=activation)
			list_of_layers.append(conv)
			layer_index+=1
			
			#add a maxpooling layer
			pool=tf.layers.max_pooling2d(inputs=conv, pool_size=[int(node.poolsize_pool), int(node.poolsize_pool)], padding='same', data_format="channels_last", strides=int(node.stride_pool))
			list_of_layers.append(pool)
			layer_index+=1
			
	
	return (list_of_layers, layer_index)
		

def cnn_model_fn(features, labels, mode, params):
	
	gen=params["gen"]
	# input_layer=create_input_layer(features["x"], 1) For Devanagari
	input_layer=create_input_layer(features["x"], 3)

	list_of_layers=[]
	list_of_nodes=[]
	list_of_layers.append(input_layer)
	layer_index=0
	node_index=-1

	if(len(gen.sequence)>0):
		for x in gen.sequence:
			node=gen.nodes[x]
			if (x == 0 or x=="0"):
				continue
			if node.type_of_layer == "dense":
				list_of_layers,layer_index = create_dense_layer(list_of_layers, list_of_nodes, layer_index, node_index, node)

			elif node.type_of_layer == "conv2d":
				list_of_layers,layer_index = create_conv_maxpool_layer(list_of_layers, list_of_nodes, layer_index, node_index, node)

			list_of_nodes.append(node)
			node_index+=1

	# print ("SHAPE")
	# for i in range(layer_index):
	# 	p,q,r,s = (list_of_layers[i].get_shape())
	# 	print p, q, r, s
	

	# FLATTEN THE TENSOR IF PREV LAYER IS CONV OR INPUT
	if (node_index == -1):
		p, q, r, s = input_layer.get_shape()
		flat_layer = tf.reshape(input_layer,[-1, q * r * s])
	elif list_of_nodes[node_index].type_of_layer=="conv2d": #if the previous layer is conv2d
		#fist flatten it to 1d vector
		p, q, r, s = list_of_layers[layer_index].get_shape()
		# print (p, q, r, s)
		flat_layer = tf.reshape(list_of_layers[layer_index],[-1, q * r * s])
	else:
		flat_layer = list_of_layers[layer_index]

	# flat_layer = list_of_layers[layer_index]
	# print (flat_layer.get_shape())

	logits = tf.layers.dense(inputs=flat_layer, units=NUMBER_OF_CLASSES)
	# print (logits.get_shape())
	predictions = {
	# Generate predictions (for PREDICT and EVAL mode)
	"classes": tf.argmax(input=logits, axis=1),
	# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	# `logging_hook`.
	"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0)
		# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
		train_op = optimizer.minimize(
	    	loss=loss,
	    	global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
	  "accuracy": tf.metrics.accuracy(
	      labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
	  mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def create_net_and_train(genome, train_feat, train_class, val_feat, val_class, test_feat, test_class):
	
	#DATA PREP SECTION
	# mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	# train_data = mnist.train.images  # Returns np.array
	# train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	# eval_data = mnist.test.images  # Returns np.array
	# eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	# Create the Estimator




	mydict={"gen":genome}

	#### To limit amount of GPU Memory used by the process##########################

	# opts=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
	# conf=tf.ConfigProto(gpu_options=opts)
	# training_config=tf.estimator.RunConfig(session_config=conf)

	# my_classifier = tf.estimator.Estimator(
	# 	model_fn=cnn_model_fn,
	# 	model_dir="F:/NeuroEvolutionTemp/convnet_model",
	# 	config=training_config,
	# 	params=mydict)

	#### To limit amount of GPU Memory used by the process###########################

	my_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn,
		model_dir="F:/NeuroEvolutionTemp/convnet_model",
		params=mydict)


	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log,
		every_n_iter=50)


	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_feat},
		y=train_class,
		batch_size=128,
		num_epochs=EPOCHS,
		shuffle=True)



	my_classifier.train(
		input_fn=train_input_fn,
		steps=None,
		hooks=[logging_hook])

	

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": val_feat},
		y=val_class,
		num_epochs=1,
		shuffle=False)

	eval_results = my_classifier.evaluate(input_fn=eval_input_fn)

	test_input_fn = tf.estimator.inputs.numpy_input_fn(
	x={"x": test_feat},
	y=test_class,
	num_epochs=1,
	shuffle=False)

	test_results = my_classifier.evaluate(input_fn=test_input_fn)
	
	return eval_results, test_results













