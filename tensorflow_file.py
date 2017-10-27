import tensorflow as tf

def data_to_tensor(x,y)
  data_x = np.array(x)
  data_y = np.array(y)

  ## x = [none = to flat array , 28 * 28 ]
  ## x = height * width , height = none.
  x =tf.placeholder("float",[None,len(word_bank)])
  y=tf.placeholder("float")

  ## 1 input layer -- 3 hidden layer -- 1 outputlayer

  ## creating hidden layer framework
  ## hidden-layer-1 = ['weights','basis'] where weights=matrix of [data,nodes] and bias = [nodes]
  return x,y

def neural_network_framework(x):
    hidden_layer_1 = {"weights":tf.Variable(tf.random_normal([len(data_x),125])),
                      "basis":tf.Variable(tf.random_normal([n_nodes1]))}

    hidden_layer_2 = {"weights": tf.Variable(tf.random_normal([125, 500])),
                      "basis": tf.Variable(tf.random_normal([500]))}

    hidden_layer_3 = {"weights": tf.Variable(tf.random_normal([500, 625])),
                      "basis": tf.Variable(tf.random_normal([625]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([625, 1])),
                      "basis": tf.Variable(tf.random_normal([1]))}


##  layer1 = data*weights + bias  ||   [1*data]*[data*nodes]+[1*bias] = layer1
##  then activation of layer1(i.e neuron burn up or not)
## ouput-layer dosnt have activation layer


    layer_1 = tf.add(tf.matmul(data_x,hidden_layer_1['weights']), hidden_layer_1['basis'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['basis'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['basis'])
    layer_3 = tf.nn.relu(layer_3)

    o_layer = tf.add(tf.matmul(layer_3, output_layer['weights']), output_layer['basis'])


    return o_layer

## hidden layer framework is completed


def train_neural_network(x,y,neural_network_framework):
##data is passed through neurals and output is get in prediction
    prediction = neural_network_framework(x)
##comparing our NEURAL-OUTPUT with (y) in cost function
	##mean is calculated of every data point
	cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost_function)
    epoch_no = 4
    with tf.Session() as sess:
    
       sess.run(tf.initialize_all_variables())
	for i in range(0,100):
                i, c = sess.run([optimizer, cost_function], feed_dict={x: data_x,y: data_y})
                                        
nn(x)