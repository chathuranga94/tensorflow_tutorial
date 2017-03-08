import tensorflow as tf

# Input images of numbers with relevant values: Here it's 60,000 samples
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/tensorflow/mnist/input_data", one_hot = True)

# Input Layer -> Hidden Layer 1 -> Hidden Layer 2 -> Hidden Layer 3 -> Output Layer
# Number of nodes in each layer 
input_nodes  = 784   # 28*28 bits of input image
layer1_nodes = 500
layer2_nodes = 500
layer3_nodes = 500
output_nodes = 10    # 0-9 numbers of output value

# Batch_size from total input images to calculate
batch_size = 1000

# Input variables to train neural network
x = tf.placeholder('float', [None, input_nodes])
y = tf.placeholder('float')

# Model to train  : setting up compuation graph
def neural_network_model(data):

    # Weights and Biases variables will be set randomly intially but will be modified.
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([input_nodes, layer1_nodes])),
                      'biases':tf.Variable(tf.random_normal([layer1_nodes]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([layer1_nodes, layer2_nodes])),
                      'biases':tf.Variable(tf.random_normal([layer2_nodes]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([layer2_nodes, layer3_nodes])),
                      'biases':tf.Variable(tf.random_normal([layer3_nodes]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([layer3_nodes, output_nodes])),
                    'biases':tf.Variable(tf.random_normal([output_nodes])),}

    
    # For Hidden Layer 1 : (input data * weights) + biases and likewise for other layers 
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    # Activation function for Hidden Layer 1 -> Rectified Linear
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    # Cost function to check variation between prediction and known label
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # Optimizator to reduce cost // optional learning_rate
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    # Number of epochs : feedforward + backpropagation
    hm_epochs = 10
    # Start session to run constructed model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # Loop through all photos according to batch_size
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # Get next batch_size chunk of images
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # Run optimize and cost function by inputting x and y
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                # Calculate loss for the entire epoch
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        # Check whether index value of prediction and y same ? 
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # Accuracy check for test images (not using training data)
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)

# Original code/tutorial from https://pythonprogramming.net/machine-learning-tutorials/