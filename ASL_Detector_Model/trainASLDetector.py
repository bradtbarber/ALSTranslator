import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("./input"))

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

data = pd.read_csv('./input/sign_mnist_train.csv')
print('Dataframe Shape:', data.shape)

data.head()

x = data.iloc[:, 1:].values
print("Number of images:", x.shape[0])
print("Number of pixels in each image:", x.shape[1])

y = data.iloc[:, :1].values.flatten()
print('Labels:\n', y)
print('Shape of Labels:', y.shape)

def next_batch(batch_size, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[: batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def display_images(data):
    x, y = data
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].reshape(28, 28), cmap = 'binary')
        ax.set_xlabel(chr(y[i] + 65))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# Display a batch of the data
display_images(next_batch(9, x, y))

z = dict(Counter(list(y)))
labels = z.keys()
frequencies = [z[i] for i in labels]
labels = [chr(i + 65) for i in z.keys()]

plt.figure(figsize = (20, 10))
plt.bar(labels, frequencies)
plt.title('Frequency Distribution of Alphabets', fontsize = 20)
plt.show()

def one_hot_encode(y):
    return np.eye(25)[y]

y_encoded = one_hot_encode(y)
print('Shape of y after encoding:', y_encoded.shape)

# Training Parameters
learning_rate = 0.001
epochs = 2000
batch_size = 128
display_step = 100

# Neural Network Hyperparameters
n_input = 784
n_classes = 25
dropout = 0.75

# Placeholders
X = tf.placeholder(tf.float32, shape = [None, n_input]) # Placeholder for Feature Matrix
Y = tf.placeholder(tf.float32, shape = [None, n_classes]) # Placeholder for Labels
keep_prob = tf.placeholder(tf.float32) # Placeholder for Dropout Rate

weights = {
    # Weight for Convolutional Layer 1: 5x5 filter, 1 input channel, 32 output channels
    'w1' : tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # Weight for Convolutional Layer 2: 5x5 filter, 32 input channels, 64 output channels
    'w2' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # Weight for Convolutional Layer 3: 5x5 filter, 64 input channels, 128 output channels
    'w3' : tf.Variable(tf.random_normal([5, 5, 64, 128])),
    # Weight for Fully Connected Layer 1: 49 * 128 input channels, 1024 output channels
    'w4' : tf.Variable(tf.random_normal([262144, 1024])),
    # Weight for Fully Connected Layer 2: 49 * 124 input channels, 1024 output channels
    'w5' : tf.Variable(tf.random_normal([1024, 512])),
    # Weight for Output Layer: 512 input channels, 25(number of classes) output channels
    'w6' : tf.Variable(tf.random_normal([512, n_classes]))
}

biases = {
    # Bias for Convolutional Layer 1
    'b1' : tf.Variable(tf.random_normal([32])),
    # Bias for Convolutional Layer 2
    'b2' : tf.Variable(tf.random_normal([64])),
    # Bias for Convolutional Layer 3
    'b3' : tf.Variable(tf.random_normal([128])),
    # Bias for Fully Connected Layer 1
    'b4' : tf.Variable(tf.random_normal([1024])),
    # Bias for Fully Connected Layer 2
    'b5' : tf.Variable(tf.random_normal([512])),
    # Bias for Outout Layer
    'b6' : tf.Variable(tf.random_normal([n_classes]))
}

# Wrapper function for creating a Convolutional Layer
def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Wrapper function for creating a Pooling Layer
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

def neural_network(x, weight, bias, dropout, k, stride):
    x = tf.reshape(x, shape = [-1, 28, 28, 1])
    
    # Convolutional Layer 1
    conv1 = conv2d(x, weight['w1'], bias['b1'], strides = stride[0])
    conv1 = maxpool2d(conv1, k = k[0])
    
    # Convolutional Layer 2
    conv2 = conv2d(conv1, weight['w2'], bias['b2'], strides = stride[1])
    conv2 = maxpool2d(conv2, k = k[1])

    # Convolutional Layer 3
    conv3 = conv2d(conv2, weight['w3'], bias['b3'], strides = stride[2])
    conv3 = maxpool2d(conv3, k = k[2])
    
    # Fully Connected Layer 1
    # Reshaping output of previous convolutional layer to fit the fully connected layer
    fc1 = tf.reshape(conv3, [-1, weights['w4'].get_shape().as_list()[0]])
   
    # Add fully connected layer
    fc1 = tf.add(tf.matmul(fc1, weight['w4']), bias['b4'])
    fc1 = tf.nn.relu(fc1) # Relu activation function
    fc1 = tf.nn.dropout(fc1, dropout) # Applying dropout on Fully Connected Layer

    # Add fully connected layer
    fc2 = tf.add(tf.matmul(fc1, weight['w5']), bias['b5'])
    fc2 = tf.nn.relu(fc2) # Relu activation function
    fc2 = tf.nn.dropout(fc2, dropout) # Applying dropout on Fully Connected Layer
    
    out = tf.add(tf.matmul(fc2, weight['w6']), bias['b6']) # Output Layer
    return out

#TRAINING NEURAL NETWORK

# Splitting the dataset into Training and Holdout(Test set)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size = 0.33, random_state = 42)
print('X train shape', X_train.shape)
print('y train shape', y_train.shape)
print('X test shape', X_test.shape)
print('y test shape', y_test.shape)

k_vals = [[2,2,2], [2,2,4]]
stride_vals = [[1,1,1], [2,2,2]]

for k in k_vals:
    for stride in stride_vals:
        print()
        print('************************************************************************************')
        print('For K = {' + str(k[0]) + ', ' + str(k[1]) + ', ' + str(k[2]) + '} and Stride = {' + str(stride[0]) + ', ' + str(stride[1]) + ', ' + str(stride[2]) + '}')
        print()

        logits = neural_network(X, weights, biases, keep_prob, k, stride)

        loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss_op)

        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # Running Initializer
            sess.run(init)
            cost_hist, acc_hist = [], []
            for epoch in range(1, epochs + 1):
                _x, _y = next_batch(batch_size, X_train, y_train)

                #print('X train shape', _x.shape)
                #print('y train shape', _y.shape)

                # Running Optimizer
                sess.run(train_op, feed_dict = { X : _x, Y : _y, keep_prob : dropout })
                if epoch % display_step == 0:
                    # Calculating Loss and Accuracy on the current Epoch
                    loss, acc = sess.run([loss_op, accuracy], feed_dict = { X : _x, Y : _y, keep_prob : 1.0 })
                    loss = sum(loss)
                    cost_hist.append(loss)
                    acc_hist.append(acc)
                    print('Epoch ' + str(epoch) + ', Cost: ' + str(loss) + ', Accuracy: ' + str(acc * 100) + ' %')
            print('-' * 50)
            print('\nOptimization Finished\n')
            print('Accuracy on Training Data: ' + str(sess.run(accuracy,
                                                            feed_dict = {
                                                                X : X_train,
                                                                Y : y_train,
                                                                keep_prob : 1.0
                                                            }) * 100) + ' %')
            print('Accuracy on Test Data: ' + str(sess.run(accuracy,
                                                        feed_dict = {
                                                            X : X_test,
                                                            Y : y_test,
                                                            keep_prob : 1.0
                                                        }) * 100) + ' %')

        plt.plot(list(range(len(cost_hist))), cost_hist)
        plt.title("Change in cost")
        plt.show()

        plt.plot(list(range(len(acc_hist))), acc_hist)
        plt.title("Change in accuracy")
        plt.show()
        print('************************************************************************************')
        print()

print('Training on the whole dataset....\n')
with tf.Session() as sess:
    sess.run(init) # Running Initializer
    cost_hist, acc_hist = [], []
    for epoch in range(1, epochs + 1):
        _x, _y = next_batch(batch_size, x, y_encoded)
        # Running Optimizer
        sess.run(train_op,
                 feed_dict = {
                     X : _x,
                     Y : _y,
                     keep_prob : dropout
                 })
        if epoch % display_step == 0:
            # Calculating Loss and Accuracy on the current Epoch
            loss, acc = sess.run([loss_op, accuracy],
                                 feed_dict = {
                                     X : _x,
                                     Y : _y,
                                     keep_prob : 1.0
                                 })
            loss = sum(loss)
            cost_hist.append(loss)
            acc_hist.append(acc)
            print('Epoch ' + str(epoch) + ', Cost: ' + str(loss) + ', Accuracy: ' + str(acc * 100) + ' %')
    print('-' * 50)
    print('\nOptimization Finished\n')
    print('Accuracy after training on whole dataset Data: ' + str(sess.run(accuracy,
                                                       feed_dict = {
                                                           X : x,
                                                           Y : y_encoded,
                                                           keep_prob : 1.0
                                                       }) * 100) + ' %')
    W = sess.run(weights)
    B = sess.run(biases)

plt.plot(list(range(len(cost_hist))), cost_hist)
plt.title("Change in cost")
plt.show()

plt.plot(list(range(len(acc_hist))), acc_hist)
plt.title("Change in accuracy")
plt.show()


#PREDICTING MODEL ON TEST DATA

data_test = pd.read_csv('.\\input\\sign_mnist_test.csv')
print('Dataframe Shape:', data_test.shape)

data_test.head()

x_test = data_test.iloc[:, 1:].values
y_test = data_test.iloc[:, :1].values.flatten()
y_test = one_hot_encode(y_test)
x_test.shape, y_test.shape

X = tf.placeholder(tf.float32, shape = [None, n_input]) # Placeholder for Feature Matrix
Y = tf.placeholder(tf.float32, shape = [None, n_classes]) # Placeholder for Labels
keep_prob = tf.placeholder(tf.float32) # Placeholder for Dropout Rate

y_pred = neural_network(X, W, B, 1.0, k)

def get_prediction(img):
    with tf.Session() as sess:
        pred = sess.run(y_pred, feed_dict = { X : img, keep_prob : 1.0 })
    img = img.reshape(28, 28)
    pred = list(pred.flatten())
    pred = chr(pred.index(max(pred)) + 65)
    return (img, pred)

image, pred = get_prediction(x_test[1].reshape(1, 784))
plt.imshow(image, cmap = 'binary')
plt.title(pred)
plt.show()

#for key in W.keys():
#    np.save(key, W[key])

#for key in B.keys():
#    np.save(key, B[key])