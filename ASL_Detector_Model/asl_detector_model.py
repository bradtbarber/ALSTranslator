import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

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
    'w1' : tf.Variable(np.load(os.path.abspath('ASL_Detector_Model\\w1.npy'))),
    # Weight for Convolutional Layer 2: 5x5 filter, 32 input channels, 64 output channels
    'w2' : tf.Variable(np.load(os.path.abspath('ASL_Detector_Model\\w2.npy'))),
    # Weight for Fully Connected Layer 1: 49 * 64 input channels, 1024 output channels
    'w3' : tf.Variable(np.load(os.path.abspath('ASL_Detector_Model\\w3.npy'))),
    # Weight for Convolutional Layer 1: 1024 input channels, 25(number of classes) output channels
    'w4' : tf.Variable(np.load(os.path.abspath('ASL_Detector_Model\\w4.npy'))),
}

biases = {
    # Bias for Convolutional Layer 1
    'b1' : tf.Variable(np.load(os.path.abspath('ASL_Detector_Model\\b1.npy'))),
    # Bias for Convolutional Layer 2
    'b2' : tf.Variable(np.load(os.path.abspath('ASL_Detector_Model\\b2.npy'))),
    # Bias for Fully Connected Layer 1
    'b3' : tf.Variable(np.load(os.path.abspath('ASL_Detector_Model\\b3.npy'))),
    # Bias for Outout Layer
    'b4' : tf.Variable(np.load(os.path.abspath('ASL_Detector_Model\\b4.npy'))),
}

def one_hot_encode(y):
    return np.eye(25)[y]

# Wrapper function for creating a Convolutional Layer
def conv2d(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Wrapper function for creating a Pooling Layer
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

def neural_network(x, weight, bias, dropout):
    x = tf.reshape(x, shape = [-1, 28, 28, 1])
    
    conv1 = conv2d(x, weight['w1'], bias['b1']) # Convolutional Layer 1
    pool1 = maxpool2d(conv1) # Pooling Layer 1
    
    conv2 = conv2d(pool1, weight['w2'], bias['b2']) # Convolutional Layer 2
    pool2 = maxpool2d(conv2) # Pooling Layer 1
    
    # Reshaping output of previous convolutional layer to fit the fully connected layer
    fc = tf.reshape(pool2, [-1, weights['w3'].get_shape().as_list()[0]])

    # Fully Connected Layer 1
    fc = tf.add(tf.matmul(fc, weight['w3']), bias['b3'])
    fc = tf.nn.relu(fc)
    
    # Applying dropout on Fully Connected Layer
    fc = tf.nn.dropout(fc, dropout)
    
    out = tf.add(tf.matmul(fc, weight['w4']), bias['b4']) # Output Layer
    return out

X = tf.placeholder(tf.float32, shape = [None, n_input]) # Placeholder for Feature Matrix
Y = tf.placeholder(tf.float32, shape = [None, n_classes]) # Placeholder for Labels
keep_prob = tf.placeholder(tf.float32) # Placeholder for Dropout Rate

y_pred = neural_network(X, weights, biases, 1.0)
init = tf.global_variables_initializer()

def get_prediction(img):
    prob_dict = {}
    with tf.Session() as sess:
        # Running Initializer
        sess.run(init)
        pred = sess.run(y_pred, feed_dict = { X : img, keep_prob : dropout })
    img = img.reshape(28, 28)
    pred = list(pred.flatten())
    maxPred = chr(pred.index(max(pred)) + 65)
    return (img, maxPred, pred)

# Run  prediction tests on the first
def run_test(limit):
    data_test = pd.read_csv(os.path.abspath('ASL_Detector_Model\\input\\sign_mnist_test.csv'))
    result = ''

    data_test.head()

    x_test = data_test.iloc[:, 1:].values
    y_test = data_test.iloc[:, :1].values.flatten()
    y_test = one_hot_encode(y_test)
    x_test.shape, y_test.shape
    i = 0
    for x in x_test:
        image, max_pred, predictions = get_prediction(x.reshape(1, 784))
        i = i + 1
        result = result + max_pred

        if i == limit and limit != 0:
            break
    return result

# Runs CNN model on provided images
# Returns the best guess based on the model's calculated certainty as first return
# Returns the full certainty arrays for each image as calculed by the model as second return   
def read_and_translate_image_capture_output(image_file):
    # Read in CSV file and get number of images
    data = pd.read_csv(image_file)
    row_count = len(data.index)
    
    # Initialize variables to hold return values
    best_guess = ''
    predictions = []

    # Iterate over all rows (images) in the csv file and try to detect the ASL sign present
    for i in range(row_count):
        # Get Prediction for Image
        x = data.iloc[i].values
        image, max_pred, pred = get_prediction(x.reshape(1, 784))
        
        # Display image with max confidence guess
        plt.imshow(image, cmap = 'gray')
        plt.title(max_pred)
        plt.show()

        # Append results to return values
        best_guess = best_guess + max_pred
        predictions.append(pred)
    
    return best_guess, predictions

def translate_asl_images(image_file):
    # Run test to make sure model is producing accurate predictions
    print('performing test run on first 5 test images...')
    expected_result = 'GFKAD'
    test_result = run_test(5)
    print('test result: ' + test_result)
    print('expected result: ' + expected_result)
    if test_result == expected_result:
        print('test successful!\n')
    else:
        print('test failed. Results may be incorrect...\n')

    # Read in data from Camera Capture output directory and attempt to translate
    best_guess, predictions = read_and_translate_image_capture_output(image_file)
    print('Translated ASL: ' + best_guess + '\n')

    return best_guess, predictions
