import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special
# to measure elapsed time
from timeit import default_timer as timer
# pandas for reading CSV files
import pandas as pd
# Keras
import keras
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import np_utils

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers import Embedding
from keras.layers import LSTM

from keras.constraints import maxnorm

from sklearn.model_selection import GridSearchCV

# suppress Tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Read fight_dataset files as a DataFrames----------------------------------------------------------------

# Read sign_mnist_test.csv (column 0 = labels, 1-785 = values)
asl_dataset = pd.read_csv('.\\input\\sign_mnist_test.csv', delimiter=',')
asl_dataset.dataframeName = 'sign_mnist_test.csv'
#---------------------------------------------------------------------------------------------------------

# Prepare fight_datasets for model training---------------------------------------------------------------

asl_dataset = asl_dataset.sample(frac=1).reset_index(drop=True)

#Create training and test datasets
training_data=asl_dataset.sample(frac=1.0,random_state=200)
test_data=asl_dataset.drop(training_data.index).reset_index(drop=True)
training_data=training_data.reset_index(drop=True)

# the first column holds the expected value
train_labels = training_data['label'].values
test_labels = test_data['label'].values

# convert labels to one-hot vectors
y_train = np_utils.to_categorical(train_labels, 25)
y_test = np_utils.to_categorical(test_labels, 25)

# all other columns are the training data  
training_data.drop(columns=['label'],inplace=True)
test_data.drop(columns=['label'],inplace=True)

# Convert inputs to vectors
x_train = np.asfarray(training_data)
x_test = np.asfarray(test_data)
#---------------------------------------------------------------------------------------------------------

# Function to create model, required for KerasClassifier -------------------------------------------------
def create_model(
    activation='relu', 
    optimizer='Adadelta', 
    init_mode='uniform', 
    dropout_rate=0.0, 
    weight_constraint=0,
    shape=[5,5],
    input_size=784
    ):

    # create model
    model = Sequential()

    for i in range(len(shape)):
        if i == 0:
            model.add(
                Dense(
                    shape[i], activation=activation, input_shape=(int(input_size),), kernel_initializer=init_mode 
                    #kernel_constraint=maxnorm(weight_constraint)
                )
            )
        else:
            model.add(
                Dense(
                    shape[i], activation=activation, kernel_initializer=init_mode 
                    #kernel_constraint=maxnorm(weight_constraint)
                )
            )
        model.add(Dropout(dropout_rate))  

    model.add(Dense(25, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
#---------------------------------------------------------------------------------------------------------

# Use GridSearch to test various model parameters --------------------------------------------------------

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#initialize hyper-parameter arrays
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#init_mode = ['normal', 'zero']
#init_mode = ['zero', 'normal']
#weight_constraint = [1, 2, 3, 4, 5]
#dropout_rate = [0.0, 0.4, 0.8]
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
optimizer = ['Adam', 'Adadelta']
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'linear']
activation = ['sigmoid']

epochs = 100
batch_size = 256
learning_rate = 0.001

f= open("tuning_results.txt","w+")
f.write("##########################################\n\n")

f.write("Defaults: {activation='relu', optimizer='SGD', init_mode='uniform', dropout_rate=0.0, weight_constraint=0}\n")
f.write("Epochs: " + str(epochs) + " Batch_size: " + str(batch_size) + "\n\n")

#################################fight_dataset_large#####################################
shape = [[784,250,50], [784,700,300,100]]
#activation = ['linear', 'relu', 'softmax', 'sigmoid']

model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

# define the grid search parameters
param_grid = dict(
    #init_mode=init_mode, 
    #weight_constraint=weight_constraint, 
    #dropout_rate=dropout_rate, 
    optimizer=optimizer, 
    activation=activation,
    shape=shape,
    input_size = [784]
)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=None)
grid_result = grid.fit(x_train, y_train)

# summarize train results
f.write("----------------fight_dataset_large----------------\n")
f.write("acc,stdev,parameters\n")

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
f.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    f.write("%f,%f,%r\n" % (mean, stdev, param))
f.write("---------------------------------------------------\n\n")
#########################################################################################

# ##################################fight_dataset_red######################################
# #shape = [[100,50,20,10,4], [50,25,12,6,3], [100,75,50,20,10,4]]
# optimizer = ['Adam']
# shape = [[112,20,10,4,2]]
# activation = ['linear']

# model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

# # define the grid search parameters
# param_grid = dict(
#     #init_mode=init_mode, 
#     #weight_constraint=weight_constraint, 
#     #dropout_rate=dropout_rate, 
#     optimizer=optimizer, 
#     activation=activation,
#     shape=shape,
#     input_size = [int(x_train_r.shape[1])]
# )

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=None)
# grid_result = grid.fit(x_train_r, y_train_r)

# # summarize train results
# f.write("----------------fight_dataset_red------------------\n")
# f.write("acc,stdev,parameters\n")

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# f.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#     f.write("%f,%f,%r\n" % (mean, stdev, param))
# f.write("---------------------------------------------------\n\n")
# #########################################################################################

# #################################fight_dataset_red_2#####################################
# shape = [[225], [225,50], [225,50,12]]
# activation = ['linear']

# model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

# # define the grid search parameters
# param_grid = dict(
#     #init_mode=init_mode, 
#     #weight_constraint=weight_constraint, 
#     #dropout_rate=dropout_rate, 
#     #optimizer=optimizer, 
#     activation=activation,
#     shape=shape,
#     input_size = [int(x_train_r2.shape[1])]
# )

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=None)
# grid_result = grid.fit(x_train_r2, y_train_r2)

# # summarize train results
# f.write("----------------fight_dataset_red_2----------------\n")
# f.write("acc,stdev,parameters\n")

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# f.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#     f.write("%f,%f,%r\n" % (mean, stdev, param))
# f.write("---------------------------------------------------\n\n")
# #########################################################################################

# #################################fight_dataset_small#####################################
# shape = [[4],[6]]
# activation = ['sigmoid', 'linear']
# dropout_rate = [0.0,0.4,0.8]
# optimizer = ['Adam']

# model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

# # define the grid search parameters
# param_grid = dict(
#     init_mode=init_mode, 
#     #weight_constraint=weight_constraint, 
#     dropout_rate=dropout_rate, 
#     optimizer=optimizer, 
#     activation=activation,
#     shape=shape,
#     input_size = [int(x_train_s.shape[1])]
# )

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=None)
# grid_result = grid.fit(x_train_s, y_train_s)

# # summarize train results
# f.write("------------------fight_dataset_s-------------------\n")
# f.write("acc,stdev,parameters\n")

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# f.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#     f.write("%f,%f,%r\n" % (mean, stdev, param))
# f.write("---------------------------------------------------\n\n")
# #########################################################################################

#################################fight_dataset_tiny######################################
#shape = [[6,4,2],[18,9,4]]
#activation = ['sigmoid']
#dropout_rate=[0.0,0.2,0.4,0.6,0.8]
#optimizer = ['Adamax','Nadam']

#model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

## define the grid search parameters
#param_grid = dict(
#    init_mode=init_mode, 
#    weight_constraint=weight_constraint, 
#    dropout_rate=dropout_rate, 
#    optimizer=optimizer, 
#    activation=activation,
#    shape=shape,
#    input_size = [int(x_train_t.shape[1])]
#)

#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=None)
#grid_result = grid.fit(x_train_t, y_train_t)

# summarize train results
#f.write("------------------fight_dataset_t-------------------\n")
#f.write("acc,stdev,parameters\n")

#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#f.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))
#    f.write("%f,%f,%r\n" % (mean, stdev, param))
#f.write("---------------------------------------------------\n\n")
#########################################################################################

#f.write("###########################################\n")
#f.close()

# for (x_tr,y_tr,x_ts,y_ts) in zip(x_train,y_train,x_test,y_test):

    # in_size = int(x_tr.shape[1])

    # #-----------------------------------------------------------------------------------------------------
    # # define the model
    # model = Sequential()
    # model.add(Dense(int(in_size / 2), activation='relu', input_shape=(int(in_size),), bias=False))
    # model.add(Dense(2, activation='sigmoid', bias=False))
    # model.summary()
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # # train the model
    # model.fit(x_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=1)

    # # test the model
    # score = model.evaluate(x_ts, y_ts)
    # print(model.metrics_names)
    # print(score)

    # #-----------------------------------------------------------------------------------------------------
    
    # # define the model
    # model_2 = Sequential()
    # model_2.add(Dense(int(in_size / 4), activation='relu', input_shape=(int(in_size),), bias=False))
    # model_2.add(Dense(2, activation='sigmoid', bias=False))
    # model_2.summary()
    # model_2.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # # train the model
    # model_2.fit(x_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=1)

    # # test the model
    # score = model_2.evaluate(x_ts, y_ts)
    # print(model_2.metrics_names)
    # print(score)
    # #-----------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------


# # number of imput, hidden and output nodes
# #-----------------------------------------------
# # input_nodes = 449
# # hidden_nodes = 200
# # hidden_nodes_2 = 100
# # hidden_nodes_3 = 50
# # hidden_nodes_4 = 25
# # hidden_nodes_5 = 10
# # output_nodes = 2

# input_nodes = 450
# hidden_nodes = 100
# hidden_nodes_2 = 25
# hidden_nodes_3 = 12
# hidden_nodes_4 = 4
# # hidden_nodes_5 = 6
# # hidden_nodes_6 = 6
# # hidden_nodes_7 = 6
# # hidden_nodes_8 = 3
# output_nodes = 2
# #-----------------------------------------------

# # learning rate
# learning_rate = 0.0001

# # create a Keras model
# model = Sequential()

# #-----------------------------------------------
# # model.add(Dense(hidden_nodes, activation='sigmoid', input_shape=(input_nodes,), bias=False))
# # model.add(Dense(hidden_nodes_2, activation='sigmoid', bias=False))
# # model.add(Dense(hidden_nodes_3, activation='sigmoid', bias=False))
# # model.add(Dense(hidden_nodes_4, activation='sigmoid', bias=False))
# # model.add(Dense(hidden_nodes_5, activation='sigmoid', bias=False))
# # model.add(Dense(output_nodes, activation='sigmoid', bias=False))

# model.add(Dense(hidden_nodes, activation='relu', input_shape=(input_nodes,), bias=False))
# # model.add(Embedding(input_dim=449, input_shape=(input_nodes,), output_dim=128))
# # model.add(LSTM(64))
# # model.add(Dropout(0.5))
# # model.add(Dense(hidden_nodes_2, activation='relu', bias=False))
# # model.add(Dropout(0.5))
# # model.add(Dense(hidden_nodes_3, activation='relu', bias=False))
# # model.add(Dropout(0.5))
# # model.add(Dense(hidden_nodes_4, activation='sigmoid', bias=False))
# # model.add(Dropout(0.5))
# # model.add(Dense(hidden_nodes_5, activation='sigmoid', bias=False, kernel_regularizer=regularizers.l2(0.01)))
# # model.add(Dropout(0.5))
# # model.add(Dense(hidden_nodes_6, activation='sigmoid', bias=False, kernel_regularizer=regularizers.l2(0.01)))
# # model.add(Dropout(0.35))
# # model.add(Dense(hidden_nodes_7, activation='sigmoid', bias=False, kernel_regularizer=regularizers.l2(0.01)))
# # model.add(Dropout(0.35))
# # model.add(Dense(hidden_nodes_8, activation='relu', bias=False, kernel_regularizer=regularizers.l2(0.01)))
# # model.add(Dropout(0.35))
# model.add(Dense(output_nodes, activation='sigmoid', bias=False))
# #-----------------------------------------------

# # print the model summary
# model.summary()

# # set the optimizer (Adam is one of many optimization algorithms derived from stochastic gadient descent)
# opt = optimizers.Adam(lr=learning_rate)
# #opt = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)

# # define the error criterion ("loss"), optimizer and an optional metric to monitor during training
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


# # epochs is the number of times the training data set is used for training
# epochs = 30

# # batch size = 1 to match the previous approach
# batch_size = 1

# # train the model
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

# # save the model
# model.save('UFC_Predictor_3layer_keras.h5')
# print('model saved')

# score = model.evaluate(x_test, y_test)

# print(model.metrics_names)
# print(score)

# #----------------------------------------------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------------------------------------------

# def build_model(dense_size=1000, dense_layers=1, dropout=0.35, conv_layers=2, conv_filters=60, conv_sz=5):
#     model = Sequential()
#     model.add(Conv2D(filters=conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu', batch_input_shape=(None, 1, 449,1)))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     if conv_layers == 2:
#         model.add(Conv2D(filters=conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Flatten())
#     for i in range(0,dense_layers):
#         model.add(Dense(dense_size, activation='relu'))
#         model.add(Dropout(dropout))
#     model.add(Dense(10, activation='softmax', activity_regularizer='l1_l2'))
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model
    
# # Uncomment one or more of the below to find the best combinations!
# param_grid = dict(
#     dense_size=(600,1000),
#     #dense_layers=(1,2),
#     #dropout=(0.35,0.50),
#     #conv_layers=(1,2),
#     #conv_filters=(60),
#     #conv_sz=(3,5),
# )

# gsc = GridSearchCV(estimator=KerasClassifier(build_fn=build_model,
#                                              batch_size=128,
#                                              epochs=1,
#                                              verbose=2),
#                    param_grid=param_grid)

# grid_result = gsc.fit(x_train, y_train)

# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))