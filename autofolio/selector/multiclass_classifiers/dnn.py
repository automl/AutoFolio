import random

import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from aslib_scenario.aslib_scenario import ASlibScenario

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

__author__ = "Marius Lindauer"
__license__ = "BSD"

LAYERS = 2

class DNN(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        try:
            classifier = cs.get_hyperparameter("multi_classifier")
            classifier.choices.append("DNN")
            classifier._num_choices += 1
        except KeyError:
            classifier = CategoricalHyperparameter(
                "multi_classifier", choices=["DNN"], default="DNN")
            cs.add_hyperparameter(classifier)

        for i in range(1,LAYERS+1):
            layer_neurons = UniformIntegerHyperparameter(
                name="dnn:layer%d_neurons" %(i), lower=10, upper=400, default=128, log=True)
            cs.add_hyperparameter(layer_neurons)
            cond = InCondition(
                child=layer_neurons, parent=classifier, values=["DNN"])
            cs.add_condition(cond)
            act_func = CategoricalHyperparameter(
                name="dnn:layer%d_act_func" %(i), choices=["elu","relu","tanh","sigmoid"], default="relu")
            cs.add_hyperparameter(act_func)
            cond = InCondition(
                child=act_func, parent=classifier, values=["DNN"])
            cs.add_condition(cond)
            dropout = UniformFloatHyperparameter(
                name="dnn:layer%d_dropout_rate" %(i), lower=0, upper=0.9, default=0.5)
            cs.add_hyperparameter(dropout)
            cond = InCondition(
                child=dropout, parent=classifier, values=["DNN"])
            cs.add_condition(cond)

        lr = UniformFloatHyperparameter(
            name="dnn:lr", lower=0.001, upper=0.1, default=0.01, log=True)
        cs.add_hyperparameter(lr)
        cond = InCondition(
            child=lr, parent=classifier, values=["DNN"])
        cs.add_condition(cond)
        
        momentum = UniformFloatHyperparameter(
            name="dnn:momentum", lower=0.6, upper=0.999, default=0.9, log=True)
        cs.add_hyperparameter(momentum)
        cond = InCondition(
            child=momentum, parent=classifier, values=["DNN"])
        cs.add_condition(cond)
        
    def __init__(self):
        '''
            Constructor
        '''

        self.model = None
        self.scaler = StandardScaler()
        
    def __str__(self):
        return "DNN"

    def fit(self, X, Y, config: Configuration):
        '''
            fit pca object to ASlib scenario data

            Arguments
            ---------
            X: numpy.array
                feature matrix
            Y: numpy.array
                performance matrix 
            config: ConfigSpace.Configuration
                configuration

        '''
        self.model = Sequential()
        self.model.add(Dense(units=config["dnn:layer1_neurons"], 
                             activation=config["dnn:layer1_act_func"],
                             input_dim=X.shape[1]))
        self.model.add(Dropout(rate=config["dnn:layer1_dropout_rate"]))
        for i in range(2,LAYERS+1):
            self.model.add(Dense(units=config["dnn:layer%d_neurons" %(i)], activation=config["dnn:layer%d_act_func" %(i)]))
            self.model.add(Dropout(rate=config["dnn:layer%d_dropout_rate" %(i)]))
            
        # output layer
        self.model.add(Dense(units=Y.shape[1]))
        self.model.add(Activation('softmax'))
        
        Y_gap = Y - np.repeat([np.min(Y, axis=1)], Y.shape[1], axis=0).T
        Y_gap /= np.max(Y_gap)            
        
        Y_hot = np.zeros(Y.shape)
        Y_sel = np.argmin(Y, axis=1)
        for i,y in enumerate(Y_sel):
            Y_hot[i,y] = 1
        
        X = self.scaler.fit_transform(X)
        print(X.shape)    
        print(np.mean(Y_gap,axis=0))
        print(np.sum(Y_hot,axis=0))
        
        def as_loss(y_true, y_pred):
            return K.dot(y_true, K.transpose(y_pred))
        
        self.model.compile(
              #loss='categorical_crossentropy',
              loss=as_loss,
              optimizer=keras.optimizers.SGD(lr=config["dnn:lr"], momentum=config["dnn:momentum"], nesterov=True),
              metrics=['accuracy',as_loss])
        
        es = EarlyStopping(monitor="val_loss", patience=1)
        
        history = self.model.fit(X, Y_gap, epochs=200, batch_size=16, validation_split=0.33,
                                 callbacks=[es])#Y.shape[0])
        
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.ylabel('model loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("loss_dnn_%d.png" %(random.randint(1,2**31)))

    def predict(self, X):
        '''
            transform ASLib scenario data

            Arguments
            ---------
            X: numpy.array
                instance feature matrix

            Returns
            -------

        '''
        
        X = self.scaler.transform(X)  
        
        return self.model.predict(X)
    
    def get_attributes(self):
        '''
            returns a list of tuples of (attribute,value) 
            for all learned attributes
            
            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        attr = []
        #attr.append("max_depth = %d" %(self.model.max_depth))
        return attr
        