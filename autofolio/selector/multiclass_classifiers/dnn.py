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
from keras.layers import Dense, Activation
from keras import backend as K

from sklearn.preprocessing import StandardScaler

__author__ = "Marius Lindauer"
__license__ = "BSD"


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

        layer1_neurons = UniformIntegerHyperparameter(
            name="dnn:layer1_neurons", lower=10, upper=100, default=64, log=True)
        cs.add_hyperparameter(layer1_neurons)
        cond = InCondition(
            child=layer1_neurons, parent=classifier, values=["DNN"])
        cs.add_condition(cond)
        layer2_neurons = UniformIntegerHyperparameter(
            name="dnn:layer2_neurons", lower=10, upper=100, default=62, log=True)
        cs.add_hyperparameter(layer2_neurons)
        cond = InCondition(
            child=layer2_neurons, parent=classifier, values=["DNN"])
        cs.add_condition(cond)
        layer3_neurons = UniformIntegerHyperparameter(
            name="dnn:layer3_neurons", lower=10, upper=100, default=16, log=True)
        cs.add_hyperparameter(layer3_neurons)
        cond = InCondition(
            child=layer3_neurons, parent=classifier, values=["DNN"])
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
        return "RandomForest"

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
        self.model.add(Dense(units=config["dnn:layer1_neurons"], input_dim=X.shape[1]))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(units=config["dnn:layer2_neurons"]))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(units=config["dnn:layer3_neurons"]))
        self.model.add(Activation('sigmoid'))
        # output layer
        self.model.add(Dense(units=Y.shape[1]))
        self.model.add(Activation('softmax'))
        
        Y_gap = Y - np.repeat([np.min(Y, axis=1)], Y.shape[1], axis=0).T
            
        X = self.scaler.fit_transform(X)    
        
        def as_loss(y_true, y_pred):
            return K.dot(y_true, K.transpose(y_pred))
        
        self.model.compile(#loss='categorical_crossentropy',
              loss=as_loss,
              optimizer=keras.optimizers.SGD(lr=config["dnn:lr"], momentum=config["dnn:momentum"], nesterov=True),
              metrics=['accuracy',as_loss])
        
        self.model.fit(X, Y_gap, epochs=100, batch_size=32)#Y.shape[0])

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
        