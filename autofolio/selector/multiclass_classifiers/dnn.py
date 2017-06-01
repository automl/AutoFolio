import random
import sys

import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition, AndConjunction
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from aslib_scenario.aslib_scenario import ASlibScenario

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from click.core import batch

__author__ = "Marius Lindauer"
__license__ = "BSD"

LAYERS = 3
EPOCHS = 5000

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

        def add_main_cond(param, further_parents:list=None):
            cs.add_hyperparameter(param)
            cond = InCondition(
                child=param, parent=classifier, values=["DNN"])
            
            if further_parents:
                conds = [cond]
                for [parent,value] in further_parents:
                    cond = InCondition(
                        child=param, parent=parent, values=[value])
                    conds.append(cond)
                cs.add_condition(AndConjunction(*conds))
                    
            else:
                cs.add_condition(cond)

        for i in range(1,LAYERS+1):
            layer_neurons = UniformIntegerHyperparameter(
                name="dnn:layer%d_neurons" %(i), lower=10, upper=400, default=128, log=True)
            add_main_cond(layer_neurons)
            act_func = CategoricalHyperparameter(
                name="dnn:layer%d_act_func" %(i), choices=["elu","relu","tanh","sigmoid"], default="relu")
            add_main_cond(act_func)
            dropout = UniformFloatHyperparameter(
                name="dnn:layer%d_dropout_rate" %(i), lower=0, upper=0.99, default=0.0)
            add_main_cond(dropout)

        #batch size
        batch_size = UniformIntegerHyperparameter(
            name="dnn:batch_size", lower=1, upper=128, default=32, log=True)
        add_main_cond(batch_size)

        ## Optimizer
        optimizer = CategoricalHyperparameter(
            name="dnn:optimizer", choices=["SGD", "Adam"], default="Adam")
        add_main_cond(optimizer)
        
        ##SGD
        lr = UniformFloatHyperparameter(
            name="dnn:sgd:lr", lower=0.000001, upper=0.1, default=0.01, log=True)
        add_main_cond(lr, [[optimizer,"SGD"]])
        momentum = UniformFloatHyperparameter(
            name="dnn:sgd:momentum", lower=0.6, upper=0.999, default=0.9, log=True)
        add_main_cond(momentum, [[optimizer,"SGD"]])
            
        ## Adam
        lr = UniformFloatHyperparameter(
            name="dnn:adam:lr", lower=0.000001, upper=1.0, default=0.0001, log=True)
        add_main_cond(lr, [[optimizer,"Adam"]])
        beta_1 = UniformFloatHyperparameter(
            name="dnn:adam:beta_1", lower=0.7, upper=0.999999, default=0.9)
        add_main_cond(beta_1, [[optimizer,"Adam"]])
        beta_2 = UniformFloatHyperparameter(
            name="dnn:adam:beta_2", lower=0.9, upper=0.999999, default=0.999)
        add_main_cond(beta_2, [[optimizer,"Adam"]])
        epsilon = UniformFloatHyperparameter(
            name="dnn:adam:epsilon", lower=1e-20, upper=0.1, default=1e-08, log=True)
        add_main_cond(epsilon, [[optimizer,"Adam"]])
        decay = UniformFloatHyperparameter(
            name="dnn:adam:decay", lower=0.0, upper=0.1, default=0.000)
        add_main_cond(decay, [[optimizer,"Adam"]])
        
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
       # Y = Y[:300,:]
       # X = X[:300,:]
        
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
        
        if config["dnn:optimizer"] == "SGD":
            optimizer = keras.optimizers.SGD(lr=config["dnn:sgd:lr"], momentum=config["dnn:sgd:momentum"], nesterov=True)
        if config["dnn:optimizer"] == "Adam":
            optimizer = keras.optimizers.Adam(lr=config["dnn:adam:lr"], 
                                          beta_1=config["dnn:adam:beta_1"], 
                                          beta_2=config["dnn:adam:beta_2"],
                                          epsilon=config["dnn:adam:epsilon"], 
                                          decay=config["dnn:adam:decay"]
                                          )
        
        self.model.compile(
              loss='categorical_crossentropy',
              #loss=as_loss,
              optimizer=optimizer,
              metrics=['accuracy',as_loss])
        
        es = EarlyStopping(monitor="val_loss", patience=1)
        
        history = self.model.fit(X, Y_hot, epochs=EPOCHS, batch_size=config["dnn:batch_size"],
                                 #validation_split=0.33,
                                 #callbacks=[es]
                                 )
        
        plt.plot(history.history["loss"])
       # plt.plot(history.history["val_loss"])
       # plt.plot(history.history["val_acc"])
       # plt.plot(history.history["val_as_loss"])
       # plt.ylabel('model loss')
        plt.xlabel('epoch')
        #plt.legend(['train loss', 'val_loss', 'val_acc', 'val_as_loss'], loc='upper right')
        plt.legend(['train loss'], loc='upper right')
        plt.savefig("loss_dnn_%d.png" %(random.randint(1,2**31)))
        
        Y_pred = self.model.predict(X)
        print(Y_pred)
        print(Y)
        print(Y_hot)
        print(Y_gap)
        #print(X)
        print("AC loss: %f" %(np.sum([np.dot(y_g,y_p.T) for y_g,y_p in zip(Y_gap, Y_pred)])))
        
        plot_model(self.model, to_file='model.png')
        
#        sys.exit(1)

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
        