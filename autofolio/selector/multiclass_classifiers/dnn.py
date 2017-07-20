import random
import sys
import logging

import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition, AndConjunction
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from aslib_scenario.aslib_scenario import ASlibScenario

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import keras
from keras import backend as K

import matplotlib.pyplot as plt
from mini_autonet.autonet import AutoNet
from param_net.param_fcnet import ParamFCNetClassification


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

    def __init__(self):
        '''
            Constructor
        '''

        self.model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger("DNN")
        
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
        MAX_EPOCHS = 2000
        
        logging.basicConfig(level="INFO")
        
        n_shape_init = X.shape[0]
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)

        # remove common timeouts        
        all_timeouts = Y.min(axis=1) >= 600
        X.drop(Y[all_timeouts].index, inplace=True)
        Y.drop(Y[all_timeouts].index, inplace=True)
        
        # remove duplicates
        dups = X.duplicated()
        Y.drop(X[dups].index, inplace=True)
        X.drop(X[dups].index, inplace=True)
        
        X = X.values
        Y = Y.values
        
        self.logger.debug("Dropped %d observations because of duplicates and common timeouts." %(n_shape_init - X.shape[0]))
        
        X = self.scaler.fit_transform(X)
        
        def as_loss(y_true, y_pred):
            return K.dot(y_true, K.transpose(y_pred))
        
        if False:
            Y_gap = Y - np.repeat([np.min(Y, axis=1)], Y.shape[1], axis=0).T
            Y_gap /= np.max(Y_gap)            
            Y = Y_gap
            
            an = AutoNet(max_layers=5, n_classes=Y.shape[1])
            config = an.fit(X_train=X, y_train=Y, X_valid=X, y_valid=Y, max_epochs=50, 
                            runcount_limit=50, loss_func=as_loss,
                            metrics=[as_loss])
            
            pc = ParamFCNetClassification(config=config, n_feat=X.shape[1],
                                              n_classes=Y.shape[1],
                                              max_num_epochs=500,
                                              metrics=[as_loss],
                                              verbose=1)
        elif True:
            Y_sel = np.argmin(Y, axis=1)
            Y_hot = keras.utils.to_categorical(Y_sel, num_classes=Y.shape[1])
            Y = Y_hot
            
            an = AutoNet(max_layers=5, n_classes=Y.shape[1])
            config = an.fit(X_train=X, y_train=Y, X_valid=X, y_valid=Y,
                            max_epochs_intensify=100, max_epochs=MAX_EPOCHS, 
                            runcount_limit=200)
            
            pc = ParamFCNetClassification(config=config, n_feat=X.shape[1],
                                  n_classes=Y.shape[1],
                                  max_num_epochs=MAX_EPOCHS,
                                  verbose=1)
        else:
            Y_sel = np.argmin(Y, axis=1)
            Y_hot = keras.utils.to_categorical(Y_sel, num_classes=Y.shape[1])
            Y = Y_sel
            cs = ParamFCNetClassification.get_config_space(max_num_layers=5,
                                                       use_l2_regularization=False,
                                                       use_dropout=False)
            config = cs.get_default_configuration()
            pc = ParamFCNetClassification(config=config, n_feat=X.shape[1],
                                  n_classes=Y_hot.shape[1],
                                  max_num_epochs=100,
                                  verbose=1)
            
        
        history = pc.train(X_train=X, y_train=Y, X_valid=X,
                           y_valid=Y, n_epochs=MAX_EPOCHS)

        self.model = pc.model
        

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
        