import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration  

from autofolio.data.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"

class PairwiseClassifier(self):
    
    @abstractmethod
    def add_params(cs:ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        
    def __init__(self):
        '''
            Constructor
        '''
        self.classifiers = []
        
        
    def fit(self, scenario:ASlibScenario, config:Configuration, classifier_class):
        '''
            fit pca object to ASlib scenario data
            
            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration
            classifier_class: selector.classifier.*
                class for classification
        '''
        
        n_algos = len(self.scenario.algorithms)
        X = scenario.feature_data.values
        for i in range(n_algos):
            for j in range(i,n_algos):
                y_i = scenario[self.scenario.algorithms[i]]
                y_j = scenario[self.scenario.algorithms[j]]
                y = y_i < y_j
                weights = np.abs(y_i - y_j)
                cls = classifier_class()
                cls.fit(self, X, y, weights, config)
                self.classifiers.append(cls)

    def predict(self, feature_vector):
        '''
            transform ASLib scenario data

            Arguments
            ---------
            feature_vector: numpy.array
                instance feature vector
                
            Returns
            -------
                
        '''
    
