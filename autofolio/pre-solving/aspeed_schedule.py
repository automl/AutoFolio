import logging
import math

import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from autofolio.data.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class Aspeed(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace, cutoff:int):
        '''
            adds parameters to ConfigurationSpace
            
            Arguments
            ---------
            cs: ConfigurationSpace
                configuration space to add new parameters and conditions
            cutoff: int
                maximal possible time for aspeed
        '''
        
        pre_solving = UniformIntegerHyperparameter("pre:cutoff", lower=0, upper=cutoff, default=0, log=True)
        cs.add_hyperparameter(pre_solving)
        

    def __init__(self, classifier_class):
        '''
            Constructor
        '''
        self.logger = logging.getLogger("Aspeed")

        self.schedule = []

    def fit(self, scenario: ASlibScenario, config: Configuration):
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
        self.logger.info("Compute Presolving Schedule with Aspeed")

        X = scenario.performance_data.values()
        times = [("i%d" % (i), "a%d" % (a), "%d" % (math.ceil(X[i, a])))
                 for i in X.shape[0] for j in X.shape[1]]
        
        kappa = config["pre:cutoff"]
        
        # call aspeed and save schedule

    def predict(self, scenario: ASlibScenario):
        '''
            transform ASLib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas

            Returns
            -------
                schedule: [(solver, time)]
                    schedule of solvers with a running time budget
        '''

        return dict((inst, self.schedule) for inst in scenario.instances)
