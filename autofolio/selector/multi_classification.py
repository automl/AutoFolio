import logging
import traceback

import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from aslib_scenario.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class MultiClassifier(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''

        try:
            selector = cs.get_hyperparameter("selector")
            selector.choices.append("MultiClassifier")
        except KeyError:
            selector = CategoricalHyperparameter(
                "selector", choices=["MultiClassifier"], default="MultiClassifier")
            cs.add_hyperparameter(selector)
            
        mclassifier = cs.get_hyperparameter("multi_classifier")
        cond = InCondition(child=mclassifier, parent=selector, values=["MultiClassifier"])
        cs.add_condition(cond)

    def __init__(self, multi_classifier_class):
        '''
            Constructor
        '''
        self.classifier = None
        self.logger = logging.getLogger("MultiClassifier")
        self.multi_classifier_class = multi_classifier_class

    def fit(self, scenario: ASlibScenario, config: Configuration):
        '''
            fit multi-class classifier to ASlib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration
        '''
        self.logger.info("Fit MultiClassifier with %s" %
                         (self.multi_classifier_class))

        self.algorithms = scenario.algorithms

        from sklearn.utils import check_array
        from sklearn.tree._tree import DTYPE

        n_algos = len(scenario.algorithms)
        X = scenario.feature_data.values
        # since sklearn (at least the RFs) 
        # uses float32 and we pass float64,
        # the normalization ensures that floats
        # are not converted to inf or -inf
        X = (X - np.min(X)) / (np.max(X) - np.min(X))
        
        Y = scenario.performance_data.values
        self.classifier = self.multi_classifier_class()
        self.classifier.fit(X, Y, config)
        
    def predict(self, scenario: ASlibScenario):
        '''
            predict schedules for all instances in ASLib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas

            Returns
            -------
                schedule: {inst -> (solver, time)}
                    schedule of solvers with a running time budget
        '''

        if scenario.algorithm_cutoff_time:
            cutoff = scenario.algorithm_cutoff_time
        else:
            cutoff = 2**31

        X = scenario.feature_data.values
        Y = self.classifier.predict(X) # vector of 0s and a single 1 for each instance
        
        algo_indx = np.argmax(Y, axis=1)
        
        schedules = dict((str(inst),[s]) for s,inst in zip([(scenario.algorithms[i], cutoff+1) for i in algo_indx], scenario.feature_data.index))
        #self.logger.debug(schedules)
        return schedules

    def get_attributes(self):
        '''
            returns a list of tuples of (attribute,value) 
            for all learned attributes
            
            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        class_attr = self.multi_classifier_class.get_attributes()
        attr = [{self.multi_classifier_class.__name__:class_attr}]

        return attr