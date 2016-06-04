import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

__author__ = "Marius Lindauer"
__license__ = "BSD"


class ImputerWrapper(object):

    @staticmethod
    def add_params(cs):
        '''
            adds parameters to ConfigurationSpace 
        '''
        stratgey = CategoricalHyperparameter(
            "imputer_strategy", choices=["mean", "median", "most_frequent"], default="mean")
        cs.add_hyperparameter(stratgey)

    def __init__(self):
        '''
            Constructor
        '''
        self.imputer = None

    def fit(self, scenario, config):
        '''
            fit pca object to ASlib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration
        '''

        self.imputer = Imputer(strategy=config.get("imputer_strategy"))
        self.imputer.fit(scenario.feature_data.values)

    def transform(self, scenario):
        '''
            transform ASLib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas

            Returns
            -------
            data.aslib_scenario.ASlibScenario
        '''
        values = self.imputer.transform(
            np.array(scenario.feature_data.values))
        scenario.feature_data = pd.DataFrame(
            data=values, index=scenario.feature_data.index, columns=scenario.feature_data.columns)

        return scenario

    def fit_transform(self, scenario, config):
        '''
            fit and transform

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration

            Returns
            -------
            data.aslib_scenario.ASlibScenario
        '''
        self.fit(scenario, config)
        scenario = self.transform(scenario)
        return scenario
