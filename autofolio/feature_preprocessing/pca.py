import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

__author__ = "Marius Lindauer"
__license__ = "BSD"


class PCAWrapper(object):

    @staticmethod
    def add_params(cs):
        '''
            adds parameters to ConfigurationSpace 
        '''
        pca_switch = CategoricalHyperparameter(
            "pca", choices=[True, False], default=True)
        n_components = UniformIntegerHyperparameter(
            "pca_n_components", lower=1, upper=20, default=7, log=True)
        cs.add_hyperparameter(pca_switch)
        cs.add_hyperparameter(n_components)
        cond = InCondition(child=n_components, parent=pca_switch, values=[True])
        cs.add_condition(cond)

    def __init__(self):
        '''
            Constructor
        '''
        self.pca = None

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

        if config.get("pca"):
            self.pca = PCA(n_components=config.get("pca_n_components"))
            self.pca.fit(scenario.feature_data.values)

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
        if self.pca:
            values = self.pca.transform(
                np.array(scenario.feature_data.values))
            
            scenario.feature_data = pd.DataFrame(
            data=values, index=scenario.feature_data.index, columns=["f%d" %(i) for i in range(self.pca.n_components_)])
            
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
        self.fit(scenario,config)
        scenario = self.transform(scenario)
        return scenario
