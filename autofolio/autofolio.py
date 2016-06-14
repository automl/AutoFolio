import logging

from ConfigSpace.configuration_space import Configuration, \
    ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from autofolio.io.cmd import CMDParser
from autofolio.data.aslib_scenario import ASlibScenario

# feature preprocessing
from autofolio.feature_preprocessing.pca import PCAWrapper
from autofolio.feature_preprocessing.missing_values import ImputerWrapper

#classifiers
from autofolio.selector.classifiers.random_forest import RandomForest

__author__= "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.0.0"

class AutoFolio(object):
    
    def __init__(self):
        ''' Constructor '''

        self._root_logger = logging.getLogger()
        self.logger = logging.getLogger("AutoFolio")
        
        
    def run(self):
        '''
            main method of AutoFolio
        '''
        
        cmd_parser = CMDParser()
        args_ = cmd_parser.parse()

        self._root_logger.setLevel(args_.verbose)
        
        scenario = ASlibScenario()
        scenario.read_scenario(args_.scenario)
        
        cs = self.get_cs()
        
        self.feature_preprocessing(scenario, cs.get_default_configuration())
        
    def get_cs(self):
        '''
            returns the parameter configuration space of AutoFolio
            (based on the automl config space: https://github.com/automl/ConfigSpace)
        '''
        
        self.cs = ConfigurationSpace()

        PCAWrapper.add_params(self.cs)
        ImputerWrapper.add_params(self.cs)
        
        #classifiers
        RandomForest.add_params(self.cs)
        
        return self.cs
        
    def feature_preprocessing(self, scenario:ASlibScenario, config:Configuration):
        '''
            performs feature preprocessing on a given ASlib scenario wrt to a given configuration
        '''
        
        imputer = ImputerWrapper()
        scenario = imputer.fit_transform(scenario,config)
        
        pca = PCAWrapper()
        scenario = pca.fit_transform(scenario, config)
        
    def fit_selector(self, scenario:ASlibScenario, config:Configuration):
        '''
            fits an algorithm selector for a given scenario wrt a given configuration
        '''
        
        
        
        
        
        
        
        