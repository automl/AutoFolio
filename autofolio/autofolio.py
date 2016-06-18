import logging

import numpy as np

from ConfigSpace.configuration_space import Configuration, \
    ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from autofolio.io.cmd import CMDParser
from autofolio.data.aslib_scenario import ASlibScenario

# feature preprocessing
from autofolio.feature_preprocessing.pca import PCAWrapper
from autofolio.feature_preprocessing.missing_values import ImputerWrapper
from autofolio.feature_preprocessing.feature_group_filtering import FeatureGroupFiltering

# classifiers
from autofolio.selector.classifiers.random_forest import RandomForest

# selectors
from autofolio.selector.pairwise_classification import PairwiseClassifier

# validation
from autofolio.validation.validate import Validator, Stats

__author__ = "Marius Lindauer"
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

        cs = self.get_cs(scenario)

        config = cs.get_default_configuration()

        self.run_cv(scenario, config, folds=10)

    def get_cs(self, scenario: ASlibScenario):
        '''
            returns the parameter configuration space of AutoFolio
            (based on the automl config space: https://github.com/automl/ConfigSpace)

            Arguments
            ---------
            scenario: autofolio.data.aslib_scenario.ASlibScenario
                aslib scenario at hand
        '''

        self.cs = ConfigurationSpace()

        # add feature steps as binary parameters
        for fs in scenario.feature_steps:
            fs_param = CategoricalHyperparameter(name="fgroup_%s" % (
                fs), choices=[True, False], default=fs in scenario.feature_steps_default)
            self.cs.add_hyperparameter(fs_param)

        # preprocessing
        PCAWrapper.add_params(self.cs)
        ImputerWrapper.add_params(self.cs)

        # classifiers
        RandomForest.add_params(self.cs)

        # selectors
        PairwiseClassifier.add_params(self.cs)

        return self.cs

    def run_cv(self, scenario: ASlibScenario, config: Configuration, folds=10):
        '''
            run a cross fold validation based on the given data from cv.arff

            Arguments
            ---------
            scenario: autofolio.data.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration to use for preprocessing
            folds: int
                number of cv-splits
        '''

        cv_stat = Stats(runtime_cutoff=scenario.algorithm_cutoff_time)
        for i in range(1, folds + 1):
            self.logger.info("CV-Iteration: %d" % (i))
            test_scenario, training_scenario = scenario.get_split(indx=i)

            feature_pre_pipeline, selector = self.fit(
                scenario=training_scenario, config=config)

            schedules = self.predict(
                test_scenario, config, feature_pre_pipeline, selector)

            val = Validator()
            stats = val.validate_runtime(
                schedules=schedules, test_scenario=test_scenario)
            cv_stat.merge(stat=stats)

        self.logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.logger.info("CV Stats")
        cv_stat.show()

    def fit(self, scenario: ASlibScenario, config: Configuration):
        '''
            fit AutoFolio on given ASlib Scenario

            Arguments
            ---------
            scenario: autofolio.data.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration to use for preprocessing

            Returns
            -------
                list of fitted feature preproccessing objects
                fitted selector
        '''

        scenario, feature_pre_pipeline = self.feature_preprocessing(
            scenario, config)

        selector = self.fit_selector(scenario, config)

        return feature_pre_pipeline, selector

    def feature_preprocessing(self, scenario: ASlibScenario, config: Configuration):
        '''
            performs feature preprocessing on a given ASlib scenario wrt to a given configuration

            Arguments
            ---------
            scenario: autofolio.data.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration to use for preprocessing

            Returns
            -------
                list of fitted feature preproccessing objects
        '''

        pipeline = []
        fgf = FeatureGroupFiltering()
        scenario = fgf.fit_transform(scenario, config)

        imputer = ImputerWrapper()
        scenario = imputer.fit_transform(scenario, config)

        pca = PCAWrapper()
        scenario = pca.fit_transform(scenario, config)

        return scenario, [fgf, imputer, pca]

    def fit_selector(self, scenario: ASlibScenario, config: Configuration):
        '''
            fits an algorithm selector for a given scenario wrt a given configuration

            Arguments
            ---------
            scenario: autofolio.data.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration
        '''

        if config.get("selector") == "PairwiseClassifier":

            clf_class = None
            if config.get("classifier") == "RandomForest":
                clf_class = RandomForest

            selector = PairwiseClassifier(classifier_class=clf_class)
            selector.fit(scenario=scenario, config=config)

        return selector

    def predict(self, scenario: ASlibScenario, config, feature_pre_pipeline, selector):
        '''
            predicts algorithm schedules wrt a given config
            and given pipelines

            Arguments
            ---------
            scenario: autofolio.data.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration
            feature_pre_pipeline: list
                list of fitted feature preprocessors
            selector: autofolio.selector.*
                fitted selector object
        '''

        self.logger.info("Predict on Test")
        for f_pre in feature_pre_pipeline:
            scenario = f_pre.transform(scenario)

        schedules = selector.predict(scenario=scenario)

        return schedules
