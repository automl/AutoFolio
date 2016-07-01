import os
import sys
import logging
import math

import numpy as np
import pandas as pd
import subprocess

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from autofolio.data.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class Aspeed(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace, cutoff: int):
        '''
            adds parameters to ConfigurationSpace

            Arguments
            ---------
            cs: ConfigurationSpace
                configuration space to add new parameters and conditions
            cutoff: int
                maximal possible time for aspeed
        '''

        pre_solving = CategoricalHyperparameter(
            "presolving", choices=[True, False], default=True)
        cs.add_hyperparameter(pre_solving)
        pre_cutoff = UniformIntegerHyperparameter(
            "pre:cutoff", lower=0, upper=cutoff, default=math.ceil(cutoff*0.1))
        cs.add_hyperparameter(pre_cutoff)
        cond = InCondition(child=pre_cutoff, parent=pre_solving, values=[True])
        cs.add_condition(cond)

    def __init__(self, clingo: str=None, runsolver: str=None, enc_fn: str=None):
        '''
            Constructor

            Arguments
            ---------
            clingo: str
                path to clingo binary
            runsolver: str
                path to runsolver binary
            enc_fn: str
                path to encoding file name
        '''
        self.logger = logging.getLogger("Aspeed")

        if not runsolver:
            self.runsolver = os.path.join(
                os.path.dirname(sys.argv[0]), "..", "aspeed", "runsolver")
        else:
            self.runsolver = runsolver
        if not clingo:
            self.clingo = os.path.join(
                os.path.dirname(sys.argv[0]), "..", "aspeed", "clingo")
        else:
            self.clingo = clingo
        if not enc_fn:
            self.enc_fn = os.path.join(
                os.path.dirname(sys.argv[0]), "..", "aspeed", "enc1.lp")
        else:
            self.enc_fn = enc_fn

        self.mem_limit = 2000  # mb
        self.cutoff = 60

        self.data_threshold = 300  # number of instances

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

        if config["presolving"]:
            self.logger.info("Compute Presolving Schedule with Aspeed")

            X = scenario.performance_data.values

            # if the instance set is too large, we subsample it
            if X.shape[0] > self.data_threshold:
                random_indx = np.random.choice(
                    range(X.shape[0]), size=self.data_threshold, replace=True)
                X = X[random_indx, :]

            #TODO: MISSING A????
            times = ["time(i%d, a%d, %d)." % (i, a, math.ceil(X[i, a]))
                     for i in range(X.shape[0]) for j in range(X.shape[1])]

            kappa = "kappa(%d)." % (config["pre:cutoff"])

            data_in = " ".join(times) + " " + kappa

            # call aspeed and save schedule
            self._call_clingo(data_in)

    def _call_clingo(self, data_in: str):
        '''
            call clingo on self.enc_fn and facts from data_in

            Arguments
            ---------
            data_in: str
                facts in format time(I,A,T) and kappa(C)
        '''
        cmd = "%s -C %d -M %d -w /dev/null %s %s -" % (
            self.runsolver, self.cutoff, self.mem_limit, self.clingo, self.enc_fn)

        self.logger.info("Call: %s" % (cmd))

        p = subprocess.Popen(cmd,
                             stdin=subprocess.PIPE, stdout=PIPE, shell=True)
        stdout, stderr = p.communicate(input=data_in)
        self.logger.info(stdout.decode())

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
