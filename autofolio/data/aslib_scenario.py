import os
import sys
import logging
import yaml
import functools
import arff  # liac-arff
import copy
import json

from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np

__author__ = "Marius Lindauer"
__version__ = "2.0.0"
__license__ = "BSD"

MAXINT = 2**32


class ASlibScenario(object):
    '''
        all data about an algorithm selection scenario
    '''

    def __init__(self):
        '''
            Constructor
        '''

        self.logger = logging.getLogger("ASlibScenario")

        # listed in description.txt
        self.scenario = None  # string
        self.performance_measure = []  # list of strings
        self.performance_type = []  # list of "runtime" or "solution_quality"
        self.maximize = []  # list of "true" or "false"
        self.algorithm_cutoff_time = None  # float
        self.algorithm_cutoff_memory = None  # integer
        self.features_cutoff_time = None  # float
        self.features_cutoff_memory = None  # integer
        self.features_deterministic = []  # list of strings
        self.features_stochastic = []  # list of strings
        self.algorithms = []  # list of strings
        self.algortihms_deterministics = []  # list of strings
        self.algorithms_stochastic = []  # list of strings
        self.feature_group_dict = {}  # string -> [] of strings
        self.feature_steps = []
        self.feature_steps_default = []

        # extracted in other files
        self.features = []
        self.ground_truths = {}  # type -> [values]

        self.feature_data = None
        self.performance_data = None
        self.runstatus_data = None
        self.feature_cost_data = None
        self.feature_runstatus_data = None
        self.ground_truth_data = None
        self.cv_data = None

        self.instances = None  # list

        self.found_files = []
        self.read_funcs = {
            "description.txt": self.read_description,
            "algorithm_runs.arff": self.read_algorithm_runs,
            "feature_costs.arff": self.read_feature_costs,
            "feature_values.arff": self.read_feature_values,
            "feature_runstatus.arff": self.read_feature_runstatus,
            "ground_truth.arff": self.read_ground_truth,
            "cv.arff": self.read_cv
        }

        self.CHECK_VALID = True

    def __getstate__(self):
        '''
            method for pickling the object;
        '''
        #state_dict = copy.copy(self.__dict__)
        state_dict = self.__dict__

        # adding explicitly the feature names as used before
        state_dict["feature_names"] = list(self.feature_data.columns)

        return state_dict

    def read_from_csv(self, perf_fn: str, feat_fn: str, objective: str, runtime_cutoff: float, maximize: bool):
        '''
            create an internal ASlib scenario from csv

            Arguments
            ---------
            perf_fn: str
                performance file name in csv format
            feat_fn: str
                instance feature file name in csv format
            objective: str
                "solution_quality" or "runtime"
            runtime_cutoff: float
                maximal runtime cutoff
            maximize: bool
                whether to maximize or minimize the objective values
        '''

        self.scenario = None  # string
        self.performance_measure = ["dummy"]  # list of strings
        # list of "runtime" or "solution_quality"
        self.performance_type = [objective]
        self.maximize = [maximize]  # list of "true" or "false"
        self.algorithm_cutoff_time = runtime_cutoff  # float
        self.algorithm_cutoff_memory = None  # integer
        self.features_cutoff_time = None  # float
        self.features_cutoff_memory = None  # integer

        self.feature_data = pd.read_csv(feat_fn, index_col=0)
        self.performance_data = pd.read_csv(perf_fn, index_col=0)

        self.algorithms = list(
            self.performance_data.columns)  # list of strings
        self.algortihms_deterministics = self.algorithms  # list of strings
        self.algorithms_stochastic = []  # list of strings

        self.features_deterministic = list(
            self.feature_data.columns)  # list of strings
        self.features_stochastic = []  # list of strings
        self.feature_group_dict = {
            "all": {"provides": self.features_deterministic}}
        self.feature_steps = ["all"]
        self.feature_steps_default = ["all"]

        self.instances = list(self.feature_data.index)  # lis

        self.runstatus_data = pd.DataFrame(
            data=np.array(
                [["ok"] * len(self.algorithms)] * len(self.instances)),
            index=self.performance_data.index,
            columns=self.performance_data.columns)

        if objective == "runtime":
            self.runstatus_data[
                self.performance_data >= runtime_cutoff] = "timeout"

        self.feature_runstatus_data = pd.DataFrame(
            data=["ok"] * len(self.instances), index=self.instances, columns=["all"])

        self.feature_cost_data = None
        self.ground_truth_data = None

        # extracted in other files
        self.features = self.features_deterministic
        self.ground_truths = {}  # type -> [values]

        self.create_cv_splits()

        if self.CHECK_VALID:
            self.check_data()

    def read_scenario(self, dn):
        '''
            read an ASlib scenario from disk

            Arguments
            ---------
            dn: str
                directory name with ASlib files
        '''
        self.logger.info("Read ASlib scenario: %s" % (dn))

        # add command line arguments in metainfo
        self.dir_ = dn
        self.find_files()
        self.read_files()

        if self.CHECK_VALID:
            self.check_data()

    def find_files(self):
        '''
            find all expected files in self.dir_
            fills self.found_files
        '''
        expected = ["description.txt", "algorithm_runs.arff",
                    "feature_values.arff", "feature_runstatus.arff"]
        optional = ["ground_truth.arff", "feature_costs.arff", "cv.arff"]

        for expected_file in expected:
            full_path = os.path.join(self.dir_, expected_file)
            if not os.path.isfile(full_path):
                self.logger.error("Required file not found: %s" % (full_path))
                sys.exit(2)
            else:
                self.found_files.append(full_path)

        for expected_file in optional:
            full_path = os.path.join(self.dir_, expected_file)
            if not os.path.isfile(full_path):
                self.logger.warn("Optional file not found: %s" % (full_path))
            else:
                self.found_files.append(full_path)

    def read_files(self):
        '''
            iterates over all found files (self.found_files) and 
            calls the corresponding function to validate file
        '''
        for file_ in self.found_files:
            read_func = self.read_funcs.get(os.path.basename(file_))
            if read_func:
                read_func(file_)

    def read_description(self, fn):
        '''
            reads description file
            and saves all meta information
        '''
        self.logger.info("Read %s" % (fn))

        with open(fn, "r") as fh:
            description = yaml.load(fh)

        self.scenario = description.get('scenario_id')
        self.performance_measure = description.get('performance_measures')

        self.performance_measure = description.get('performance_measures') if isinstance(description.get('performance_measures'), list) else \
            [description.get('performance_measures')]

        maximize = description.get('maximize')
        self.maximize = maximize if isinstance(maximize, list) else \
            [maximize]

        performance_type = description.get('performance_type')
        self.performance_type  = performance_type if isinstance(performance_type, list) else \
            [performance_type]

        self.algorithm_cutoff_time = description.get('algorithm_cutoff_time')
        self.features_cutoff_memory = description.get(
            'algorithm_cutoff_memory')
        self.features_cutoff_time = description.get('features_cutoff_time')
        self.features_cutoff_memory = description.get('features_cutoff_memory')
        self.features_deterministic = description.get('features_deterministic')
        if self.features_deterministic is None:
            self.features_deterministic = set()
        self.features_stochastic = description.get('features_stochastic')
        if self.features_stochastic is None:
            self.features_stochastic = set()
        self.algortihms_deterministics = description.get(
            'algorithms_deterministic')
        if self.algortihms_deterministics is None:
            self.algortihms_deterministics = set()
        self.algorithms_stochastic = description.get('algorithms_stochastic')
        if self.algorithms_stochastic is None:
            self.algorithms_stochastic = set()
        self.feature_group_dict = description.get('feature_steps')
        self.feature_steps = list(self.feature_group_dict.keys())
        self.feature_steps_default = description.get('default_steps')

        for step, d in self.feature_group_dict.items():
            if d.get("requires") and not isinstance(d["requires"], list):
                self.feature_group_dict[step]["requires"] = [d["requires"]]

        self.algorithms = sorted(list(
            set(self.algorithms_stochastic).union(
                self.algortihms_deterministics)))

        self.algorithms = sorted(list(
            set(self.algorithms_stochastic).union(self.algortihms_deterministics)))

        if not self.scenario:
            self.logger.warn("Have not found SCENARIO_ID")
        if not self.performance_measure:
            self.logger.warn("Have not found PERFORMANCE_MEASURE")
        if not self.performance_type:
            self.logger.warn("Have not found PERFORMANCE_TYPE")
        if not self.maximize:
            self.logger.warn("Have not found MAXIMIZE")
        if not self.algorithm_cutoff_time:
            self.logger.error("Have not found algorithm_cutoff_time")
            sys.exit(2)
        if not self.algorithm_cutoff_memory:
            self.logger.warn("Have not found algorithm_cutoff_memory")
        if not self.features_cutoff_time:
            self.logger.warn("Have not found features_cutoff_time")
            self.logger.warn(
                "Assumption FEATURES_CUTOFF_TIME == ALGORITHM_CUTOFF_TIME ")
            self.features_cutoff_time = self.algorithm_cutoff_time
        if not self.features_cutoff_memory:
            self.logger.warn("Have not found features_cutoff_memory")
        if not self.features_deterministic:
            self.logger.warn("Have not found features_deterministic")
        if not self.features_stochastic:
            self.logger.warn("Have not found features_stochastic")
        if not self.algortihms_deterministics:
            self.logger.warn("Have not found algortihms_deterministics")
        if not self.algorithms_stochastic:
            self.logger.warn("Have not found algorithms_stochastic")
        if not self.feature_group_dict:
            self.logger.warn("Have not found any feature step")

        feature_intersec = set(self.features_deterministic).intersection(
            self.features_stochastic)
        if feature_intersec:
            self.logger.warn("Intersection of deterministic and stochastic features is not empty: %s" % (
                str(feature_intersec)))
        algo_intersec = set(self.algortihms_deterministics).intersection(
            self.algorithms_stochastic)
        if algo_intersec:
            self.logger.warn(
                "Intersection of deterministic and stochastic algorithms is not empty: %s" % (str(algo_intersec)))

    def read_algorithm_runs(self, fn):
        '''
            read performance file
            and saves information
            add Instance() in self.instances

            unsuccessful runs are replaced by algorithm_cutoff_time if performance_type is runtime

            EXPECTED HEADER:
            @RELATION ALGORITHM_RUNS_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE algorithm STRING
            @ATTRIBUTE PAR10 NUMERIC
            @ATTRIBUTE Number_of_satisfied_clauses NUMERIC
            @ATTRIBUTE runstatus {ok, timeout, memout, not_applicable, crash, other}
        '''
        self.logger.info("Read %s" % (fn))

        with open(fn, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (file_))

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (file_))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "repetition as second attribute is missing in %s" % (file_))
            sys.exit(3)
        if arff_dict["attributes"][2][0].upper() != "ALGORITHM":
            self.logger.error(
                "algorithm as third attribute is missing in %s" % (file_))
            sys.exit(3)

        i = 0
        for performance_measure in self.performance_measure:
            if arff_dict["attributes"][3 + i][0].upper() != performance_measure.upper():
                self.logger.error(
                    "\"%s\" as attribute is missing in %s" % (performance_measure, file_))
                sys.exit(3)
            i += 1

        if arff_dict["attributes"][3 + i][0].upper() != "RUNSTATUS":
            self.logger.error(
                "runstatus as last attribute is missing in %s" % (file_))
            sys.exit(3)

        pairs_inst_rep_alg = []

        algo_inst_perf = {}

        self.performance_data = pd.DataFrame(
            np.array(arff_dict["data"])[:,:4], columns=["instance_id", "repetition", "algorithm", "score"]) # take only the first 4 columns
        self.performance_data.drop("repetition", axis=1) 
        self.performance_data = self.performance_data.pivot("instance_id", "algorithm", "score")
        self.performance_data = self.performance_data.apply(lambda x: pd.to_numeric(x))
        
        self.runstatus_data = pd.DataFrame(
            np.array(arff_dict["data"])[:,[0,1,2,-1]], columns=["instance_id", "repetition", "algorithm", "runstatus"]) # take only the first 4 columns
        self.runstatus_data.drop("repetition", axis=1) 
        self.runstatus_data = self.runstatus_data.pivot("instance_id", "algorithm", "runstatus")
        
        self.instances = list(self.performance_data.index)

    def read_feature_values(self, file_):
        '''
            reads feature file
            and saves them in self.instances

            Expected Header:
            @RELATION FEATURE_VALUES_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE number_of_variables NUMERIC
            @ATTRIBUTE number_of_clauses NUMERIC
            @ATTRIBUTE first_local_min_steps NUMERIC
        '''

        self.logger.info("Read %s" % (file_))

        with open(file_, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (file_))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (file_))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "repetition as second attribute is missing in %s" % (file_))
            sys.exit(3)

        feature_set = set(self.features_deterministic).union(
            self.features_stochastic)

        for f_name in arff_dict["attributes"][2:]:
            f_name = f_name[0]
            self.features.append(f_name)
            if not f_name in feature_set:
                self.logger.error(
                    "Feature \"%s\" was not defined as deterministic or stochastic" % (f_name))
                sys.exit(3)

        pairs_inst_rep = []
        encoutered_features = []
        inst_feats = {}
        for data in arff_dict["data"]:
            inst_name = data[0]
            repetition = data[1]
            features = data[2:]

            if len(features) != len(self.features):
                self.logger.error(
                    "Number of features in attributes does not match number of found features; instance: %s" % (inst_name))
                sys.exit(3)

            # TODO: handle feature repetitions
            inst_feats[inst_name] = features

            # not only Nones in feature vector and previously seen
            if functools.reduce(lambda x, y: True if (x or y) else False, features, False) and features in encoutered_features:
                self.logger.warn(
                    "Feature vector found twice: %s" % (",".join(map(str, features))))
            else:
                encoutered_features.append(features)

            if (inst_name, repetition) in pairs_inst_rep:
                self.logger.warn(
                    "Pair (%s,%s) is not unique in %s" % (inst_name, repetition, file_))
            else:
                pairs_inst_rep.append((inst_name, repetition))

        # convert to pandas
        data = np.array(arff_dict["data"])
        cols = list(map(lambda x: x[0], arff_dict["attributes"][2:]))
        self.feature_data = pd.DataFrame(
            data[:, 2:], index=data[:, 0], columns=cols, dtype=np.float)

    def read_feature_costs(self, file_):
        '''
            reads feature time file
            and saves in self.instances

            Expected header:
            @RELATION FEATURE_COSTS_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE preprocessing NUMERIC
            @ATTRIBUTE local_search_probing NUMERIC

        '''
        self.logger.info("Read %s" % (file_))

        with open(file_, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (file_))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "\"instance_id\" as first attribute is missing in %s" % (file_))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "\"repetition\" as second attribute is missing in %s" % (file_))
            sys.exit(3)
        found_groups = list(
            map(str, sorted(map(lambda x: x[0], arff_dict["attributes"][2:]))))
        for meta_group in self.feature_group_dict.keys():
            if meta_group not in found_groups:
                self.logger.error(
                    "\"%s\" as attribute is missing in %s" % (meta_group, file_))
                sys.exit(3)

        inst_cost = {}

        # convert to pandas
        data = np.array(arff_dict["data"])
        cols = list(map(lambda x: x[0], arff_dict["attributes"][2:]))
        self.feature_cost_data = pd.DataFrame(
            data[:, 2:], index=data[:, 0], columns=cols, dtype=np.float)

        self.feature_cost_data[pd.isnull(self.feature_cost_data)] = 0

    def read_feature_runstatus(self, file_):
        '''
            reads run stati of all pairs instance x feature step
            and saves them self.instances

            Expected header:
            @RELATION FEATURE_RUNSTATUS_2013 - SAT - Competition
            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE preprocessing { ok , timeout , memout , presolved , crash , other }
            @ATTRIBUTE local_search_probing { ok , timeout , memout , presolved , crash , other }
        '''
        self.logger.info("Read %s" % (file_))

        with open(file_, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (file_))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (file_))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "repetition as second attribute is missing in %s" % (file_))
            sys.exit(3)

        for f_name in arff_dict["attributes"][2:]:
            f_name = f_name[0]
            if not f_name in self.feature_group_dict.keys():
                self.logger.error(
                    "Feature step \"%s\" was not defined in feature steps" % (f_name))
                sys.exit(3)

        if len(self.feature_group_dict.keys()) != len(arff_dict["attributes"][2:]):
            self.logger.error("Number of feature steps in description.txt (%d) and feature_runstatus.arff (%d) does not match." % (
                len(self.feature_group_dict.keys()), len(arff_dict["attributes"][2:-1])))
            sys.exit(3)

        # convert to pandas
        data = np.array(arff_dict["data"])
        cols = list(map(lambda x: x[0], arff_dict["attributes"][2:]))
        self.feature_runstatus_data = pd.DataFrame(
            data[:, 2:], index=data[:, 0], columns=cols)

    def read_ground_truth(self, file_):
        '''
            read ground truths of all instances
            and save them in self.instances

            @RELATION GROUND_TRUTH_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE SATUNSAT {SAT,UNSAT}
            @ATTRIBUTE OPTIMAL_VALUE NUMERIC
        '''

        self.logger.info("Read %s" % (file_))

        with open(file_, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (file_))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (file_))
            sys.exit(3)

        # extract feature names
        for attr in arff_dict["attributes"][1:]:
            self.ground_truths[attr[0]] = attr[1]

        # convert to panda
        data = np.array(arff_dict["data"])
        cols = list(map(lambda x: x[0], arff_dict["attributes"][1:]))
        self.ground_truth_data = pd.DataFrame(
            data=data[:, 1:], index=data[:, 0].tolist(), columns=cols)

    def read_cv(self, file_):
        '''
            read cross validation <file_>

            @RELATION CV_2013 - SAT - Competition
            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE fold NUMERIC
        '''
        self.logger.info("Read %s" % (file_))

        with open(file_, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (file_))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (file_))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "repetition as second attribute is missing in %s" % (file_))
            sys.exit(3)
        if arff_dict["attributes"][2][0].upper() != "FOLD":
            self.logger.error(
                "fold as third attribute is missing in %s" % (file_))
            sys.exit(3)

        # convert to pandas
        data = np.array(arff_dict["data"])
        cols = list(map(lambda x: x[0], arff_dict["attributes"][2:]))
        self.cv_data = pd.DataFrame(
            data[:, 2:], index=data[:, 0], columns=cols, dtype=np.float)

    def check_data(self):
        '''
            checks whether all data objects are valid according to ASlib specification
            and makes some transformations
        '''

        if self.performance_type[0] == "runtime" and self.maximize[0]:
            self.logger.error("Maximizing runtime is not supported")
            sys.exit(3)

        if self.performance_type[0] == "runtime":
            # replace all non-ok scores with par10 values
            self.logger.debug(
                "Replace all runtime data with PAR10 values for non-OK runs")
            self.performance_data[
                self.runstatus_data != "ok"] = self.algorithm_cutoff_time * 10

        if self.performance_type[0] == "solution_quality" and self.maximize[0]:
            self.logger.info(
                "Multiply all performance data by -1, since autofolio minimizes the scores but the objective is to maximize")
            self.performance_data *= -1

        all_data = [self.feature_data, self.feature_cost_data,
                    self.performance_data, self.feature_runstatus_data,
                    self.ground_truth_data, self.cv_data]

        # all data should have the same instances
        set_insts = set(self.instances)
        for data in all_data:
            if data is not None and set_insts.difference(data.index):
                self.logger.error("Not all data matrices have the same instances: %s" % (
                    set_insts.difference(data.index)))
                sys.exit(3)

            # each instance should be listed only once
            if data is not None and len(list(set(data.index))) != len(data.index):
                self.logger.error("Some instances are listed more than once")
                sys.exit(3)

    def get_split(self, indx=1):
        '''
            returns a copy of self but only with the data of the i-th cross validation split according to cv.arff

            Arguments
            ---------
                indx : int
                    indx of the cv split (should be in most cases within [1,10]

            Returns
            -------
                training split : ASlibScenario
                test split : ASlibScenario
        '''

        if self.cv_data is None:
            self.logger.warn(
                "The ASlib scenario has not provided any cv.arff; create CV split...")
            self.create_cv_splits()

        test_insts = self.cv_data[
            self.cv_data["fold"] == float(indx)].index.tolist()
        training_insts = self.cv_data[
            self.cv_data.fold != float(indx)].index.tolist()

        test = copy.copy(self)
        training = copy.copy(self)

        # feature_data
        test.feature_data = test.feature_data.drop(training_insts).sort_index()
        training.feature_data = training.feature_data.drop(
            test_insts).sort_index()
        # performance_data
        test.performance_data = test.performance_data.drop(
            training_insts).sort_index()
        training.performance_data = training.performance_data.drop(
            test_insts).sort_index()
        # runstatus_data
        test.runstatus_data = test.runstatus_data.drop(
            training_insts).sort_index()
        training.runstatus_data = training.runstatus_data.drop(
            test_insts).sort_index()
        # self.feature_runstatus_data
        test.feature_runstatus_data = test.feature_runstatus_data.drop(
            training_insts).sort_index()
        training.feature_runstatus_data = training.feature_runstatus_data.drop(
            test_insts).sort_index()
        # feature_cost_data
        if self.feature_cost_data is not None:
            test.feature_cost_data = test.feature_cost_data.drop(
                training_insts).sort_index()
            training.feature_cost_data = training.feature_cost_data.drop(
                test_insts).sort_index()
        # ground_truth_data
        if self.ground_truth_data is not None:
            test.ground_truth_data = test.ground_truth_data.drop(
                training_insts).sort_index()
            training.ground_truth_data = training.ground_truth_data.drop(
                test_insts).sort_index()
        test.cv_data = None
        training.cv_data = None

        test.instances = test_insts
        training.instances = training_insts

        self.used_feature_groups = None

        return test, training

    def create_cv_splits(self, n_folds: int=10):
        '''
            creates cv splits and saves them in self.cv_data

            Argumnents
            ----------
            n_folds: int
                number of splits
        '''

        kf = KFold(len(self.instances), n_folds=n_folds, shuffle=True)
        self.cv_data = pd.DataFrame(
            data=np.zeros(len(self.instances)), index=self.instances, columns=["fold"], dtype=np.float)

        for indx, (train, test) in enumerate(kf):
            # print(self.cv_data.loc(np.array(self.instances[test]).tolist()))
            self.cv_data.iloc[test] = indx + 1.
