import os
import sys
import pandas as pd
import logging
import yaml
import arff #liac-arff

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
        self.scenario = None # string
        self.performance_measure = [] #list of strings
        self.performance_type = [] # list of "runtime" or "solution_quality"
        self.maximize = [] # list of "true" or "false"
        self.algorithm_cutoff_time = None # float
        self.algorithm_cutoff_memory = None # integer
        self.features_cutoff_time = None # float
        self.features_cutoff_memory = None # integer
        self.features_deterministic = [] # list of strings
        self.features_stochastic = [] # list of strings
        self.algorithms = [] # list of strings
        self.algortihms_deterministics = [] # list of strings
        self.algorithms_stochastic = [] # list of strings
        self.feature_group_dict = {} # string -> [] of strings
        self.feature_steps = []

        # extracted in other files
        self.features = []
        self.ground_truths = {} # type -> [values]
        self.cv_given = False
        
        self.feature_data = None
        self.performance_data = None
        
        self.instances = None # list
        
        self.found_files = []
        self.read_funcs = {                        
                       "description.txt" : self.read_description,
                       "algorithm_runs.arff" : self.read_algorithm_runs,
                       #"feature_costs.arff": self.read_feature_costs,
                       #"feature_values.arff": self.read_feature_values,
                       #"feature_runstatus.arff": self.read_feature_runstatus,
                       #"ground_truth.arff": self.read_ground_truth,
                       #"cv.arff": self.read_cv                   
                }
        
    def read_scenario(self, dn):
        '''
            read an ASlib scenario from disk
            
            Arguments
            ---------
            dn: str
                directory name with ASlib files
        '''
        logging.info("Read ASlib scenario: %s" %(dn))
        
        #add command line arguments in metainfo
        self.dir_ = dn
        self.find_files()
        self.read_files()
        
    def find_files(self):
        '''
            find all expected files in self.dir_
            fills self.found_files
        '''
        expected = ["description.txt", "algorithm_runs.arff", "feature_values.arff", "feature_runstatus.arff"]
        optional = ["ground_truth.arff", "feature_costs.arff", "citation.bib", "cv.arff"]
        
        expected = ["description.txt", "algorithm_runs.arff"]
        optional = []
        
        for expected_file in expected:
            full_path = os.path.join(self.dir_,expected_file)
            if not os.path.isfile(full_path):
                self.logger.error("Required file not found: %s" %(full_path))
                sys.exit(2)
            else:
                self.found_files.append(full_path)
                
        for expected_file in optional:
            full_path = os.path.join(self.dir_,expected_file)
            if not os.path.isfile(full_path):
                self.logger.warn("Optional file not found: %s" %(full_path))
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
        self.logger.info("Read %s" %(fn))
        
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
        self.features_cutoff_memory = description.get('algorithm_cutoff_memory')
        self.features_cutoff_time = description.get('features_cutoff_time')
        self.features_cutoff_memory = description.get('features_cutoff_memory')
        self.features_deterministic = description.get('features_deterministic')
        if self.features_deterministic is None:
            self.features_deterministic = set()
        self.features_stochastic = description.get('features_stochastic')
        if self.features_stochastic is None:
            self.features_stochastic = set()
        self.algortihms_deterministics = description.get('algorithms_deterministic')
        if self.algortihms_deterministics is None:
            self.algortihms_deterministics = set()
        self.algorithms_stochastic = description.get('algorithms_stochastic')
        if self.algorithms_stochastic is None:
            self.algorithms_stochastic = set()
        self.feature_group_dict = description.get('feature_steps')
        self.feature_steps = description.get('default_steps')
        
        for step, d in self.feature_group_dict.items():
            if  d.get("requires") and not isinstance(d["requires"], list):
                self.feature_group_dict[step]["requires"] = [d["requires"]]

        self.algorithms = list(
            set(self.algorithms_stochastic).union(
                self.algortihms_deterministics))
        
        self.algorithms = list(set(self.algorithms_stochastic).union(self.algortihms_deterministics))
                  
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
            self.logger.warn("Assumption FEATURES_CUTOFF_TIME == ALGORITHM_CUTOFF_TIME ")
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
           
        feature_intersec = set(self.features_deterministic).intersection(self.features_stochastic)
        if feature_intersec:
            self.logger.warn("Intersection of deterministic and stochastic features is not empty: %s" %(str(feature_intersec)))
        algo_intersec = set(self.algortihms_deterministics).intersection(self.algorithms_stochastic)
        if algo_intersec:
            self.logger.warn("Intersection of deterministic and stochastic algorithms is not empty: %s" %(str(algo_intersec)))
        
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
        self.logger.info("Read %s" %(fn))
        
        with open(fn,"r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error("Parsing of arff file failed (%s) - maybe conflict of header and data." %(file_))
        
        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error("instance_id as first attribute is missing in %s" %(file_))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error("repetition as second attribute is missing in %s" %(file_))    
            sys.exit(3)
        if arff_dict["attributes"][2][0].upper() != "ALGORITHM":
            self.logger.error("algorithm as third attribute is missing in %s" %(file_))
            sys.exit(3)
            
        i = 0
        for performance_measure in self.performance_measure:                       
            if arff_dict["attributes"][3 + i][0].upper() != performance_measure.upper():
                self.logger.error("\"%s\" as attribute is missing in %s" %(performance_measure, file_))
                sys.exit(3)
            i += 1
        
        if arff_dict["attributes"][3 + i][0].upper() != "RUNSTATUS":
            self.logger.error("runstatus as last attribute is missing in %s" %(file_))
            sys.exit(3)
               
        pairs_inst_rep_alg = []
        
        algo_inst_perf = {}
        
        self.instances = set()        
        
        for data in arff_dict["data"]:
            inst_name = str(data[0])
            repetition = data[1]
            algorithm = str(data[2])
            perf_list = data[3:-1]
            status = data[-1]
            
            self.instances.add(inst_name)
            
            for p_measure, p_type, max_bool, perf in zip(self.performance_measure, self.performance_type, self.maximize, perf_list):       
                if perf is None:
                    self.logger.warn("The following performance data has missing values.\n"+ 
                                    "%s" % (",".join(map(str,data))))
                    perf = MAXINT
                    
                if max_bool:
                    perf *= -1 # we always minimize
                    
                algo_inst_perf[algorithm] = algo_inst_perf.get(algorithm,{})
                algo_inst_perf[algorithm][inst_name] = (perf,status)
                
                break #TODO: we consider only the first performance value right now 
            
            
            if (inst_name,repetition, algorithm) in pairs_inst_rep_alg:
                self.logger.warn("Pair (%s,%s,%s) is not unique in %s" %(inst_name, repetition, algorithm, file_))
            else:
                pairs_inst_rep_alg.append((inst_name,repetition, algorithm))
        
        # convert to panda
        
        perf_array = []
        status_array = []
        self.instances = list(self.instances)
        for inst in self.instances:
            perf_vec = []
            status_vec = []
            for algo in self.algorithms:
                perf_vec.append(algo_inst_perf[algo][inst][0])
                status_vec.append(algo_inst_perf[algo][inst][1])
            perf_array.append(perf_vec)
            status_array.append(status_vec)
        
        self.performance_data = pd.DataFrame(perf_array, index=self.instances, columns=self.algorithms)
        self.status_data = pd.DataFrame(status_array, index=self.instances, columns=self.algorithms)
