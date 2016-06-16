import logging

from autofolio.data.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"

class Stats(object):

    def __init__(self):
        ''' Constructor '''
        self.par1 = 0
        self.par10 = 0
        self.timeouts = 0

class Validator(object):
    
    def __init__(self):
        ''' Constructor '''
        
    def validate_runtime(self, schedules:dict, test_scenario:ASlibScenario):
        '''
            validate selected schedules on test instances
            
            Arguments
            ---------
            schedules: dict {instance name -> tuples [algo, bugdet]}
                algorithm schedules per instance
            test_scenario: ASlibScenario
                ASlib scenario with test instances
        '''
        stat = Stats()
        
        for inst, schedule in schedules.items():
            perf_test = test_scenario.performance_data.loc(inst)
            print(perf_test[schedule[0]])
            
            if test_scenario.feature_cost_data is not None:
                used_time = test_scenario.feature_cost_data[test_scenario.used_feature_groups]
            else:
                used_time = 0
                
            print(used_time)
            
        return stat
            
            