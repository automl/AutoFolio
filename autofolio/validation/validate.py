import logging

from autofolio.data.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class Stats(object):

    def __init__(self, runtime_cutoff):
        ''' Constructor 

            Arguments
            ---------
            runtime_cutoff: int
                maximal running time
        '''
        self.par1 = 0.0
        self.par10 = 0.0
        self.timeouts = 0
        self.solved = 0
        self.unsolvable = 0
        self.presolved_feats = 0

        self.runtime_cutoff = runtime_cutoff

        self.logger = logging.getLogger("Stats")

    def show(self, remove_unsolvable: bool=True):
        '''
            shows statistics

            Arguments
            --------
            remove_unsolvable : bool
                remove unsolvable from stats
                
            Returns
            -------
            par10: int
                penalized average runtime 
        '''

        if remove_unsolvable:
            rm_string = "removed"
            timeouts = self.timeouts - self.unsolvable
            par1 = self.par1 - (self.unsolvable * self.runtime_cutoff)
            par10 = self.par10 - (self.unsolvable * self.runtime_cutoff * 10)
        else:
            rm_string = "not removed"
            timeouts = self.timeouts
            par1 = self.par1
            par10 = self.par10

        n_samples = timeouts + self.solved
        self.logger.info("PAR1: %.4f" % (par1 / n_samples))
        self.logger.info("PAR10: %.4f" % (par10 / n_samples))
        self.logger.info("Timeouts: %d / %d" % (timeouts, n_samples))
        self.logger.info("Presolved during feature computation: %d / %d" %(self.presolved_feats, n_samples))
        self.logger.info("Solved: %d / %d" % (self.solved, n_samples))
        self.logger.info("Unsolvable (%s): %d / %d" %
                         (rm_string, self.unsolvable, n_samples))
        
        return par10 / n_samples

    def merge(self, stat):
        '''
            adds stats from another given Stats objects

            Arguments
            ---------
            stat : Stats
        '''
        self.par1 += stat.par1
        self.par10 += stat.par10
        self.timeouts += stat.timeouts
        self.solved += stat.solved
        self.unsolvable += stat.unsolvable
        self.presolved_feats += stat.presolved_feats


class Validator(object):

    def __init__(self):
        ''' Constructor '''
        self.logger = logging.getLogger("Validation")

    def validate_runtime(self, schedules: dict, test_scenario: ASlibScenario):
        '''
            validate selected schedules on test instances

            Arguments
            ---------
            schedules: dict {instance name -> tuples [algo, bugdet]}
                algorithm schedules per instance
            test_scenario: ASlibScenario
                ASlib scenario with test instances
        '''
        stat = Stats(runtime_cutoff=test_scenario.algorithm_cutoff_time)

        feature_times = False
        if test_scenario.feature_cost_data is not None and test_scenario.performance_type[0] == "runtime":
            f_times = test_scenario.feature_cost_data[
                test_scenario.used_feature_groups].sum(axis=1)
            feature_times = True

        feature_stati = test_scenario.feature_runstatus_data[
            test_scenario.used_feature_groups]

        ok_status = test_scenario.runstatus_data == "ok"
        unsolvable = ok_status.sum(axis=1) == 0
        stat.unsolvable += unsolvable.sum()

        for inst, schedule in schedules.items():
            self.logger.debug("Validate %s on %s" %(schedule, inst))
            used_time = 0
            if feature_times:
                used_time += f_times[inst]
                self.logger.debug("Used Feature time: %f" % (used_time))

            presolved = False
            for fg in test_scenario.used_feature_groups:
                if "presolved" in feature_stati[fg][inst]:
                    presolved = True
                    break

            if presolved and used_time < test_scenario.algorithm_cutoff_time:
                stat.par1 += used_time
                stat.solved += 1
                stat.presolved_feats += 1
                self.logger.debug("Presolved during feature computation")
                continue
            elif presolved and used_time >= test_scenario.algorithm_cutoff_time:
                stat.par1 += test_scenario.algorithm_cutoff_time
                stat.timeouts += 1
                continue

            for algo, budget in schedule:
                time = test_scenario.performance_data[algo][inst]
                used_time += min(time, budget)
                if time <= budget and used_time <= test_scenario.algorithm_cutoff_time and test_scenario.runstatus_data[algo][inst] == "ok":
                    stat.par1 += used_time
                    stat.solved += 1
                    self.logger.debug("Solved by %s (budget: %f -- required to solve: %f)" %(algo, budget, time))
                    break

                if used_time > test_scenario.algorithm_cutoff_time:
                    stat.par1 += test_scenario.algorithm_cutoff_time
                    stat.timeouts += 1
                    self.logger.debug("Timeout after %d" %(used_time))
                    break

        if test_scenario.performance_type[0] == "runtime":
            stat.par10 = stat.par1 + 9 * \
                test_scenario.algorithm_cutoff_time * stat.timeouts
        stat.show()

        return stat
