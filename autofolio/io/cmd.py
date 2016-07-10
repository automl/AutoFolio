import argparse
import sys
import os
import logging

__author__ = "Marius Lindauer"
__version__ = "2.0.0"
__license__ = "BSD"


class CMDParser(object):

    def __init__(self):
        '''
            Constructor
        '''
        self.logger = logging.getLogger("CMDParser")

        self._arg_parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        aslib = self._arg_parser.add_argument_group(
            "Reading from ASlib Format")
        aslib.add_argument("-s", "--scenario", default=None,
                           help="directory with ASlib scenario files (required if not using --load or csv input files")

        csv = self._arg_parser.add_argument_group("Reading from CSV Format")
        csv.add_argument("--performance_csv", default=None,
                         help="performance data in csv table (column: algorithm, row: instance, delimeter: ,)")
        csv.add_argument("--feature_csv", default=None,
                         help="instance features data in csv table (column: features, row: instance, delimeter: ,)")
        csv.add_argument("--objective", default="solution_quality", choices=[
                         "runtime", "solution_quality"], help="Are the objective values in the performance data runtimes or an arbitrary solution quality (or cost) value")
        csv.add_argument("--runtime_cutoff", default=None, type=float,
                         help="cutoff time for each algorithm run for the performance data")
        csv.add_argument("--maximize", default=False, action="store_true", help="Set this parameter to indicate maximization of the performance metric (default: minimization)")

        opt = self._arg_parser.add_argument_group("Optional Options")
        opt.add_argument("-t", "--tune", action="store_true", default=False,
                         help="uses SMAC3 to determine a better parameter configuration")
        opt.add_argument(
            "-v", "--verbose", choices=["INFO", "DEBUG"], default="INFO", help="verbose level")
        opt.add_argument("--save", type=str, default=None,
                         help="trains AutoFolio and saves AutoFolio's state in the given filename")
        opt.add_argument("--load", type=str, default=None,
                         help="loads model (from --save); other modes are disabled with this options")
        opt.add_argument("--feature_vec", default=None, nargs="*",
                         help="feature vector to predict algorithm to use -- has to be used in combination with --load")

    def parse(self):
        '''
            uses the self._arg_parser object to parse the cmd line arguments

            Returns
            -------
                parsed arguments
                unknown arguments
        '''

        self.args_, misc_params = self._arg_parser.parse_known_args()

        return self.args_, misc_params

    def _check_args(self):
        '''
            checks whether all provides options are ok (e.g., existence of files)
        '''

        if not os.path.isdir(self.args_.scenario):
            self.logger.error(
                "ASlib Scenario directory not found: %s" % (self.args_.scenario))
            sys.exit(1)
