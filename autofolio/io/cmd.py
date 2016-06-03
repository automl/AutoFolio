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

        req = self._arg_parser.add_argument_group("Required Options")
        req.add_argument("-s", "--scenario", required=True, help="directory with ASlib scenario files")
        
        opt = self._arg_parser.add_argument_group("Optional Options")
        opt.add_argument("-v", "--verbose", choices=["INFO","DEBUG"], default="INFO", help="verbose level")
        
    def parse(self):
        
        self.args_ = self._arg_parser.parse_args()
        
        return self.args_
    
    
    def _check_args(self):
        '''
            checks whether all provides options are ok (e.g., existence of files)
        '''
        
        if not os.path.isdir(self.args_.scenario):
            self.logger.error("ASlib Scenario directory not found: %s" %(self.args_.scenario))
            sys.exit(1)
            
        
        