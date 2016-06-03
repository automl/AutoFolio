import logging

from autofolio.io.cmd import CMDParser
from autofolio.data.aslib_scenario import ASlibScenario

__author__= "Marius Lindauer"
__license__ = "BSD"

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
        
        
        
        
        