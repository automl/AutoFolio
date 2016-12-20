#!/usr/bin/env python

import logging
import sys
import os
import inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0,cmd_folder)

import pickle
import argparse
from graphviz import Digraph
import traceback

def load_model(model_fn: str):
    '''
        loads saved model

        Arguments
        --------
        model_fn: str
            file name of saved model
            
        Returns
        -------
        scenario, feature_pre_pipeline, pre_solver, selector, config
    '''
    with open(model_fn, "br") as fp:
        scenario, feature_pre_pipeline, pre_solver, selector, config = pickle.load(
            fp)

    for fpp in feature_pre_pipeline:
        fpp.logger = logging.getLogger("Feature Preprocessing")
    if pre_solver:
        pre_solver.logger = logging.getLogger("Aspeed PreSolving")
    selector.logger = logging.getLogger("Selector")
    
    return scenario, feature_pre_pipeline, pre_solver, selector, config

def visualize(feature_pre_pipeline, pre_solver, selector):
    '''
        visualize all loaded components
        
        Arguments
        ---------
        feature_pre_pipeline: list
                list of fitted feature preprocessors
        pre_solver: Aspeed
                pre solver object with a saved static schedule
        selector: autofolio.selector.*
                fitted selector object
    '''
    
    dot = Digraph(comment='AutoFolio')
    for idx,fpp in enumerate(feature_pre_pipeline):
        dot.node('fpp_%d' %(idx), fpp.__class__.__name__)
        if idx > 0:
            dot.edge('fpp_%d' %(idx-1),'fpp_%d' %(idx))
        add_attributes(obj=fpp, node_name='fpp_%d' %(idx), dot=dot)
            
    for idx,presolver in enumerate(pre_solver.schedule):
        dot.node('pre_%d' %(idx), "%s for %d sec" %(presolver[0], presolver[1]))
        if idx > 0:
            dot.edge('pre_%d' %(idx-1),'pre_%d' %(idx))
        elif feature_pre_pipeline:
            dot.edge('fpp_%d' %(len(feature_pre_pipeline)-1),'pre_%d' %(idx))
            
    dot.node("selector", selector.__class__.__name__)
    if pre_solver:
        dot.edge('pre_%d' %(len(pre_solver.schedule)-1), "selector")
    add_attributes(obj=selector, node_name="selector", dot=dot)
                   
        
    dot.render('test-output/autofolio', view=True)
    
def add_attributes(obj, node_name:str, dot:Digraph):
    '''
        add attributes of <obj> with <node_name> to <dot>
        
        Arguments
        ---------
        obj: Object
            should implement get_attributes()
        node_name: str
            node name of obj in dot
        dot: Digraph
            digraph of graphviz
    '''
    
    try:
        attributes = obj.get_attributes()
    except AttributeError:
        #traceback.print_exc()
        return
    for name, attr in attributes:
        if isinstance(attr, str):
            dot.node(name, "%s:%s"%(name, attr))
        elif isinstance(attr, list):
            dot.node(name, "%s"%(name))
            for s in attr:
                dot.node(str(s), str(s))
                dot.edge("%s"%(name), str(s))
        else:
            print("UNKNOWN TYPE: %s" %(attr))
            
        dot.edge(node_name,name)
        
    

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--load", type=str, default=None,
                         help="loads model (from --save); other modes are disabled with this options")
args = parser.parse_args()
        
scenario, feature_pre_pipeline, pre_solver, selector, config = load_model(args.load)

print(config)

visualize(feature_pre_pipeline, pre_solver, selector)



