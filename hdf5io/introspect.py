import logging
import logging.handlers
log = logging.getLogger('introspect')

from contextlib import contextmanager
import time
import re
import sys
import numpy
import copy

import weakref
import subprocess
from collections.abc import Iterable
import os


def is_iterable(obj):
    """
    True if obj is iterable but not a string-like object (str, unicode, bytearray)

    """
    return not hasattr(obj, 'lower') and isinstance(obj, Iterable)

@contextmanager
def nostderr():
    savestderr = sys.stderr
    class Devnull(object):
        def write(self, _): pass
    sys.stderr = Devnull()
    yield
    sys.stderr = savestderr

def hash_variables(variables, digest = None):
        import hashlib
        import pickle
        myhash = hashlib.md5()
        if not isinstance(variables, (list, tuple)): variables = [variables]
        for v in variables:
            myhash.update(pickle.dumps(v))
        return myhash.digest() if digest is None else myhash.hexdigest()

def nameless_dummy_object_with_methods(*methods):
    d = {}
    for sym in methods:
        d[sym] = lambda self,*args,**kwargs: None
    return type("",(object,),d)()

def list_of_empty_mutables(n, prototype=list()):
    return [copy.deepcopy(prototype) for _ in range(n)]

def dict_of_empty_mutables(keys,prototype=list(),dict_type = dict):
    return dict_type(list(zip(keys,list_of_empty_mutables(len(keys),prototype))))
    
def traverse(obj,  attrchain):
    '''Walks trough the attribute chain starting from obj and returns the last element of the chain. E.g.
    attrchain = '.h5f.root.rawdata' will return obj.h5f.root.rawdata if all members of the chain exist'''
    attrs = re.findall('\.*(\w+)\.*', attrchain)
    for a in attrs:
        if not hasattr(obj, a):#not a in obj.__dict__:
            return None
        obj = getattr(obj, a)
    return obj

class Timer(object):
    '''Simple measurement utility, use as:
    with Timer('give_a_name'):
        statement1
        statementN
    '''
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(('[%s]' % self.name))
        print(('Elapsed: %s' % (time.time() - self.tstart)))

def list_type(item):
    """
    Assumes input is a list and analyzes what items the list contains
    Parameters
    ----------
    item : can be any object, expects either a list or a numpy array.

    Returns
    -------
    string
        'arrayized' means input data could be transformed into numpy array
        'list_of_lists' means input is a list containing either lists or tuples
        'list_of_dicts' if all elements are dicts
        'list_of_arrays' if all elements are numpy arrays
        'list_of_uniform_shaped_recarrays' if all elements are recarrays with same shape
        'list_of_recarrays' if shapes of recarrays are different
        'inhomogenous_list' if elements are mixed, i.e. none of the other cases apply

    """
    try: # is data convertible to numpy array of scalar (including string) values?
        item2 = numpy.array(item)
        if isinstance(item, (list, tuple)) and item2.dtype.names is None and item2.dtype != object:#numpy.issctype(item2.dtype):
            return 'arrayized'
    except Exception as e:
        print(e)
    if isinstance(item,(list,tuple)):
        if isinstance(item[0],(list,tuple)):
            response='list_of_lists'
        elif sum(isinstance(i0,dict) for i0 in item)==len(item):
            response='list_of_dicts'
        elif sum(hasattr(i0,'shape') and len(i0.dtype) == 0 for i0 in item)==len(item):
            response = 'list_of_arrays'
        elif sum(hasattr(i0,'shape') and len(i0.dtype) > 0 for i0 in item)==len(item):
            if all([i0.shape==item[0].shape for i0 in item]):
                response = 'list_of_uniform_shaped_recarrays'
            else:
                response = 'list_of_recarrays'
        else:
            response = 'inhomogenous_list'
    else:
        response=None
    return response

def dict_isequal(d1,d2):
    if set(d1.keys()) != set(d2.keys()): return False
    for k in list(d1.keys()):
        if hasattr(d1[k],'shape'):
            if numpy.any(d1[k]!=d2[k]): return False
        else: 
            if d1[k]!=d2[k]: return False
    return True







    
