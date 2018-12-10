"""
This module defines python wrappers to linecool c routines.
Wrappers are not optimized and may not be appropriate for efficient calculation.
Basic code structure based on shell_sim.py in
https://bitbucket.org/krumholz/dusty_resolution_tests
"""

# List of routines provided
__all__ = ['get_transition_probability']

# Imports
import numpy as np
import pandas as pd
import numpy.ctypeslib as npct
from ctypes import c_double, c_int, c_bool, pointer, POINTER, Structure, byref
import os

# # Utility type definition
LP_c_double = POINTER(c_double)
# array_1d_double = npct.ndpointer(dtype=np.double, ndim=1,
#                                  flags="CONTIGUOUS")
# array_1d_int = npct.ndpointer(dtype=c_int, ndim=1,
#                               flags="CONTIGUOUS")

# Persistent value to hold pointer to c library
__libptr = None

#########################################################################
# Routine to load the library and define an interface to it             #
#########################################################################
def loadlib(path=None):
    """
    Function to load the library
    Parameters
    ----------
    path : string
        path to library; default is current directory

    Returns
    -------
    nothing
    """
    
    # Point to global libptr
    global __libptr

    # Do nothing if already loaded
    if __libptr is not None:
        return

    # Set path if not specified
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..')
        
    # Load library
    __libptr = npct.load_library("linecool_c_wrapper",
                                 os.path.realpath(path))

    # Define interface to functions
    __libptr.get_transition_probability.restype = c_double
    __libptr.fH2.argtypes = [c_int, c_int]
