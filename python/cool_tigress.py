"""
This module defines python wrappers to the cool_tigress c routines.
Wrappers are not optimized and may not be appropriate for efficient calculation.
Basic code structure based on shell_sim.py in
https://bitbucket.org/krumholz/dusty_resolution_tests
"""

# List of routines provided
__all__ = ['fH2', 'fCplus', 'fHplus', 'fions', 'fe', 'fCO',
           'heatingCR', 'heatingH2pump', 'heatingPE',
           'coolingLya', 'coolingOI', 'coolingCII', 'coolingCI',
           'coolingCO', 'coolingRec', 'coolingHot',
           'get_abundances',
           'get_heating', 'get_cooling',
           '_CII_rec_rate', 'q10CII_', 'cooling2Level_', 'cooling3Level_'
           ]

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
    __libptr = npct.load_library("cool_tigress",
                                 os.path.realpath(path))

    # Define interface to functions
    __libptr.fH2.restype = c_double
    __libptr.fH2.argtypes = [c_double, c_double, c_double, c_double,
                             c_double]

    __libptr.fHplus.restype = c_double
    __libptr.fHplus.argtypes = [c_double, c_double, c_double, c_double,
                                c_double, c_double, c_double, c_double]

    __libptr.fCplus.restype = c_double
    __libptr.fCplus.argtypes = [c_double, c_double, c_double, c_double,
                                c_double, c_double, c_double, c_double,
                                c_double]

    __libptr.fions.restype = c_double
    __libptr.fions.argtypes = [c_double, c_double, c_double, c_double,
                               c_double, c_double, c_double, c_double,
                               c_double]

    __libptr.fe.restype = c_double
    __libptr.fe.argtypes = [c_double, c_double, c_double, c_double,
                            c_double, c_double, c_double, c_double,
                            c_double]

    __libptr.fCO.restype = c_double
    __libptr.fCO.argtypes = [c_double, c_double, c_double, c_double,
                             c_double, c_double, c_double]

    __libptr.heatingCR.restype = c_double
    __libptr.heatingCR.argtypes = [c_double, c_double, c_double, c_double,
                                   c_double]

    __libptr.heatingPE.restype = c_double
    __libptr.heatingPE.argtypes = [c_double, c_double, c_double, c_double,
                                   c_double]

    __libptr.heatingH2pump.restype = c_double
    __libptr.heatingH2pump.argtypes = [c_double, c_double, c_double, c_double,
                                       c_double]

    __libptr.coolingLya.restype = c_double
    __libptr.coolingLya.argtypes = [c_double, c_double, c_double, c_double]

    __libptr.coolingOI.restype = c_double
    __libptr.coolingOI.argtypes = [c_double, c_double, c_double, c_double,
                                   c_double, c_double]

    __libptr.coolingCII.restype = c_double
    __libptr.coolingCII.argtypes = [c_double, c_double, c_double, c_double,
                                    c_double, c_double]

    __libptr.coolingCI.restype = c_double
    __libptr.coolingCI.argtypes = [c_double, c_double, c_double, c_double,
                                   c_double, c_double]

    __libptr.coolingCO.restype = c_double
    __libptr.coolingCO.argtypes = [c_double, c_double, c_double, c_double,
                                   c_double, c_double, c_double]

    __libptr.coolingRec.restype = c_double
    __libptr.coolingRec.argtypes = [c_double, c_double, c_double, c_double,
                                    c_double]

    __libptr.coolingHot.restype = c_double
    __libptr.coolingHot.argtypes = [c_double, c_double]

    __libptr.get_cooling.restype = c_double
    __libptr.get_cooling.argtypes = [c_double, c_double, c_double, c_double,
                                     c_double, c_double, c_double, c_double,
                                     c_double, c_double, c_double, c_double]

    __libptr.get_heating.restype = c_double
    __libptr.get_heating.argtypes = [c_double, c_double, c_double, c_double,
                                     c_double, c_double, c_double, c_double,
                                     c_double]
        
    __libptr.get_abundances.restype = None
    __libptr.get_abundances.argtypes = [c_double, c_double, c_double, c_double,
                                        c_double, c_double, c_double, c_double,
                                        c_double,
                                        LP_c_double, LP_c_double,
                                        LP_c_double, LP_c_double,
                                        LP_c_double, LP_c_double,
                                        LP_c_double]
        
    __libptr.CII_rec_rate_.restype = c_double
    __libptr.CII_rec_rate_.argtypes = [c_double]

    __libptr.q10CII_.restype = c_double
    __libptr.q10CII_.argtypes = [c_double, c_double, c_double, c_double]

    __libptr.cooling2Level_.restype = c_double
    __libptr.cooling2Level_.argtypes = [c_double, c_double, c_double, c_double,
                                        c_double]

    __libptr.cooling3Level_.restype = c_double
    __libptr.cooling3Level_.argtypes = [c_double, c_double, c_double, c_double,
                                        c_double, c_double, c_double, c_double,
                                        c_double, c_double, c_double, c_double,
                                        c_double]
        
        
################################################################################
# Python wrappers to c routines
# The use of np.vectorize makes the code numpy-aware but doesn't make it faster.
################################################################################

# Equilibrium H2 fraction
@np.vectorize
def fH2(nH, T, Z_d, xi_CR, G_H2):
    """
    Compute equilibrium x_H2
    """
    loadlib()
    return __libptr.fH2(nH, T, Z_d, xi_CR, G_H2)

@np.vectorize
def fCplus(x_e, x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI):
    """
    Compute equilibrium x_C+
    """
    loadlib()
    return __libptr.fCplus(x_e, x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI)

@np.vectorize
def fHplus(x_e, x_Cplus, x_H2, nH, T, Z_d, xi_CR, G_PE):
    """
    Compute equilibrium x_H+
    """
    loadlib()
    return __libptr.fHplus(x_e, x_Cplus, x_H2, nH, T, Z_d, xi_CR, G_PE)

@np.vectorize
def fions(x_e, x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI):
    """
    Compute equilibrium x_C+ + x_H+
    """
    loadlib()
    return __libptr.fions(x_e, x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI)

@np.vectorize
def fe(x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI):
    """
    Compute equilibrium x_e
    """
    loadlib()
    return __libptr.fions(x_e, x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI)

@np.vectorize
def fCO(x_H2, x_Cplus, nH, Z_d, Z_g, xi_CR, G_CO):
    """
    Compute equilibrium x_CO
    """
    loadlib()
    return __libptr.fCO(x_H2, x_Cplus, nH, Z_d, Z_g, xi_CR, G_CO)

@np.vectorize
def heatingCR(x_e, x_HI, x_H2, nH, xi_CR):
    """
    Compute heating rate by cosmic rays [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.heatingCR(x_e, x_HI, x_H2, nH, xi_CR)

@np.vectorize
def heatingPE(x_e, nH, T, Z_d, G_PE):
    """
    Compute photoelectric heating rate per H by dust [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.heatingPE(x_e, nH, T, Z_d, G_PE)

@np.vectorize
def heatingH2pump(x_HI, x_H2, nH, T, G_H2):
    """
    Compute heating rate per H by UV pumping of H2 [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.heatingH2pump(x_HI, x_H2, nH, T, G_H2)

@np.vectorize
def coolingLya(x_e, x_HI, nH, T):
    """
    Compute cooling rate per H by Lya [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingLya(x_e, x_HI, nH, T)

@np.vectorize
def coolingOI(x_e, x_OI, x_HI, x_H2, nH, T):
    """
    Compute cooling rate per H by OI [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingOI(x_e, x_OI, x_HI, x_H2, nH, T)

@np.vectorize
def coolingCII(x_e, x_Cplus, x_HI, x_H2, nH, T):
    """
    Compute cooling rate per H by CII [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingCII(x_e, x_Cplus, x_HI, x_H2, nH, T)

@np.vectorize
def coolingCI(x_e, x_CI, x_HI, x_H2, nH, T):
    """
    Compute cooling rate per H by OI [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingCI(x_e, x_CI, x_HI, x_H2, nH, T)

@np.vectorize
def coolingCO(x_e, x_CO, x_HI, x_H2, nH, T, dvdr):
    """
    Compute cooling rate per H by OI [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingCO(x_e, x_CO, x_HI, x_H2, nH, T, dvdr)

@np.vectorize
def coolingRec(x_e, nH, T, Z_d, G_PE):
    """
    Compute cooling rate per H by recombination of e on PAHs [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingRec(x_e, nH, T, Z_d, G_PE)

@np.vectorize
def coolingHot(T, Z_g):
    """
    Compute cooling rate per H by recombination of e on PAHs [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingHot(T, Z_g)

def get_abundances(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2):
    """
    Compute equilibrium abundances
    """
    loadlib()
    x_e = c_double()
    x_HI = c_double()
    x_H2 = c_double()
    x_Cplus = c_double()
    x_CI = c_double()
    x_CO = c_double()
    x_OI = c_double()
    __libptr.get_abundances(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2,
                            byref(x_e), byref(x_HI), byref(x_H2),
                            byref(x_Cplus), byref(x_CI),
                            byref(x_CO), byref(x_OI))

    return np.array([x_e.value, x_HI.value, x_H2.value, x_Cplus.value,
                     x_CI.value, x_CO.value, x_OI.value], dtype=np.float64)

get_abundances = np.vectorize(get_abundances, otypes=[np.ndarray],
                              signature='(),(),(),(),(),(),(),(),()->(i)')

@np.vectorize
def get_heating(x_e, x_HI, x_H2, nH, T, Z, xi_CR, G_PE, G_H2):
    """
    Compute total heating rate [erg cm^3 s^-1]
    """
    loadlib()
    return __libptr.get_heating(x_e, x_HI, x_H2, nH, T, Z, xi_CR, G_PE, G_H2)

@np.vectorize
def get_cooling(x_e, x_HI, x_H2, x_Cplus, x_CI, x_CO, x_OI,
                nH, T, dvdr, Z, G_PE):
    """
    Compute total cooling rate [erg cm^3 s^-1]
    """
    loadlib()
    return __libptr.get_cooling(x_e, x_HI, x_H2, x_Cplus, x_CI, x_CO, x_OI,
                                nH, T, dvdr, Z, G_PE)

def CII_rec_rate_(T):
    """
    Compute CII (radiative + dielectronic) recombination rate coefficient
    [cm^3 s^-1]
    """
    loadlib()
    return __libptr.CII_rec_rate_(T)

@np.vectorize
def q10CII_(nHI, nH2, ne, T):
    """
    Compute collisional rate for CII [s^-1]
    """
    loadlib()
    return __libptr.q10CII_(nHI, nH2, ne, T)

@np.vectorize
def cooling2Level_(q01, q10, A10, E10, xs):
    """
    Compute cooling rate per H for 2 level system [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.cooling2Level_(q01, q10, A10, E10, xs)

@np.vectorize
def cooling3Level_(q01, q10, q02, q20, q12, q21, A10, A20, A21,
                   E10, E20, E21, xs):
    """
    Compute collisional rate per H for three-level system [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.cooling3Level_(q01, q10, q02, q20, q12, q21,
                                   A10, A20, A21, E10, E20, E21, xs)

#########################################################################
# Define a class to call wrappers more conveniently                     #
#########################################################################
class CoolTigress(object):
    """
    A class to compute tigress cooling and heating rates.
    """
    def __init__(self, nH=1.0, T=1e2,
                 Z=1.0, xi_CR=2.0e-16, dvdr=9.0e-14,
                 G_PE=1.0, G_CI=1.0, G_CO=1.0, G_H2=0.0, equil=False):
        
        self.nH = nH
        self.T = T
        self.Z = Z
        self.Z_g = self.Z
        self.Z_d = self.Z
        self.xi_CR = xi_CR
        self.dvdr = dvdr
        self.G_PE = G_PE
        self.G_CI = G_CI
        self.G_CO = G_CO
        self.G_H2 = G_H2

        self._set_par()

        if equil:
            self.get_nHeq()

        self.get_abundances()
        self.get_cooling()
        self.get_heating()

        # (thermal pressure)/k_B [cm^-3 K]
        self.pok = self.nH*self.T*(1.1 + self.x_e - self.x_H2)
            
    def get_heating(self):
        par = tuple([getattr(self, p) for p in self.par['get_heating']])
        self.heating = get_heating(*par)
        self.Gamma = self.nH*self.heating
        return self.heating

    def get_cooling(self):
        par = tuple([getattr(self, p) for p in self.par['get_cooling']])
        self.cooling = get_cooling(*par)
        self.Lambda = self.cooling
        return self.cooling
        
    def get_abundances(self):
        par = tuple([getattr(self, p) for p in self.par['get_abundances']])
        r = get_abundances(*par)

        self.x_e = r[...,0]
        self.x_HI = r[...,1]
        self.x_H2 = r[...,2]
        self.x_Cplus = r[...,3]
        self.x_CI = r[...,4]
        self.x_CO = r[...,5]
        self.x_OI = r[...,6]

    def get_heatingCR(self):
        par = tuple([getattr(self, p) for p in self.par['heatingCR']])
        self.heatingCR = heatingCR(*par)
        return self.heatingCR

    def get_heatingPE(self):
        par = tuple([getattr(self, p) for p in self.par['heatingPE']])
        self.heatingPE = heatingPE(*par)
        return self.heatingPE

    def get_heatingH2pump(self):
        par = tuple([getattr(self, p) for p in self.par['heatingH2pump']])
        self.heatingH2pump = heatingH2pump(*par)
        return self.heatingH2pump

    def get_coolingLya(self):
        par = tuple([getattr(self, p) for p in self.par['coolingLya']])
        self.coolingLya = coolingLya(*par)
        return self.coolingLya

    def get_coolingOI(self):
        par = tuple([getattr(self, p) for p in self.par['coolingOI']])
        self.coolingOI = coolingOI(*par)
        return self.coolingOI

    def get_coolingCII(self):
        par = tuple([getattr(self, p) for p in self.par['coolingCII']])
        self.coolingCII = coolingCII(*par)
        return self.coolingCII
    
    def get_coolingCI(self):
        par = tuple([getattr(self, p) for p in self.par['coolingCI']])
        self.coolingCI = coolingCI(*par)
        return self.coolingCI

    def get_coolingCO(self):
        par = tuple([getattr(self, p) for p in self.par['coolingCO']])
        self.coolingCO = coolingCO(*par)
        return self.coolingCO

    def get_coolingRec(self):
        par = tuple([getattr(self, p) for p in self.par['coolingRec']])
        self.coolingRec = coolingRec(*par)
        return self.coolingRec

    def get_coolingHot(self):
        par = tuple([getattr(self, p) for p in self.par['coolingHot']])
        self.coolingHot = coolingHot(*par)
        return self.coolingHot
    
    def get_nHeq(self, tol=1e-3):
        nHeq = []
        # Should we start from high T and low nHeq?
        # Doesn't seem to matter though...
        nHeq_ = 1e-4
        for T_ in np.flip(self.T):
            nHeq_ = CoolTigress._get_nHeq(
                nHeq_, T_, self.dvdr, self.Z, self.xi_CR,
                self.G_PE, self.G_CI, self.G_CO, self.G_H2, tol=tol)
            nHeq.append(nHeq_)

        self.nH = np.flip(np.array(nHeq))

    @staticmethod
    @np.vectorize
    def _get_nHeq(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2, tol=1e-3):
        """
        Given gas temperature and other parameters, compute nH for which
        heating is equal to cooling
        """

        from scipy import optimize
        def f(nH):
            x_e, x_HI, x_H2, x_Cplus, x_CI, x_CO, x_OI = \
              get_abundances(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2)

            heat = get_heating(x_e, x_HI, x_H2,
                               nH, T, Z, xi_CR, G_PE, G_H2)
            cool = get_cooling(x_e, x_HI, x_H2, x_Cplus, x_CI, x_CO, x_OI,
                               nH, T, dvdr, Z, G_PE)
            
            return np.abs((cool - heat)/(heat + cool))

        return optimize.brent(f, tol=tol)
    
    def _set_par(self):
        self.par = dict()
        
        # Should match in order and name input parameters of wrapper functions
        self.par['get_abundances'] = ['nH', 'T', 'dvdr', 'Z', 'xi_CR',
                                      'G_PE', 'G_CI', 'G_CO', 'G_H2']
        self.par['get_heating'] = ['x_e', 'x_HI', 'x_H2',
                                   'nH', 'T', 'Z', 'xi_CR', 'G_PE', 'G_H2']
        self.par['get_cooling'] = ['x_e', 'x_HI', 'x_H2', 'x_Cplus',
                                   'x_CI', 'x_CO', 'x_OI',
                                   'nH', 'T', 'dvdr', 'Z', 'G_PE']

        self.par['heatingCR'] = ['x_e', 'x_HI', 'x_H2', 'nH', 'xi_CR']
        self.par['heatingPE'] = ['x_e', 'nH', 'T', 'Z_d', 'G_PE']
        self.par['heatingH2pump'] = ['x_HI', 'x_H2', 'nH', 'T', 'G_H2']
        
        self.par['coolingLya'] = ['x_e', 'x_HI', 'nH', 'T']
        self.par['coolingOI'] = ['x_e', 'x_OI', 'x_HI', 'x_H2',
                                 'nH', 'T']
        self.par['coolingCII'] = ['x_e', 'x_Cplus', 'x_HI', 'x_H2',
                                  'nH', 'T']
        self.par['coolingCI'] = ['x_e', 'x_CI', 'x_HI', 'x_H2',
                                 'nH', 'T']
        self.par['coolingCO'] = ['x_e', 'x_CO', 'x_HI', 'x_H2',
                                 'nH', 'T', 'dvdr']
        self.par['coolingRec'] = ['x_e', 'nH', 'T', 'Z_d', 'G_PE']

        self.par['coolingHot'] = ['T', 'Z_g']
        
    def __repr__(self):
        return 'CoolingTigress:' + \
          'dvdr:{:g} Z: {:g} xi_CR: {:g} G_PE: {:g} G_CO: {:g} G_CI: {:g} G_H2: {:g}'.\
          format(self.dvdr, self.Z, self.xi_CR,
                 self.G_PE, self.G_CO, self.G_CI, self.G_H2)
