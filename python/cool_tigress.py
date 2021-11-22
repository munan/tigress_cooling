"""
This module defines python wrappers to the cool_tigress and linecool c routines.
Wrappers are not optimized and are not appropriate for efficient calculation.
Basic code structure based on shell_sim.py in
https://bitbucket.org/krumholz/dusty_resolution_tests
"""

# List of routines provided
__all__ = [ # From cool_tigress.c
           'fH2', 'fCplus', 'fHplus', 'fHplus_gr', 'fHplus_ng',
           'fions', 'fe', 'fCO',
           'heatingCR', 'heatingH2pump', 'heatingPE',
           'coolingLya', 'coolingOI', 'coolingCII', 'coolingCI',
           'coolingCO', 'coolingRec',
           'coolingHot', 'coolingHotHHe', 'coolingHotMetal',
           'get_abundances', 'get_abundances_fast',
           'get_heating', 'get_cooling',
           'CII_rec_rate_', 'q10CII_', 'cooling2Level_', 'cooling3Level_',
           'fShield_CO_V09_',
           # From linecool.cpp
           'get_EinsteinA',
           'get_energy_diff',
           'get_statistical_weight',
           'get_linecool_5lv',
           'get_linecool_all',
           'Photx'
           ]

# Imports
import os
import os.path as osp
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_double, c_int, c_uint8, \
    POINTER, byref # Structure, pointer, c_bool
import astropy.constants as ac
import pickle

# Import other objects here...(temporary)
from species_enum import EnumAtom, EnumLineCoolElem, EnumLineCoolTransition
from photx import Photx
from rec_rate import RecRate

# Number of five-level and two-level species
_N2LV = 3
_N5LV = len(EnumLineCoolElem) - _N2LV
_NTRANS5LV = len(EnumLineCoolTransition)

# Utility type definition
LP_c_double = POINTER(c_double)
array_1d_double = npct.ndpointer(dtype=c_double, ndim=1,
                                 flags="CONTIGUOUS")

# array_1d_int = npct.ndpointer(dtype=c_int, ndim=1,
#                               flags="CONTIGUOUS")

# Persistent value to hold pointer to c library
__libptr = None

#########################################################################
# Routine to load the library and define an interface to it             #
#########################################################################


def loadlib(path=None):
    """
    Function to load the cool_tigress library

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
        path = os.path.join(os.path.dirname(__file__), '../lib')

    __libptr = npct.load_library("cool_tigress",
                                 os.path.realpath(path))

    # Define interface to functions
    __libptr.fH2.restype = c_double
    __libptr.fH2.argtypes = [c_double, c_double, c_double, c_double,
                             c_double]

    __libptr.fHplus.restype = c_double
    __libptr.fHplus.argtypes = [c_double, c_double, c_double, c_double,
                                c_double, c_double, c_double, c_double]

    __libptr.fHplus_gr.restype = c_double
    __libptr.fHplus_gr.argtypes = [c_double, c_double, c_double, c_double,
                                   c_double, c_double, c_double]

    __libptr.fHplus_ng.restype = c_double
    __libptr.fHplus_ng.argtypes = [c_double, c_double, c_double, c_double,
                                   c_double, c_double]
        
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
                            c_double, c_int]

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

    __libptr.coolingHotHHe.restype = c_double
    __libptr.coolingHotHHe.argtypes = [c_double]

    __libptr.coolingHotMetal.restype = c_double
    __libptr.coolingHotMetal.argtypes = [c_double, c_double]
        
    __libptr.get_cooling.restype = c_double
    __libptr.get_cooling.argtypes = [c_double, c_double, c_double, c_double,
                                     c_double, c_double, c_double, c_double,
                                     c_double, c_double, c_double, c_double]

    __libptr.get_heating.restype = c_double
    __libptr.get_heating.argtypes = [c_double, c_double, c_double, c_double,
                                     c_double, c_double, c_double, c_double,
                                     c_double]

    __libptr.get_abundances.restype = None
    __libptr.get_abundances.argtypes = \
        [c_double, c_double, c_double, c_double,
         c_double, c_double, c_double, c_double,
         c_double, c_double, c_int,
         LP_c_double, LP_c_double,
         LP_c_double, LP_c_double,
         LP_c_double, LP_c_double,
         LP_c_double]
        
    __libptr.get_abundances_fast.restype = None
    __libptr.get_abundances_fast.argtypes = \
        [c_double, c_double, c_double, c_double,
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

    __libptr.fShield_CO_V09_.restype = c_double
    __libptr.fShield_CO_V09_.argtypes = [c_double, c_double]

    # functions from linecool module
    __libptr.get_EinsteinA.restype = c_double
    __libptr.get_EinsteinA.argtypes = [EnumLineCoolElem, c_int]

    __libptr.get_energy_diff.restype = c_double
    __libptr.get_energy_diff.argtypes = [EnumLineCoolElem, c_int]

    __libptr.get_statistical_weight.restype = c_double
    __libptr.get_statistical_weight.argtypes = [EnumLineCoolElem, c_uint8]

    __libptr.get_linecool_5lv.restype = c_double
    __libptr.get_linecool_5lv.argtypes = [EnumLineCoolElem, c_double,
                                          c_double, c_double]

    __libptr.get_linecool_all.restype = None
    __libptr.get_linecool_all.argtypes = [c_double, c_double, array_1d_double,
                                          array_1d_double, array_1d_double]

###############################################################################
# Python wrappers to c routines
# Using np.vectorize makes the code numpy-aware but doesn't make it faster.
###############################################################################

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
def fHplus_gr(x_e, x_H2, nH, T, Z_d, xi_CR, G_PE):
    """
    Compute equilibrium x_H+ with grain recombination 
    (ignores contribution of C+)
    """
    loadlib()
    return __libptr.fHplus_gr(x_e, x_H2, nH, T, Z_d, xi_CR, G_PE)

@np.vectorize
def fHplus_ng(x_H2, nH, T, Z_d, xi_CR, G_PE):
    """
    Compute equilibrium x_H+ (without grain assisted recombination)
    """
    loadlib()
    return __libptr.fHplus_ng(x_H2, nH, T, Z_d, xi_CR, G_PE)

@np.vectorize
def fions(x_e, x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI):
    """
    Compute equilibrium x_C+ + x_H+
    """
    loadlib()
    return __libptr.fions(x_e, x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI)

@np.vectorize
def fe(x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI, x_e_init, maxiter):
    """
    Compute equilibrium x_e
    """
    loadlib()
    return __libptr.fe(x_H2, nH, T, Z_d, Z_g, xi_CR, G_PE, G_CI, x_e_init, maxiter)


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
    Compute cooling rate per H by hot gas [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingHot(T, Z_g)

@np.vectorize
def coolingHotHHe(T):
    """
    Compute cooling rate per H by hot gas (H and He only) [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingHotHHe(T)

@np.vectorize
def coolingHotMetal(T, Z_g):
    """
    Compute cooling rate per H by hot gas (metal only) [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.coolingHotMetal(T, Z_g)


def get_abundances(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2,
                   x_e_init, maxiter):
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
                            x_e_init, maxiter,
                            byref(x_e), byref(x_HI), byref(x_H2),
                            byref(x_Cplus), byref(x_CI),
                            byref(x_CO), byref(x_OI))

    return np.array([x_e.value, x_HI.value, x_H2.value, x_Cplus.value,
                     x_CI.value, x_CO.value, x_OI.value], dtype=np.float64)

get_abundances = np.vectorize(get_abundances, otypes=[np.ndarray],
                              signature='(),(),(),(),(),(),(),(),(),(),()->(i)')

def get_abundances_fast(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2):
    """
    Compute equilibrium abundances (without approximation)
    """
    loadlib()
    x_e = c_double()
    x_HI = c_double()
    x_H2 = c_double()
    x_Cplus = c_double()
    x_CI = c_double()
    x_CO = c_double()
    x_OI = c_double()
    __libptr.get_abundances_fast(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2,
                            byref(x_e), byref(x_HI), byref(x_H2),
                            byref(x_Cplus), byref(x_CI),
                            byref(x_CO), byref(x_OI))

    return np.array([x_e.value, x_HI.value, x_H2.value, x_Cplus.value,
                     x_CI.value, x_CO.value, x_OI.value], dtype=np.float64)

get_abundances_fast = np.vectorize(get_abundances_fast, otypes=[np.ndarray],
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


@np.vectorize
def fShield_CO_V09_(NCO, NH2):
    """
    Compute CO shielding factor (Visser+09 Table 5)
    """
    loadlib()
    return __libptr.fShield_CO_V09_(NCO, NH2)


@np.vectorize
def get_EinsteinA(elem, trans):
    """
    Get Einstein A coefficient for transition [s^-1]
    """
    loadlib()
    return __libptr.get_EinsteinA(elem, trans)


@np.vectorize
def get_energy_diff(elem, trans):
    """
    Get energy difference
    """
    loadlib()
    return __libptr.get_energy_diff(elem, trans)


@np.vectorize
def get_statistical_weight(elem, level):
    """
    Get statistical weight of level
    """
    loadlib()
    return __libptr.get_statistical_weight(elem, level)


@np.vectorize
def get_linecool_5lv(elem, T, ne, abundance):
    """
    Get line cooling rate [erg s^-1 H^-1]
    """
    loadlib()
    return __libptr.get_linecool_5lv(elem, T, ne, abundance)


def get_linecool_abd_arr(abd):

    Elem = EnumLineCoolElem
    abd_arr = np.empty(_N5LV + _N2LV)
    for elem in Elem:
        abd_arr[elem.value] = abd[elem.name]

    return abd_arr


def get_linecool_all(T, n_e, abd):
    """
    Get line cooling rate [erg s^-1 H^-1]
    """
    loadlib()
    abd_arr = get_linecool_abd_arr(abd)
    lcool_5lv = np.zeros(_N5LV*_NTRANS5LV)
    lcool_2lv = np.zeros(_N2LV)
    __libptr.get_linecool_all(T, n_e, abd_arr, lcool_5lv, lcool_2lv)

    return np.append(lcool_5lv, lcool_2lv)


get_linecool_all = np.vectorize(get_linecool_all, otypes=[np.ndarray],
                                signature='(),(),()->(i)')


#########################################################################
# Define a class to call wrappers more conveniently                     #
#########################################################################
class CoolTigress(object):
    """
    A class to compute tigress cooling and heating rates.
    """

    def __init__(self, nH=1.0, T=1e2,
                 Z=1.0, xi_CR=2.0e-16, dvdr=9.0e-14,
                 G_PE=1.0, G_CI=1.0, G_CO=1.0, G_H2=1.0, equil=False,
                 x_e_init=0.5, maxiter=200,
                 fast_flag=True, load_from_pickle=False):

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
        self.fast_flag = fast_flag
        self._set_par()

        self.x_e_init = x_e_init
        self.maxiter = maxiter
        
        if equil:
            self.get_nHeq()

        if self.fast_flag:
            self.get_abundances_fast()
        else:
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

        self.x_e = r[..., 0]
        self.x_HI = r[..., 1]
        self.x_H2 = r[..., 2]
        self.x_Cplus = r[..., 3]
        self.x_CI = r[..., 4]
        self.x_CO = r[..., 5]
        self.x_OI = r[..., 6]

    def get_abundances_fast(self):
        par = tuple([getattr(self, p) for p in self.par['get_abundances_fast']])
        r = get_abundances_fast(*par)

        self.x_e = r[..., 0]
        self.x_HI = r[..., 1]
        self.x_H2 = r[..., 2]
        self.x_Cplus = r[..., 3]
        self.x_CI = r[..., 4]
        self.x_CO = r[..., 5]
        self.x_OI = r[..., 6]

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

    def get_nHeq(self, tol=2e-3):
        nHeq = []
        # Should we start from high T and low nHeq?
        # Doesn't seem to matter though...
        nHeq_ = 1e-4
        for T_ in np.flip(self.T):
            nHeq_ = CoolTigress._get_nHeq(
                nHeq_, T_, self.dvdr, self.Z, self.xi_CR,
                self.G_PE, self.G_CI, self.G_CO, self.G_H2,
                self.x_e_init, self.maxiter, self.fast_flag,
                tol=tol)
            nHeq.append(nHeq_)

        self.nH = np.flip(np.array(nHeq))

    def save(self, savdir=None, verbose=True):

        if savdir is None:
            savdir = osp.join(osp.dirname(__file__), 'pickle')
        if not osp.isdir(savdir):
            os.makedirs(savdir)
        fpkl = osp.join(savdir, 'cool_equil_xi{0:.1e}_GPE{1:.1g}_Z{2:.1g}'.\
                        format(self.xi_CR, self.G_PE, self.Z))
        if self.fast_flag:
            fpkl += '_fast'
        self.fpkl = fpkl + '.p'

        res = dict()
        for k in self.__dict__.keys():
            res[k] = getattr(self, k)
       
        pickle.dump(res, open(self.fpkl, 'wb'))
        if verbose:
            print('pickled to ', self.fpkl)
            
    def load(self, savdir=None):

        if savdir is None:
            savdir = osp.join(osp.dirname(__file__), 'pickle')
        if not osp.isdir(savdir):
            os.makedirs(savdir)
        fpkl = osp.join(savdir, 'cool_equil_xi{0:.1e}_GPE{1:.1g}_Z{2:.1g}'.\
                        format(self.xi_CR, self.G_PE, self.Z))
        if self.fast_flag:
            fpkl += '_fast'
        self.fpkl = fpkl + '.p'

        res = pickle.load(open(self.fpkl, 'rb'))
        return res        
            
    @staticmethod
    @np.vectorize
    def _get_nHeq(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2, 
                  x_e_init, maxiter, fast_flag, tol=1e-4, verbose=False):
        """
        Given gas temperature and other parameters, compute nH for which
        heating is equal to cooling
        """
        
        # lognH = np.log10(nH)
        from scipy import optimize
        if fast_flag:
            def f(nH):
                # nH = 10.0**lognH
                x_e, x_HI, x_H2, x_Cplus, x_CI, x_CO, x_OI = \
                    get_abundances_fast(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2)

                heat = get_heating(x_e, x_HI, x_H2,
                                   nH, T, Z, xi_CR, G_PE, G_H2)
                cool = get_cooling(x_e, x_HI, x_H2, x_Cplus, x_CI, x_CO, x_OI,
                                   nH, T, dvdr, Z, G_PE)
                
                return np.abs((cool - heat)/(heat + cool))
                # return (cool - heat)**2/(heat + cool)**2

        else:
            def f(nH):
                # nH = 10.0**lognH
                x_e, x_HI, x_H2, x_Cplus, x_CI, x_CO, x_OI = \
                    get_abundances(nH, T, dvdr, Z, xi_CR, G_PE, G_CI, G_CO, G_H2,
                                   x_e_init, maxiter)
                heat = get_heating(x_e, x_HI, x_H2,
                                   nH, T, Z, xi_CR, G_PE, G_H2)
                cool = get_cooling(x_e, x_HI, x_H2, x_Cplus, x_CI, x_CO, x_OI,
                                   nH, T, dvdr, Z, G_PE)
                
                return np.abs((cool - heat)/(heat + cool))
                #return (cool - heat)**2/(heat + cool)**2

        nHeq, fval, it, funcalls = optimize.brent(f, tol=tol,
                                                  full_output=True)
        # nHeq = 10.0**xmin
        if fval > tol:
            # if verbose:
            #    print('xmin,fval,iter,funcalls',xmin, fval, it, funcalls)
            return np.nan
        else:
            return nHeq


        return nHeq

    def _set_par(self):
        self.par = dict()

        # Should match in order and name input parameters of wrapper functions
        self.par['get_abundances'] = ['nH', 'T', 'dvdr', 'Z', 'xi_CR',
                                      'G_PE', 'G_CI', 'G_CO', 'G_H2',
                                      'x_e_init', 'maxiter']
        self.par['get_abundances_fast'] = ['nH', 'T', 'dvdr', 'Z', 'xi_CR',
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
            'dvdr:{:g} Z: {:g} xi_CR: {:g} G_PE: {:g} G_CO: \
            {:g} G_CI: {:g} G_H2: {:g}'. \
            format(self.dvdr, self.Z, self.xi_CR,
                   self.G_PE, self.G_CO, self.G_CI, self.G_H2)


class LineCool(object):
    """
    A class to compute line cooling rates from ionized gas.
    """

    def __init__(self, n_e=1e2, T=8.0e3, kind='Orion'):

        # Enum objects
        self.Elem = EnumLineCoolElem
        self.Trans = EnumLineCoolTransition

        self.n_e = n_e
        self.T = T
        self.abd = self.set_linecool_abd(kind=kind)

        self.alphaB = self.get_alphaB(self.T)
        # Cooling by radiative recombination from Draine (2011)
        self.cooling_rr = self.alphaB*(0.684 - 0.0416*np.log(T/1e4)) *\
            ac.k_B.cgs.value*self.T*self.n_e
        # Cooling by free-free emission from Draine (2011)
        self.cooling_ff = 0.54*(self.T/1e4)**0.37*ac.k_B.cgs.value*self.T *\
            self.alphaB*self.n_e

    @staticmethod
    def get_alphaB(T):
        """Function to calculate the case B radiative recombination rate of
        Hydrogen"""
        T4 = T/1e4
        return 2.59e-13*(T4)**(-0.833 - 0.035*np.log(T4))

    def get_linecool_all(self):

        cool_all = get_linecool_all(self.T, self.n_e, self.abd)

        self.cool_5lv = cool_all[..., 0:_N5LV*_NTRANS5LV].\
            reshape(-1, _N5LV, _NTRANS5LV)
        self.cool_2lv = cool_all[..., _N5LV*_NTRANS5LV:]

        t = self.Trans
        # NII
        lines = dict()
        lines['NII'] = dict()
        lines['NII']['6585A'] = t.T23
        lines['NII']['6550A'] = t.T13

        # OII
        lines['OII'] = dict()
        lines['OII']['3730A'] = t.T01
        lines['OII']['3727A'] = t.T02

        # OIII
        lines['OIII'] = dict()
        lines['OIII']['88mu'] = t.T01
        lines['OIII']['52mu'] = t.T12
        lines['OIII']['5008A'] = t.T23
        lines['OIII']['4960A'] = t.T13
        lines['OIII']['4364A'] = t.T34

        # SII
        lines['SII'] = dict()
        lines['SII']['6733A'] = t.T01
        lines['SII']['6718A'] = t.T02

        # SIII
        lines['SIII'] = dict()
        lines['SIII']['33mu'] = t.T01
        lines['SIII']['19mu'] = t.T12
        lines['SIII']['9071A'] = t.T13
        lines['SIII']['9533A'] = t.T23

        r = dict()
        for i, e in enumerate(self.Elem):
            en = e.name
            r[en] = dict()
            if i < _N5LV:
                r[en]['tot'] = self.cool_5lv[..., e, :].sum(axis=1)
            else:
                r[en]['tot'] = self.cool_2lv[..., e - _N5LV]

            try:
                for l, trans in lines[en].items():
                    if i < _N5LV:
                        r[en][l] = self.cool_5lv[..., e, trans]
                    else:
                        r[en][l] = self.cool_2lv[..., e - _N5LV]
            except KeyError:
                # print('No registered line for {:s}'.format(en))
                pass

        # # e = Elem.NII
        # # en = e.name
        # # r[en] = lstr_5lv[..., e, :]
        # # r[en + 'tot'] = lstr_5lv[..., e, :].sum(axis=1)
        # # r[en + '6585A'] = lstr_5lv[..., e, t.T23]
        # # r[en + '6550A'] = lstr_5lv[..., e, t.T13]

        # # e = self.Elem.OII
        # # en = e.name
        # # r[e.name] = lstr_5lv[...,e,:]
        # # r[en+'tot'] = lstr_5lv[...,e,:].sum(axis=1)
        # # r[en+'6585'] = lstr_5lv[...,e,t.T23]
        # # r[en+'6550'] = lstr_5lv[...,e,t.T13]

        self.cooling_ce_tot = np.array(
            [r_['tot'] for r_ in r.values()]).sum(axis=0)
        self.cooling_ce = r

        return r

    def set_linecool_abd(self, kind='Orion',
                         CI_C=0.0, NI_N=0.0, OI_O=0.0, NeI_Ne=0.0, SI_S=0.0,
                         CII_C=1.0, NII_N=0.8, OII_O=0.8, NeII_Ne=0.8,
                         SII_S=0.2, SIII_S=0.8):
        """Function to set gas phase elemental abundance relative to H.
        Neglect molecular abundance.

        Parameters
        ----------
        kind: str
            Use gas phase abundance of Orion or Lexington benchmark tests

        Returns
        -------
        abd : dict
            Abundances of atoms and ions
        """

        self.kind = kind

        abd = dict()
        # Abundances in the Lexington benchmark tests
        # (from Table 2 in Vandenbroucke+18)
        if kind == 'Lexington':
            abd['He'] = 0.1
            abd['C'] = 2.2e-4
            abd['N'] = 4.0e-5
            abd['O'] = 3.3e-4
            abd['Ne'] = 5.0e-5
            abd['S'] = 9.0e-6
        elif kind == 'AGN2':  # p.61 in Osterbrock & Ferland 2006
            abd['He'] = 0.1
            abd['C'] = 2.2e-4  # Not included?
            abd['N'] = 9.0e-5
            abd['O'] = 7.0e-4
            abd['Ne'] = 9.0e-5
            abd['S'] = 9.0e-6  # Not included?
        elif kind == 'Orion':
            # Orion gas phase abundance (Table 17 in Esteban+98)
            abd['He'] = 10.0**(-12.0 + 10.99)
            abd['C'] = 10.0**(-12.0 + 8.39)
            abd['N'] = 10.0**(-12.0 + 7.78)
            abd['O'] = 10.0**(-12.0 + 8.64)
            abd['Ne'] = 10.0**(-12.0 + 7.89)
            abd['S'] = 10.0**(-12.0 + 7.17)
        else:
            raise Exception("Unsupported abundance kind {:s}.".format(kind))

        abd['CI'] = abd['C']*CI_C
        abd['NI'] = abd['N']*NI_N
        abd['OI'] = abd['O']*OI_O
        abd['NeI'] = abd['Ne']*NeI_Ne
        abd['SI'] = abd['S']*SI_S

        abd['CII'] = abd['C']*CII_C
        abd['NII'] = abd['N']*NII_N
        abd['OII'] = abd['O']*OII_O
        abd['NeII'] = abd['Ne']*NeII_Ne
        abd['SII'] = abd['S']*SII_S

        abd['CIII'] = abd['C'] - abd['CI'] - abd['CII']
        abd['NIII'] = abd['N'] - abd['NI'] - abd['NII']
        abd['OIII'] = abd['O'] - abd['OI'] - abd['OII']
        abd['NeIII'] = abd['Ne'] - abd['NeI'] - abd['NeII']

        abd['SIII'] = abd['S']*SIII_S
        abd['SIV'] = abd['S'] - abd['SI'] - abd['SII'] - abd['SIII']

        return abd
