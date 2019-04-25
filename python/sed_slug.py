"""
Class to read SED from SLUG cluster simulation output. slugpy should be
installed or added to PYTHONPATH.
"""

import os, time
import numpy as np
from astropy import units as au
from astropy import constants as ac
import pickle

from slugpy import read_cluster_spec, read_cluster_phot, read_cluster_prop

class SEDSlug(object):
    """Class to read SED from SLUG cluster simulation output.
    """
    
    def __init__(self,
                 output_dir='/tigress/jk11/slug_cluster_data', fmt='bin',
                 model_base='cluster_logM', 
                 logM_all=np.linspace(2.0, 5.0, 16),
                 proc_all=range(5)):

        ## simulation and output parameters
        self.output_dir = output_dir
        self.fmt = fmt
        
        self.model_base = model_base
        self.logM_all = logM_all
        # thread id (p0001...p000x)
        self.proc_all = proc_all
        
        # physical units and constants
        self.Angs = (1.0*au.Angstrom).cgs.value
        self.hc = (1.0*(ac.h*ac.c).cgs).value
        self.L_sun = (1.0*au.L_sun).cgs.value
        
        # pickle directory
        self.pkl_dir = os.path.join(output_dir, 'pickle')
        if not os.path.exists(self.pkl_dir):
            os.makedirs(self.pkl_dir)
        
    def get_sed(self, logM):
        """Function to read all sed for logM

        Returns
        -------
        (time, wl, sed): tuple
            numpy arrays for age of cluster [yr], wavelenth [Angstrom],
            and sed [erg/s/A].
        """
        
        print('logM{:02d} proc: '.format(int(10.0*logM)), end=' ')

        self.ntrial = 0
        for i, proc in enumerate(self.proc_all):
            print(proc, end=' ')
            self.model_name = self.model_base + \
              '{0:02d}_p{1:05d}_n00000_0000'.format(int(10.0*logM), proc)
            self.prop = read_cluster_prop(self.model_name,
                                          output_dir=self.output_dir,
                                          fmt=self.fmt)
            self.phot = read_cluster_phot(self.model_name,
                                          output_dir=self.output_dir,
                                          fmt=self.fmt, nofilterdata=True)
            self.spec = read_cluster_spec(self.model_name,
                                          output_dir=self.output_dir,
                                          fmt=self.fmt)

            trial = np.unique(self.spec.trial)
            self.ntrial += len(trial)
            if i == 0:
                self.ntime = self.spec.spec.shape[0] // self.ntrial
                self.time = self.phot.time[0:self.ntime]

            sed_ = np.reshape(self.spec.spec,
                              (len(trial), self.ntime, self.spec.wl.size))
            sed_ = sed_ / self.prop.target_mass[0]

            if i == 0:
                self.sed = sed_
            else:
                self.sed = np.vstack((self.sed, sed_))

        print(' ')
                
        return self.time, self.spec.wl, self.sed
        
    def get_sed_med_avg(self, force_override=False, verbose=True):

        fpkl = os.path.join(self.pkl_dir,
                            self.model_base + "_sed_med_avg.p")
        
        # Check if pickle exists
        print(fpkl)
        if not force_override and os.path.isfile(fpkl):
            if verbose:
                print('Read from pickle:{:s}'.format(fpkl))
            return pickle.load(open(fpkl,'rb'))
        else:
            if verbose:
                print('[get_sed_med_avg]: Read data from binary output.')
                
        sed_med = dict()
        sed_avg = dict()
        ntrial = dict()
        
        #for i, logM in enumerate([3.0, 5.0]): # For test
        for i, logM in enumerate(self.logM_all):
            self.get_sed(logM=logM)
            sed_avg['logM{0:02d}'.format(int(logM*10.))] = \
              np.average(self.sed, axis=0)
            sed_med['logM{0:02d}'.format(int(logM*10.))] = \
              np.median(self.sed, axis=0)
            ntrial['logM{0:02d}'.format(int(logM*10.))] = self.ntrial

        out = dict(sed_avg=sed_avg, sed_med=sed_med, wl=self.spec.wl, time=self.time,\
                   ntrial=ntrial, ntime=self.ntime)
        
        with open(fpkl,'wb') as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

        return out

