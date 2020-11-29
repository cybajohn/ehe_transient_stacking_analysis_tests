"""
Just to test ratio values, they should be the same... right?
"""
import gc  # Manual garbage collection
import os
import json
import gzip
import argparse
import numpy as np
import sys

from tdepps.utils import make_src_records
from tdepps.grb import GRBLLH, GRBModel, MultiGRBLLH
from tdepps.grb import TimeDecDependentBGDataInjector
from tdepps.grb import MultiBGDataInjector
from tdepps.grb import GRBLLHAnalysis
import tdepps.utils.phys as phys
from _paths import PATHS
import _loader

from tdepps.utils import interval_overlap

from tdepps.utils import make_time_dep_dec_splines
from tdepps.utils import make_grid_interp_kintscher_style_second_attempt
from tdepps.utils import make_grid_interp_from_hist_ratio

import histlite as hl
from matplotlib import pyplot as plt

def flux_model_factory(model, **model_args):
    """
    Returns a flux model callable `flux_model(trueE)`.

    Parameters
    ----------
    model : str
        Name of a method in ``tdeps.utils.phys``.
    model_args : dict
        Arguments passed to ``tdeps.utils.phys.<model>``.

    Returns
    -------
    flux_model : callable
        Function of single parameter, true energy, with fixed model args.
    """
    def flux_model(trueE):
        flux_mod = getattr(phys, model)
        return flux_mod(trueE, **model_args)
    return flux_model
	
def give_sample(n_events, sigma, time=10.):
	sam = {"ra": [], "dec": [], "time" : [], "logE": [], "sigma": []}
	sam_dec = np.linspace(0. ,2. * np.pi, n_events)
	sam_dec = np.resize(sam_dec, n_events**2)
	sam_ra = np.linspace(0. ,2. * np.pi, n_events)
	sam_ra = np.repeat(sam_ra,n_events)
	sam_time = np.ones(n_events**2) * time
	sam_logE = np.linspace(1. ,9. , n_events)
	sam_logE = np.repeat(sam_logE, n_events)
	sam_sigma = np.ones(n_events**2) * sigma
	sam["ra"] = sam_ra
	sam["dec"] = sam_dec
	sam["logE"] = sam_logE
	sam["sigma"] = sam_sigma
	sam["time"] = sam_time
	return sam

def give_srcs(times=[10.], length=5. ,nsrcs=1):
	dtype = [("ra", float), ("dec", float), ("time", float),
        	("dt0", float), ("dt1", float), ("w_theo", float),
        	("dt0_origin", float), ("dt1_origin", float)]
	srcs_rec = np.empty((nsrcs,), dtype=dtype)
	for i in range(nsrcs):
		srcs_rec["time"][i] = times[i]
        	srcs_rec["dec"][i] = np.pi
        	srcs_rec["ra"][i] = np.pi
        	srcs_rec["dt0"][i] = -length
        	srcs_rec["dt1"][i] = length
        	srcs_rec["dt0_origin"][i] = -length
        	srcs_rec["dt1_origin"][i] = length
	return srcs_rec
	



#load models one after another to save memory
sample_names = _loader.source_list_loader()
sample_names = [sample_names[2]] # 2012-2014
source_type = "ehe" # "ehe" or "hese"


for key in sample_names:
        print("\n" + 80 * "#")
        print("# :: Setup for sample {} ::".format(key))
        #opts = _loader.settings_loader(key)[key].copy()
        opts = _loader._common_loader(key,folder="saved_test_model/settings",info="settings")[key].copy()
        print("load off data")
        exp_off = _loader.off_data_loader(key)[key]
        print("load on data")
        exp_on = _loader.on_data_loader(key)[key]
        X = np.concatenate((exp_off,exp_on))
        print("load mc")
        mc = _loader.mc_loader(source_type=source_type,names=key)[key]

        fmod = opts["model_energy_opts"].pop("flux_model")
        flux_model = flux_model_factory(fmod["model"], **fmod["args"])
        opts["model_energy_opts"]["flux_model"] = flux_model
        runlist = _loader.runlist_loader(key)[key]
        
	# give toy data
	sam_start = np.amin(X["time"])
	sam_end = np.amax(X["time"])
	sam_mid = 1./2. * (sam_start + sam_end)
	srcs_rec = give_srcs(times=[sam_mid])
	sigma_x = np.amax(X["sigma"])
	sample = give_sample(n_events=10, sigma=sigma_x, time=sam_mid)
        
	# Setup LLHs
        llhmod1 = GRBModel(#X=exp_off, MC=mc, srcs=srcs_rec, run_list=runlist,
                      spatial_opts=opts["model_spatial_opts"],
                      energy_opts=opts["model_energy_opts"],
                      time_opts = opts["model_time_opts"])
        llhmod1.fit(X=X, MC=mc, srcs=srcs_rec, run_list=runlist)
	
	values1_sig = (llhmod1._sig_bg_time(sample["time"]) *
                llhmod1._sig_spatial(sample["ra"],np.sin(sample["dec"]), sample["sigma"]))

        values1_bg = (llhmod1._sig_bg_time(sample["time"]) *
                llhmod1._bg_spatial(np.sin(sample["dec"])))

        values1 = values1_sig/values1_bg

		
	opts["model_time_opts"]["window_opt"] = "non_overlapping"
	
	llhmod2 = GRBModel(#X=exp_off, MC=mc, srcs=srcs_rec, run_list=runlist,
                      spatial_opts=opts["model_spatial_opts"],
                      energy_opts=opts["model_energy_opts"],
                      time_opts = opts["model_time_opts"])
        llhmod2.fit(X=X, MC=mc, srcs=srcs_rec, run_list=runlist)
	
	values2 = (llhmod2._soverb_time(sample["time"]) *
                llhmod2._soverb_spatial(sample["ra"], np.sin(sample["dec"]), sample["sigma"]))	
	
        #llhs[key] = GRBLLH(llh_model=llhmod, llh_opts=opts["llh_opts"])
	print(sample)
	print(X["sigma"])
	
	
	print(values1)
	print(values2)
	
	if np.array_equal(values1,values2):
		print("all values are equal (and shape)")
	else:
		print("not equal :( (or shape)")
		
        del exp_off
        del exp_on
	del X
        del mc
        gc.collect()


print("fin")


