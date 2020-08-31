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

SECINDAY = 60. * 60. * 24.

dt = np.logspace(np.log10(1 * SECINDAY), np.log10(400 * SECINDAY), 21)

dt0 = -dt
dt1 = dt

#load models one after another to save memory
sample_names = _loader.source_list_loader()
sample_names = [sample_names[2]] # just IC86_2012-2014
source_type = "ehe" # "ehe" or "hese"

all_srcs = []
for key in sample_names:
        print(type(_loader.source_list_loader(key)[key]))
        all_srcs.extend(_loader.source_list_loader(key)[key])

weight_dict = {"theo": [], "time": [], "dec": [], "weight": [], "window": []}

for key in sample_names:
        print("\n" + 80 * "#")
        print("# :: Setup for sample {} ::".format(key))
        opts = _loader.settings_loader(key)[key].copy()
        print("load off data")
        exp_off = _loader.off_data_loader(key)[key]
        print("load on data")
        exp_on = _loader.on_data_loader(key)[key]
        print("load mc")
        mc = _loader.mc_loader(source_type=source_type,names=key)[key]
        print("load srcs")
        srcs = _loader.source_list_loader(key)[key]
        print("load runlist")
        runlist = _loader.runlist_loader(key)[key]
	print("setup ops")
        fmod = opts["model_energy_opts"].pop("flux_model")
        flux_model = flux_model_factory(fmod["model"], **fmod["args"])
        opts["model_energy_opts"]["flux_model"] = flux_model

	for i,dt in enumerate(dt0):
        	# Process to tdepps format
        	#srcs_rec = make_src_records(srcs, dt0=dt0, dt1=dt1)
        	srcs_rec = phys.make_src_records_from_all_srcs(all_srcs, dt0=dt, dt1=dt1[i],
                	                                X=np.concatenate((exp_off,exp_on)))

	        # Setup LLH model and LLH
		print("make grbmodel")
        	llhmod = GRBModel(#X=exp_off, MC=mc, srcs=srcs_rec, run_list=runlist,
                	          spatial_opts=opts["model_spatial_opts"],
                          	  energy_opts=opts["model_energy_opts"],
                          	  time_opts = opts["model_time_opts"])
        	print("fit model")
        	llhmod.fit(X=np.concatenate((exp_off,exp_on)), MC=mc, srcs=srcs_rec, run_list=runlist)

		theo = llhmod._llh_args["src_w_theo"]
		dec = llhmod._llh_args["src_w_dec"]
	        time = llhmod._llh_args["src_w_time"]
		weights = theo*dec*time
		weights = weights/np.sum(weights)
		weight_dict["time"].append(list(time))
		weight_dict["dec"].append(list(dec))
		weight_dict["theo"].append(list(theo))
		weight_dict["weight"].append(list(weights))
		weight_dict["window"].append(dt*2.)
with open("plot_stash/data/source_weights.json", "w") as outf:
        json.dump(obj=dict(weight_dict), fp=outf, indent=2)


print(weight_dict)
print("fin")

