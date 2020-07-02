import numpy as np
import _loader

sample_start_endtime = np.array([[2,8],[12,18],[22,28]])
sample_names = _loader.source_list_loader()
sample_names = sample_names[1:]
for name in sample_names:
	exp_off = _loader.off_data_loader(name)[name]
	print(name)
	print(np.amin(exp_off["time"]))
	print(np.amax(exp_off["time"]))



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

parser = argparse.ArgumentParser(description="ehe_stacking")
parser.add_argument("--rnd_seed", type=int)
parser.add_argument("--ntrials", type=int)
parser.add_argument("--job_id", type=str)
parser.add_argument("--tw_id", type=int)
args = parser.parse_args()
rnd_seed = args.rnd_seed
ntrials = args.ntrials
job_id = args.job_id
tw_id = args.tw_id

rndgen = np.random.RandomState(rnd_seed)
# just one tw
dt0, dt1 = _loader.time_window_loader(19)

# Load files and build the models one after another to save memory
bg_injs = {}
sig_injs = {}
llhs = {}




#load models one after another to save memory
sample_names = _loader.source_list_loader()
sample_names = sample_names[2:] # without IC79, IC86_2011
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
        print("load mc")
        mc = _loader.mc_loader(source_type=source_type,names=key)[key]
        print("load srcs")
        srcs = _loader.source_list_loader(key)[key]
        print("load runlist")
        runlist = _loader.runlist_loader(key)[key]
        # Process to tdepps format
        srcs_rec = make_src_records(srcs, dt0=dt0, dt1=dt1)

        # Setup BG injector
        print("setup bg injector")
        bg_inj_i = TimeDecDependentBGDataInjector(inj_opts=opts["bg_inj_opts"],
                                                      random_state=rndgen)
        #print("fit bg injector")
        #bg_inj_i.fit(X=exp_off, srcs=srcs_rec, run_list=runlist)
        #bg_injs[key] = bg_inj_i

        # Setup LLH model and LLH
        print("setup ops")
        fmod = opts["model_energy_opts"].pop("flux_model")
        flux_model = flux_model_factory(fmod["model"], **fmod["args"])
        opts["model_energy_opts"]["flux_model"] = flux_model
        """"
        llhmod = GRBModel(X=exp_off, MC=mc, srcs=srcs_rec, run_list=runlist,
                          spatial_opts=opts["model_spatial_opts"],
                          energy_opts=opts["model_energy_opts"])
        # fit? TypeError: __init__() got an unexpected keyword argument 'MC'
        """
        print("make grbmodel")
        llhmod = GRBModel(#X=exp_off, MC=mc, srcs=srcs_rec, run_list=runlist,
                          spatial_opts=opts["model_spatial_opts"],
                          energy_opts=opts["model_energy_opts"],
                          time_opts = opts["model_time_opts"])
        print("fit model")
        llhmod.fit(X=exp_off, MC=mc, srcs=srcs_rec, run_list=runlist)
	llhs[key] = GRBLLH(llh_model=llhmod, llh_opts=opts["llh_opts"])

# Build the multi models
multi_bg_inj = MultiBGDataInjector()
multi_bg_inj.fit(bg_injs)

multi_llh_opts = _loader.settings_loader("multi_llh")["multi_llh"]
multi_llh = MultiGRBLLH(llh_opts=multi_llh_opts)
multi_llh.fit(llhs=llhs)

print(multi_llh.model)

# Following code is how the sample weights will be calculated
# It is using the interval_overlap function from utils/misc.py
secinday = 24. * 60. * 60.

source_times = []
dataset_times_start = []
dataset_times_stop = []
for key, model in multi_llh.model.items():
	for k, time in enumerate(model.srcs["time"]):
		start = model.srcs["dt0"][k]/secinday + time
		stop = model.srcs["dt1"][k]/secinday + time
		source_times.append([start,stop])
	dataset_times_start.append(model.dataset_bounds[0])
	dataset_times_stop.append(model.dataset_bounds[1])


weights = np.zeros(shape=(len(source_times),len(dataset_times_start)))

for k, src_time in enumerate(source_times):	
	overlap = interval_overlap(src_time[0],src_time[1], dataset_times_start, dataset_times_stop)
	weights[k] = overlap/np.sum(overlap)
print(weights)
print(np.sum(weights,axis=0))
print("finished")

