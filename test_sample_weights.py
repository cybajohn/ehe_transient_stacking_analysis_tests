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

# lets play
dayinsec = 60.*60.*24.
dt0 = -400*dayinsec
dt1 = 400*dayinsec

# Load files and build the models one after another to save memory
bg_injs = {}
sig_injs = {}
llhs = {}




#load models one after another to save memory
sample_names = _loader.source_list_loader()
sample_names = sample_names[1:] # without IC79, IC86_2011
source_type = "ehe" # "ehe" or "hese"

all_srcs = []
for key in sample_names:
	print(type(_loader.source_list_loader(key)[key]))
	all_srcs.extend(_loader.source_list_loader(key)[key])

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
        #srcs_rec = make_src_records(srcs, dt0=dt0, dt1=dt1)
	srcs_rec = phys.make_src_records_from_all_srcs(all_srcs, dt0=dt0, dt1=dt1, 
						X=np.concatenate((exp_off,exp_on)))
        # Setup BG injector
        print("setup bg injector")
        bg_inj_i = TimeDecDependentBGDataInjector(inj_opts=opts["bg_inj_opts"],
                                                      random_state=rndgen)
        print("fit bg injector")
        bg_inj_i.fit(X=np.concatenate((exp_off,exp_on)), srcs=srcs_rec, run_list=runlist)
        bg_injs[key] = bg_inj_i

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
        llhmod.fit(X=np.concatenate((exp_off,exp_on)), MC=mc, srcs=srcs_rec, run_list=runlist)
	llhs[key] = GRBLLH(llh_model=llhmod, llh_opts=opts["llh_opts"])

# Build the multi models
multi_bg_inj = MultiBGDataInjector()
multi_bg_inj.fit(bg_injs)

multi_llh_opts = _loader.settings_loader("multi_llh")["multi_llh"]
multi_llh = MultiGRBLLH(llh_opts=multi_llh_opts)
multi_llh.fit(llhs=llhs)
for i in range(20):
	bg_sample = multi_bg_inj.sample()
	print(bg_sample)
print(multi_llh.model)

ana = GRBLLHAnalysis(multi_llh, multi_bg_inj, sig_inj=None)

# Do the background trials
# Seed close to zero, which is close to the minimum for most cases
ntrials = 100
print("test_trials")
trials, nzeros, _ = ana.do_trials(n_trials=ntrials, n_signal=None, ns0=0.1,
                                  full_out=False)

print("trials: ", trials)
print("nzeros: ", nzeros)

# Following code is how the sample weights will be calculated
# It is using the interval_overlap function from utils/misc.py
secinday = 24. * 60. * 60.

source_times = []
dataset_times_start = []
dataset_times_stop = []
for key, model in multi_llh.model.items():
	j_start, j_stop = model.dataset_bounds
	for k, time in enumerate(model.srcs["time"]):
		is_in_sample = (time < j_stop) and (time > j_start)
		if is_in_sample:
			start = model.srcs["dt0_origin"][k]/secinday + time
			stop = model.srcs["dt1_origin"][k]/secinday + time
			source_times.append([start,stop])
	dataset_times_start.append(j_start)
	dataset_times_stop.append(j_stop)


weights = np.zeros(shape=(len(source_times),len(dataset_times_start)))

for k, src_time in enumerate(source_times):	
	overlap = interval_overlap(src_time[0],src_time[1], dataset_times_start, dataset_times_stop)
	weights[k] = overlap/np.sum(overlap)
print(weights)
print(np.sum(weights,axis=0))
print("finished")

print(multi_llh._ns_weights)
print(dt0,dt1)

