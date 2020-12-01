# coding: utf-8

"""
Single job of background only trials.
Loads data and settings and builds the models, likelihoods and injectors to do
the trials with.
"""

import gc  # Manual garbage collection
import os
import json
import gzip
import argparse
import numpy as np

from tdepps.utils import make_src_records
from tdepps.grb import GRBLLH, GRBModel, MultiGRBLLH
from tdepps.grb import TimeDecDependentBGDataInjector
from tdepps.grb import MultiBGDataInjector
from tdepps.grb import GRBLLHAnalysis
import tdepps.utils.phys as phys
from _paths import PATHS
import _loader


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
dt0, dt1 = _loader.time_window_loader(tw_id)

# Load files and build the models one after another to save memory
bg_injs = {}
sig_injs = {}
llhs = {}

# Load files and build the models one after another to save memory
sample_names = _loader.source_list_loader()
sample_names = [sample_names[2]] # just 2012-2014
source_type = "ehe" # "ehe" or "hese"

all_srcs = []
for key in sample_names:
        print(type(_loader.source_list_loader(key)[key]))
        all_srcs.extend(_loader.source_list_loader(key)[key])

# just two sources to inspect overlapping area
# used source [5] against skylab
#all_srcs = [all_srcs[5]]

for key in sample_names:
    print("\n" + 80 * "#")
    print("# :: Setup for sample {} ::".format(key))
    opts = _loader.settings_loader(key)[key].copy()
    exp_off = _loader.off_data_loader(key)[key]
    exp_on = _loader.on_data_loader(key)[key]
    X_all = np.concatenate((exp_off,exp_on))
    mc = _loader.mc_loader(source_type=source_type,names=key)[key]
    # outcomment for original all_srcs
    all_srcs = _loader.source_list_loader(key)[key]
    all_srcs = [all_srcs[4],all_srcs[6]]
    runlist = _loader.runlist_loader(key)[key]
    # Process to tdepps format
    srcs_rec = phys.make_src_records_from_all_srcs(all_srcs, dt0=dt0,
						    dt1=dt1, X=X_all)

    # Setup BG injector
    bg_inj_i = TimeDecDependentBGDataInjector(inj_opts=opts["bg_inj_opts"],
                                              random_state=rndgen)
    bg_inj_i.fit(X=X_all, srcs=srcs_rec, run_list=runlist)
    bg_injs[key] = bg_inj_i

    # Setup LLH model and LLH
    fmod = opts["model_energy_opts"].pop("flux_model")
    flux_model = flux_model_factory(fmod["model"], **fmod["args"])
    opts["model_energy_opts"]["flux_model"] = flux_model
    """"
    llhmod = GRBModel(X=exp_off, MC=mc, srcs=srcs_rec, run_list=runlist,
                      spatial_opts=opts["model_spatial_opts"],
                      energy_opts=opts["model_energy_opts"])
    # fit? TypeError: __init__() got an unexpected keyword argument 'MC'
    """
    llhmod = GRBModel(#X=exp_off, MC=mc, srcs=srcs_rec, run_list=runlist,
                      spatial_opts=opts["model_spatial_opts"],
                      energy_opts=opts["model_energy_opts"],
		      time_opts = opts["model_time_opts"])
    llhmod.fit(X=X_all, MC=mc, srcs=srcs_rec, run_list=runlist)
    llhs[key] = GRBLLH(llh_model=llhmod, llh_opts=opts["llh_opts"])

    del exp_off
    del exp_on
    del mc
    gc.collect()

# Build the multi models
multi_bg_inj = MultiBGDataInjector()
multi_bg_inj.fit(bg_injs)

multi_llh_opts = _loader.settings_loader("multi_llh")["multi_llh"]
multi_llh = MultiGRBLLH(llh_opts=multi_llh_opts)
multi_llh.fit(llhs=llhs)

ana = GRBLLHAnalysis(multi_llh, multi_bg_inj, sig_inj=None)

# Do the background trials
print("Time window ID is: {}".format(tw_id))
print(":: Starting {} background trials ::".format(ntrials))
# Seed close to zero, which is close to the minimum for most cases
trials, nzeros, _ = ana.do_trials(n_trials=ntrials, n_signal=None, ns0=0.1,
                                  full_out=False)
print(":: Done ::")

# Save as JSON
outpath = os.path.join(PATHS.data, "bg_trials")
if not os.path.isdir(outpath):
    os.makedirs(outpath)

out = {"ns": trials["ns"].tolist(),
       "ts": trials["ts"].tolist(),
       "nzeros": nzeros,
       "time_window": [dt0, dt1],
       "time_window_id": tw_id,
       "rnd_seed": rnd_seed,
       "ntrials": ntrials}

fname = os.path.join(outpath, "tw_{:02d}_job_{}.json.gz".format(tw_id, job_id))
with gzip.open(fname, "w") as outfile:
    json.dump(out, fp=outfile, indent=2)
    print("Saved to:\n  {}".format(fname))
