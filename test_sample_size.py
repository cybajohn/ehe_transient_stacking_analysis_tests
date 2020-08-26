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

from tqdm import tqdm

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

sample_lambdas = {sample_names[0]:[],sample_names[1]:[],sample_names[2]:[]}


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
	sample_lambdas[key].append(bg_inj_i._nb)
        bg_injs[key] = bg_inj_i


for key,item in sample_lambdas.items():
	np.savetxt("plot_stash/data/sample_{}_lambdas.txt".format(key),item, delimiter=",")

# Build the multi models
multi_bg_inj = MultiBGDataInjector()
multi_bg_inj.fit(bg_injs)
sample_sizes = {sample_names[0]:[],sample_names[1]:[],sample_names[2]:[]}

for i in tqdm(range(1000)):
        bg_sample = multi_bg_inj.sample()
	for name in sample_names:
		sample_sizes[name].append(len(bg_sample[name]))

for key,item in sample_sizes.items():
        np.savetxt("plot_stash/data/sample_{}_sizes.txt".format(key),item, delimiter=",")

print("fin")

plt.hist(sample_sizes[sample_names[1]],bins=10, label=sample_names[1])
plt.legend(loc='best')
plt.savefig("plot_stash/sampling/sample_size.pdf")

plt.clf()
