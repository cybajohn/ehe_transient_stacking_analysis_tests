"""
script to test the model, check pfds etc.
single bg trial to be fair, nothing special really
"""



import gc  # Manual garbage collection
import os
import json
import gzip
import argparse
import numpy as np
import sys

from tdepps.utils import make_src_records, make_grid_interp_from_hist_ratio, make_grid_interp_from_hist
from tdepps.grb import GRBLLH, GRBModel, MultiGRBLLH
from tdepps.grb import TimeDecDependentBGDataInjector
from tdepps.grb import MultiBGDataInjector
from tdepps.grb import GRBLLHAnalysis
import tdepps.utils.phys as phys
from _paths import PATHS
import _loader

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
sample_names = [sample_names[2]] # just 2012-2014
source_type = "ehe" # "ehe" or "hese"
for key in sample_names:
	print("\n" + 80 * "#")
	print("# :: Setup for sample {} ::".format(key))
	#opts = _loader.settings_loader(key)[key].copy()
	opts = _loader._common_loader(key,folder="saved_test_model/settings",info="settings")[key].copy()
	print("load off data")
	exp_off = _loader.off_data_loader(key)[key]
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
	print("done fitting")



print(llhmod._spl_info["mc_sin_dec_pdf_spline"])

spline = llhmod._spl_info["mc_sin_dec_pdf_spline"]

x = np.linspace(-1,1,1000)

plt.plot(x,spline(x))
plt.savefig("plot_stash/llh_model_mc_sin_dec_spline.pdf")
plt.clf()

bg_energy_spline = llhmod._energy_interpol[0]
sig_energy_spline = llhmod._energy_interpol[1]

print(type(llhmod))
saved_model = llhmod.__dict__
print(saved_model)

print(sig_energy_spline)
print("test point: ",sig_energy_spline([1,4]))

x = np.linspace(-0.9,0.9,100)
y = np.linspace(2.1,8.9,100)

points = np.array([x,y]).T

"""
for x_val in x:
	for y_val in y:
		#print(x_val,y_val, bg_energy_spline([x_val,y_val]))
		plt.scatter(x_val,y_val,bg_energy_spline([x_val,y_val])[0])
plt.savefig("plot_stash/llh_model_bg_energy_spline.pdf")
plt.clf()

for x_val in x:
	for y_val in y:
		plt.scatter(x_val,y_val, x_val+y_val)
plt.savefig("plot_stash/test_scatter.pdf")
plt.clf()
"""


sys.path.append("../")


from my_modules import plot_weighted_hist_mc_no_ehe, plot_weights, load_sources, load_mc_data, plot_energy_dec_hist

def give_source_dec(sources):
        sources_dec = {}
        for key in sources.keys():
                source_list = []
                for source in sources[key]:
                        source_list.append(source["dec"])
                sources_dec.update({key:source_list})
        return sources_dec

names = ["IC86_2011", "IC86_2012-2014", "IC86_2015"]
test_path = "../../../../data/user/jkollek/tests_for_ehe_filter/rawout_tests/"
source_path = "out_test/source_list/source_list.json"

print("loading data")
mc = load_mc_data(names, test_path + "mc_no_ehe_alert")
MC = mc[names[1]]
#source_pos = {list(source_pos.keys())[1]:list(source_pos.values())[1]} # just 2012-2014

# X is exp_off, just so you know
X = exp_off
w_bg = llhmod._energy_opts["mc_bg_w"]
if w_bg is None:
	sin_dec_bg = np.sin(X["dec"])
        logE_bg = X["logE"]
else:
        sin_dec_bg = np.sin(MC["dec"])
        logE_bg = MC["logE"]
sin_dec_sig = np.sin(MC["dec"])
logE_sig = MC["logE"]
w_sig = MC["ow"] * llhmod._energy_opts["flux_model"](MC["trueE"])

_bx, _by = llhmod._energy_opts["bins"]
h_bg, _, _ = np.histogram2d(sin_dec_bg, logE_bg, weights=w_bg,
                            bins=[_bx, _by], normed=True)
h_sig, _, _ = np.histogram2d(sin_dec_sig, logE_sig, weights=w_sig,
                            bins=[_bx, _by], normed=True)

hists = [h_bg, h_sig]
energy_interpol = []
if llhmod._time_opts["window_opt"] == "non_overlapping":
	#src_w_time = np.ones_like(srcs["w_theo"])
        energy_interpol.append(make_grid_interp_from_hist_ratio(
                h_bg=h_bg, h_sig=h_sig, bins=[_bx, _by],
                edge_fillval=llhmod._energy_opts["edge_fillval"],
                interp_col_log=llhmod._energy_opts["interp_col_log"],
                force_y_asc=llhmod._energy_opts["force_logE_asc"]))
elif llhmod._time_opts["window_opt"] == "overlapping":
        #src_w_time = np.sum(self._soverb_time(X["time"]), axis=1)
        for hist in hists:
	       energy_interpol.append(make_grid_interp_from_hist(
                    hist=hist, bins=[_bx, _by],
                    edge_fillval=llhmod._energy_opts["edge_fillval"],
                    interp_col_log=llhmod._energy_opts["interp_col_log"],
                    force_y_asc=llhmod._energy_opts["force_logE_asc"]))






plt.hist2d(sin_dec_bg, logE_bg, weights=w_bg, bins=[_bx, _by], normed=True, cmap=plt.cm.get_cmap("OrRd"))
plt.colorbar()
plt.savefig("plot_stash/bg_sindec_energy_2dhist.pdf")
plt.clf()

plt.hist2d(sin_dec_sig, logE_sig, weights=w_sig, bins=[_bx, _by], normed=True, cmap=plt.cm.get_cmap("OrRd"))
plt.colorbar()
plt.savefig("plot_stash/sig_sindec_energy_2dhist.pdf")
plt.clf()
