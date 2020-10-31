"""
Energy PDFs only depend on the events, hence there is no need to make separate PDFs for signal/bg.
This script compares these two methods.
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

# lets play
dayinsec = 60.*60.*24.
dt0 = -400*dayinsec
dt1 = 400*dayinsec


#load models one after another to save memory
sample_names = _loader.source_list_loader()
sample_names = sample_names[1:] # without IC79, IC86_2011
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
        MC = _loader.mc_loader(source_type=source_type,names=key)[key]
	
	fmod = opts["model_energy_opts"].pop("flux_model")
        flux_model = flux_model_factory(fmod["model"], **fmod["args"])
        opts["model_energy_opts"]["flux_model"] = flux_model
	
	energy_opts = opts["model_energy_opts"]
	
	# build energy pdfs
        w_bg = energy_opts["mc_bg_w"]
        if w_bg is None:
            sin_dec_bg = np.sin(X["dec"])
            logE_bg = X["logE"]
        else:
            sin_dec_bg = np.sin(MC["dec"])
            logE_bg = MC["logE"]
        sin_dec_sig = np.sin(MC["dec"])
        logE_sig = MC["logE"]
        w_sig = MC["ow"] * energy_opts["flux_model"](MC["trueE"])

        _bx, _by = energy_opts["bins"]
        h_bg, _, _ = np.histogram2d(sin_dec_bg, logE_bg, weights=w_bg,
                                    bins=[_bx, _by], normed=True)
        h_sig, _, _ = np.histogram2d(sin_dec_sig, logE_sig, weights=w_sig,
                                     bins=[_bx, _by], normed=True)
	
	
        ratio = make_grid_interp_from_hist_ratio(
                h_bg=h_bg, h_sig=h_sig, bins=[_bx, _by],
                edge_fillval=energy_opts["edge_fillval"],
                interp_col_log=energy_opts["interp_col_log"],
                force_y_asc=energy_opts["force_logE_asc"]))
        bg, sig = make_grid_interp_kintscher_style_second_attempt(
                h_bg=h_bg, h_sig=h_sig, bins=[_bx, _by])



print("fin")
