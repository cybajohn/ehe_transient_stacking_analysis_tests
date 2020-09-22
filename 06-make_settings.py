# coding: utf-8

"""
Generate setting files to build the analysis objects with, including:

1) Settings for the tdepps injectors
2) Settings for the tdepps LLH models

By storing and reading the settings we can change them easily for tests without
messing up stuff in each analysis script.
"""

import os
import json
import numpy as np

from _paths import PATHS
from _loader import source_list_loader, runlist_loader
from _loader import off_data_loader, on_data_loader, mc_loader


outpath = os.path.join(PATHS.local, "settings")
if not os.path.isdir(outpath):
    os.makedirs(outpath)

# Binning used for injector and LLH models alike
# Finer resolution around the horizon region, where we usually switch the event
# selections from northern to southern samples
hor = np.sin(np.deg2rad(30))
sd_lo, sd_hi = -1., 1.
sindec_bins = np.unique(np.concatenate([
                        np.linspace(sd_lo, -hor, 3 + 1),  # south
                        np.linspace(-hor, +hor, 14 + 1),  # horizon
                        np.linspace(+hor, sd_hi, 3 + 1),  # north
                        ]))

# Make settings for each module per sample
sample_names = source_list_loader()
sample_names = [sample_names[1],sample_names[2],sample_names[3]] # except 79
source_type = "ehe"
for key in sample_names:
    print("Building settings file for sample '{}'".format(key))
    # Load data that settings depend on
    srcs = source_list_loader(key)[key] # not needed
    runlist = runlist_loader(key)[key] # not needed
    exp_off = off_data_loader(key)[key]
    exp_on = on_data_loader(key)[key]
    exp_off = np.concatenate((exp_off, exp_on))
    mc = mc_loader(source_type, key)[key]

    # :: BG injector ::
    # Rebinning for the rate model fits, use monthly bins
    if key == "IC86_2012-2014":
        n_rate_bins = 36
    else:
        n_rate_bins = 12
    rate_rebins = np.linspace(np.amin(exp_off["time"]),
                              np.amax(exp_off["time"]), n_rate_bins)
    # We want the parameter spline to stick a little bit to the points
    spl_s = len(sindec_bins) / 2
    bg_inj_opts = {
        "sindec_bins": sindec_bins.tolist(),  # JSON only saves lists
        "rate_rebins": rate_rebins.tolist(),
        "spl_s": spl_s,
        "n_scan_bins": 25,
        "n_data_evts_min": 100,
        }

    # :: Signal injector ::
    # Use a power law flux with somewhat generic gamma=2. 'model' must be a
    # method from `tdepps.utils.phys`, 'args' are the method parameters.
    gamma = 3.7
    E0 = 1.    # Reference energy in GeV
    phi0 = 1.  # This doesn't matter here, but set it anyway
    flux_model_opts = {
        "model": "power_law_flux",
        "args": {"gamma": gamma, "E0": E0, "phi0": phi0}
        }
    # Inject full sky, select events from a 2° band around each source
    sig_inj_opts = {
        "dec_range": [np.arcsin(sd_lo), np.arcsin(sd_hi)],
        "mode": "band",
        "sindec_inj_width": np.sin(np.deg2rad(2.)),
        "flux_model": flux_model_opts,
        }

    # :: LLH model ::
    # Use same settings as in injector. Use kent as signal PDF, discard events
    # more than 5sigma away from each source in LLH calculation
    model_spatial_opts = {
        "sindec_bins": sindec_bins.tolist(),
        "rate_rebins": rate_rebins.tolist(),
        "spl_s": spl_s,
        "n_scan_bins": 25,
        "kent": True,
        "select_ev_sigma": 5.,
        "n_mc_evts_min": 500,
        }
    # This is kind of arbitrary, but seems to produce robust PDFs
    logE_bins = np.linspace(
        np.floor(np.amin(mc["logE"])), np.ceil(np.amax(mc["logE"])), 30)
    model_energy_opts = {
        "bins": [sindec_bins.tolist(), logE_bins.tolist()],
        "flux_model": flux_model_opts,
        # Use data to estimtate bg energy distribution
        "mc_bg_w": None,
        # Force the energy PDF to be ascending, as expected from gamma=2
        "force_logE_asc": True,
        # Make a conservative choice for missing vals
        "edge_fillval": "minmax_col",
        # Technical: Interpolate missing vals per column in normal space
        "interp_col_log": False,
        }
    # Set time options: "non_overlapping" for old model,
    # "overlapping" for the new model
    # Todo: Maybe add time dependent scenarios 
    model_time_opts = {
	"window_opt": "overlapping"
	}

    # :: LLH ::
    llh_opts = {
        # Discard events when signal-bg-ratio < 0.001
        "sob_abs_eps": 0.001,
        "sob_rel_eps": 0,
        # Minimizer opts for the single LLHs are not used, but set them anyway
        "minimizer": "L-BFGS-B",
        "minimizer_opts": {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000},
        "ns_bounds": [0.0, None],
        }

    # :: Save settings per sample ::
    settings = {
        "bg_inj_opts": bg_inj_opts,
        "sig_inj_opts": sig_inj_opts,
        "model_spatial_opts": model_spatial_opts,
        "model_energy_opts": model_energy_opts,
	"model_time_opts": model_time_opts,
        "llh_opts": llh_opts,
        }

    fname = os.path.join(outpath, key + ".json")
    with open(fname, "w") as outf:
        json.dump(obj=settings, fp=outf, indent=2)
        print("  Saved to:\n    {}".format(fname))

# :: Multi LLH ::
# Options for the multi LLH minimization get saved extra
print("Building settings file for the multi LLH module")
multi_llh_opts = {
    "minimizer": "L-BFGS-B",
    "minimizer_opts": {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000},
    "ns_bounds": [0.0, None],
    }

fname = os.path.join(outpath, "multi_llh.json")
with open(fname, "w") as outf:
    json.dump(obj=multi_llh_opts, fp=outf, indent=2)
    print("  Saved to:\n    {}".format(fname))
