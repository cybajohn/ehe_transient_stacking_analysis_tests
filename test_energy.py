import gc  # Manual garbage collection
import os
import json
import gzip
import argparse
import numpy as np
import sys

from tdepps.utils import make_src_records, make_grid_interp_from_hist_ratio, make_grid_interp_kintscher_style_second_attempt
from tdepps.utils import fill_dict_defaults
from tdepps.grb import GRBLLH, GRBModel, MultiGRBLLH
from tdepps.grb import TimeDecDependentBGDataInjector
from tdepps.grb import MultiBGDataInjector
from tdepps.grb import GRBLLHAnalysis
import tdepps.utils.phys as phys
from _paths import PATHS
import _loader

from myi3scripts import arr2str

from astropy.time import Time as astrotime

from skylab.datasets import Datasets


from tdepps.utils import make_time_dep_dec_splines

from matplotlib import pyplot as plt

print("hello world")

### functions

def sec2timestr(sec):
    sec = np.around(sec, decimals=3)
    d = int(sec / SECINDAY)
    sec -= d * SECINDAY
    h = int(sec / SECINHOUR)
    sec -= h * SECINHOUR
    m = int(sec / SECINMIN)
    s = sec - m * SECINMIN
    return "{:d}d : {:02d}h : {:02d}m : {:06.3f}s".format(d, h, m, s)

def remove_ehe_from_on_data(exp_on, src_dicts):
    """
    Mask source events in ontime data sample.

    Parameters
    ----------
    exp_on : record-array
        Ontime data, needs names ``'Run', 'Event'``.

    src_dicts : list of dicts
        One dict per source, must have keys ``'run_id', 'event_id'``.

    Returns
    -------
    is_ehe_src : array-like, shape (len(exp_on),)
        Mask: ``True`` where a source EHE event is in the sample.
    """
    is_ehe_src = np.zeros(len(exp_on), dtype=bool)
    # Check each source and combine masks
    print("  EHE events in ontime data:")
    for i, src in enumerate(src_dicts):
        mask_i = ((src["event_id"] == exp_on["Event"]) &
                  (src["run_id"] == exp_on["Run"]))
        is_ehe_src = np.logical_or(is_ehe_src, mask_i)
        print("  - Source {}: {}. Dec: {:.2f} deg. logE: {} log(GeV)".format(
            i, np.sum(mask_i), np.rad2deg(src["dec"]), exp_on[mask_i]["logE"]))
    return is_ehe_src

def split_data_on_off(ev_t, src_dicts, dt0, dt1):
    """
    Returns a mask to split experimental data in on and off source regions.

    Parameters
    ----------
    ev_t : array-like
        Experimental times in MJD days.
    src_dicts : list of dicts
        One dict per source, must have key ``'mjd'``.
    dt0 : float
        Left border of ontime window for all sources in seconds relative to the
        source times, should be negative for earlier times.
    dt1 : float
        Right border of ontime window for all sources in seconds relative to the
        source times.

    Returns
    -------
    offtime : array-like, shape (len(exp),)
        Mask: ``True`` when event in ``exp`` is in the off data region.
    """
    SECINDAY = 24. * 60. * 60.
    nevts, nsrcs = len(ev_t), len(src_dicts)

    dt0_mjd = np.empty(nsrcs, dtype=float)
    dt1_mjd = np.empty(nsrcs, dtype=float)
    for i, src in enumerate(src_dicts):
        dt0_mjd[i] = src["mjd"] + dt0 / SECINDAY
        dt1_mjd[i] = src["mjd"] + dt1 / SECINDAY

    # Broadcast to test every source at once
    ev_t = np.atleast_1d(ev_t)[None, :]
    dt0_mjd, dt1_mjd = dt0_mjd[:, None], dt1_mjd[:, None]

    ontime = np.logical_and(ev_t >= dt0_mjd, ev_t <= dt1_mjd)
    offtime = np.logical_not(np.any(ontime, axis=0))
    assert np.sum(np.any(ontime, axis=0)) + np.sum(offtime) == nevts

    print("  Ontime window duration: {:.2f} sec".format(dt1 - dt0))
    print("  Ontime events: {} / {}".format(np.sum(ontime), nevts))
    for i, on_per_src in enumerate(ontime):
        print("  - Source {}: {} on time".format(i, np.sum(on_per_src)))
    return offtime

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


source_type="ehe"

### load data, lets just have 2012-2014 for starters

ps_tracks = Datasets["PointSourceTracks"]
ps_sample_names = [
                #"IC79", 
                #"IC86, 2011",
                "IC86, 2012-2014"
                ]
#gfu_tracks = Datasets["GFU"]
#gfu_sample_names = ["IC86, 2015"]
all_sample_names = sorted(ps_sample_names
#                         + gfu_sample_names
                        )

sources = _loader.source_list_loader("all")
runlists = _loader.runlist_loader("all")


### step1

SECINMIN = 60.
SECINHOUR = 60. * SECINMIN
SECINDAY = 24. * SECINHOUR

# typo incoming... Can you spot it?
# Time window lower and upper times relative to sourve time in seconds.
# Time windows increase logarithmically from +-1 sec to +-2.5 days.
# 20+1 steps were default
dt = np.logspace(0, np.log10(15 * SECINDAY), 4)
dt = [5 * SECINDAY]


for name in all_sample_names:
    print("Working with sample {}".format(name))

    if name in ps_sample_names:
        tracks = ps_tracks
    else:
        tracks = gfu_tracks

    exp_file, mc_file = tracks.files(name)
    exp = tracks.load(exp_file)
    #mc = tracks.load(mc_file)

    print("  Loaded {} track sample from skylab:".format(
        "PS" if name in ps_sample_names else "GFU"))
    _info = arr2str(exp_file if isinstance(exp_file, list) else [exp_file],
                    sep="\n    ")
    print("    Data:\n      {}".format(_info))
    print("    MC  :\n      {}".format(mc_file))

    name = name.replace(", ", "_")

    # Remove events before first and after last run per sample
    first_run = min(map(lambda d: astrotime(d["good_tstart"],
                                            format="iso").mjd, runlists[name]))
    last_run = max(map(lambda d: astrotime(d["good_tstop"],
                                           format="iso").mjd, runlists[name]))
    is_inside_runs = (exp["time"] >= first_run) & (exp["time"] <= last_run)
    print("  Removing {} / {} events outside runs.".format(
        np.sum(~is_inside_runs), len(exp)))
    exp = exp[is_inside_runs]

    MC = _loader.mc_loader(source_type=source_type,names=name)[name]


    # or here for loop:?
    for time_window in dt:
        # Split data in on and off parts with the largest time window
        is_offtime = split_data_on_off(exp["time"], sources[name], -time_window, time_window)

        # Remove EHE source events from ontime data
        exp_on = exp[~is_offtime]
        is_ehe_src = remove_ehe_from_on_data(exp_on, sources[name])
        exp_on = exp_on[~is_ehe_src]
        ### step3 
        opts = _loader._common_loader(name,folder="saved_test_model/settings",info="settings")[name].copy()

        # Check and setup spatial PDF options # from llh_model
        spatial_opts=opts["model_spatial_opts"]
        req_keys = ["sindec_bins", "rate_rebins"]
        opt_keys = {"select_ev_sigma": 5., "spl_s": None, "n_scan_bins": 50,
                            "kent": True, "n_mc_evts_min": 500}
        spatial_opts = fill_dict_defaults(spatial_opts, req_keys, opt_keys)

        spatial_opts["sindec_bins"] = np.atleast_1d(spatial_opts["sindec_bins"])
        spatial_opts["rate_rebins"] = np.atleast_1d(spatial_opts["rate_rebins"])

        sin_dec_bins = spatial_opts["sindec_bins"]

	# Check and setup energy PDF options
	energy_opts = opts["model_energy_opts"]
	
	fmod = opts["model_energy_opts"].pop("flux_model")
        flux_model = flux_model_factory(fmod["model"], **fmod["args"])
        opts["model_energy_opts"]["flux_model"] = flux_model

        req_keys = ["bins", "flux_model"]
        opt_keys = {"mc_bg_w": None, "force_logE_asc": True,
                    "edge_fillval": "minmax_col", "interp_col_log": False}
        energy_opts = fill_dict_defaults(energy_opts, req_keys, opt_keys)

        sin_dec_bins, logE_bins = map(np.atleast_1d, energy_opts["bins"])
        energy_opts["bins"] = [sin_dec_bins, logE_bins]

        try:
            energy_opts["flux_model"](1.)  # Standard units are E = 1GeV
        except Exception:
            raise TypeError("'flux_model' must be a function `f(trueE)`.")

        if (energy_opts["edge_fillval"] not in
                ["minmax", "col", "minmax_col", "min"]):
            raise ValueError("'edge_fillval' must be one of " +
                             "['minmax'|'col'|'minmax_col'|'min'].")

        if energy_opts["edge_fillval"] == "min" and energy_opts["force_y_asc"]:
            raise ValueError("`edge_fillval` is 'min' and 'force_y_asc' is " +
                             "`True`, which doesn't make sense together.")

        if len(energy_opts["bins"]) != 2:
            raise ValueError("Bins for energy hist must be of format " +
                             "`[sin_dec_bins, logE_bins]`.")

        if np.any(sin_dec_bins < -1.) or np.any(sin_dec_bins > 1.):
            raise ValueError("sinDec declination bins for energy hist not " +
                             "in valid range `[-1, 1]`.")

        if energy_opts["mc_bg_w"] is not None:
            energy_opts["mc_bg_w"] = np.atleast_1d(energy_opts["mc_bg_w"])

	
        # need to make source records I suppose
        srcs = sources[name]
        exp_off = exp[is_offtime]
        srcs = make_src_records(srcs, dt0=-time_window, dt1=time_window)
	"""
        sin_dec_splines, spl_info = make_time_dep_dec_splines(
                X=exp_off, srcs=srcs, run_list=runlists[name], sin_dec_bins=sin_dec_bins,
                rate_rebins=spatial_opts["rate_rebins"],
                spl_s=spatial_opts["spl_s"],
                n_scan_bins=spatial_opts["n_scan_bins"],
                ignore_zero_runs=False)
	"""
	X = exp_off
	w_bg = energy_opts["mc_bg_w"]
	if w_bg is None:
		print("data")
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

	plt.hist2d(sin_dec_bg, logE_bg, weights=w_bg, bins=[_bx, _by], normed=True, cmap=plt.cm.get_cmap("OrRd"))
	plt.ylabel(r'Data $\log(E)$')
	plt.xlabel(r'Data $\sin(\delta)$')
	plt.colorbar()
	plt.savefig("plot_stash/bg_sindec_energy_2dhist.pdf")
	plt.clf()
	
	plt.hist2d(sin_dec_sig, logE_sig, weights=w_sig, bins=[_bx, _by], normed=True, cmap=plt.cm.get_cmap("OrRd"))
	plt.ylabel(r'MC $\log(E)$')
	plt.xlabel(r'MC $\sin(\delta)$')
	plt.colorbar()
	plt.savefig("plot_stash/sig_sindec_energy_2dhist.pdf")
	plt.clf()

	bg, sig, interp_info = make_grid_interp_kintscher_style_second_attempt(h_bg, h_sig, bins = [_bx, _by], give_info=True)
	print("info: ", interp_info)
	
	print("x_bins: ",_bx)
	print("y_bins: ",_by)
	sig_energy_list = np.zeros(shape=(40,40))
	x = np.linspace(_bx[0],_bx[-1],40)
	y = np.linspace(_by[0],_by[-1],40)
	for _x,i in enumerate(x):
		for _y,j in enumerate(y):
			print(i,j)
			sig_energy_list[_x][_y] = sig([i,j])

	np.savetxt("plot_stash/data/sig_energy_spline.txt",sig_energy_list, delimiter=",")
	
	bg_energy_list = np.zeros(shape=(40,40))
	for _x,i in enumerate(x):
                for _y,j in enumerate(y):
                        print(i,j)
                        bg_energy_list[_x][_y] = bg([i,j])

        np.savetxt("plot_stash/data/bg_energy_spline.txt",bg_energy_list, delimiter=",")


