"""
Makes rate plots, off_time and on_time with depending on how large the largest time window is.
1. Need to create different time windows which are all 'the largest'
2. Need to adjust exp_off data to every 'largest' time window
3. Calculate and plot rates for every exp_off data created
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

from astropy.time import Time as astrotime

from skylab.datasets import Datasets


from tdepps.utils import make_time_dep_dec_splines

from matplotlib import pyplot as plt

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

### load data

ps_tracks = Datasets["PointSourceTracks"]
ps_sample_names = [
                #"IC79", 
                "IC86, 2011",
                "IC86, 2012-2014"
                ]
gfu_tracks = Datasets["GFU"]
gfu_sample_names = ["IC86, 2015"]
all_sample_names = sorted(ps_sample_names
                         + gfu_sample_names
                        )


### step1

SECINMIN = 60.
SECINHOUR = 60. * SECINMIN
SECINDAY = 24. * SECINHOUR

# Time window lower and upper times relative to sourve time in seconds.
# Time windows increase logarithmically from +-1 sec to +-2.5 days.
dt = np.logspace(0, np.log10(2.5 * SECINDAY), 20 + 1)
dt = np.vstack((-dt, dt)).T


### step2

for name in all_sample_names:
    print("Working with sample {}".format(name))

    if name in ps_sample_names:
        tracks = ps_tracks
    else:
        tracks = gfu_tracks

    exp_file, mc_file = tracks.files(name)
    exp = tracks.load(exp_file)
    mc = tracks.load(mc_file)

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

    # Split data in on and off parts with the largest time window
    is_offtime = split_data_on_off(exp["time"], sources[name], dt0_min, dt1_max)

    # Remove EHE source events from ontime data
    exp_on = exp[~is_offtime]
    is_ehe_src = remove_ehe_from_on_data(exp_on, sources[name])
    exp_on = exp_on[~is_ehe_src]



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
        print("done fitting")



### step3

sin_dec_bins = llhmod._spatial_opts["sindec_bins"]
srcs = llhmod._srcs
sin_dec_splines, spl_info = make_time_dep_dec_splines(
        X=X, srcs=srcs, run_list=runlist, sin_dec_bins=sin_dec_bins,
        rate_rebins=llhmod._spatial_opts["rate_rebins"],
        spl_s=llhmod._spatial_opts["spl_s"],
        n_scan_bins=llhmod._spatial_opts["n_scan_bins"])

# to plot rate
ev_t = X["time"]
ev_sin_dec = np.sin(X["dec"])
src_t = srcs["time"]
src_trange = np.vstack((srcs["dt0"], srcs["dt1"])).T
sin_dec_bins = np.atleast_1d(sin_dec_bins)
rate_rebins = np.atleast_1d(llhmod._spatial_opts["rate_rebins"])

norm = np.diff(sin_dec_bins)
rate_rec = phys.make_rate_records(run_list=runlist, ev_runids=X["Run"])

# 1) Get phase offset from allsky fit for good amp and baseline fits.
#    Only fix the period to 1 year, as expected from seasons.
p_fix = 365.
#    Fit amp, phase and base using rebinned rates
rates, new_rate_bins, rates_std, _ = phys.rebin_rate_rec(
            rate_rec, bins=rate_rebins, ignore_zero_runs=True)
rate_bin_mids = 0.5 * (new_rate_bins[:-1] + new_rate_bins[1:])
rate_bin_err = np.ones_like(rate_bin_mids)*(0.5 * (new_rate_bins[1]-new_rate_bins[0]))

mjd = np.linspace(new_rate_bins[0],new_rate_bins[-1]+new_rate_bins[1]-new_rate_bins[0],1000)
allsky_rate = spl_info["allsky_rate_func"].fun(mjd,spl_info["allsky_best_params"])
plt.plot(mjd,allsky_rate*10**3, label="sine fit")
plt.errorbar(rate_bin_mids,rates*10**3, xerr=rate_bin_err, yerr=rates_std*10**3, fmt="", ls="none", label="exp_off_data monthly bins")
for time in src_t:
        plt.axvline(time, color="red")
plt.ylim(0,10)
plt.ylabel("Rate in mHz")
plt.xlabel("Time in MJD days")
plt.title("allsky_rate bg 2012-2014")
plt.legend(loc="best")
plt.savefig("plot_stash/allsky_rate_bg_2012-2014.pdf")
plt.clf()


