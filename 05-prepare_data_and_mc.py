# coding: utf-8

"""
1) Data events outside the earliest / latest runs per sample are removed,
   because they don't contribute anyway here, because here the runlists were
   constructed non-overlapping.
2) Split analysis datasets into off and ontime time data for trial handling.
   The largest tested source window is split off as on-time data, the rest is
   kept as off-data to build models and injectors from.
3) Remove EHE like events identified in `04-check_ehe_mc_ids` from the
   simulation files.
4) Remove EHE events from on time data sets.
"""

import os
import json
import gzip
import numpy as np
from astropy.time import Time as astrotime

from skylab.datasets import Datasets

from _paths import PATHS
from _loader import source_list_loader, time_window_loader, runlist_loader
from myi3scripts import arr2str


def remove_ehe_from_mc(mc, eheids): # or rather identify them
    """
    Mask all values in ``mc`` that have the same run and event ID combination
    as in ``eheids``.

    Parameters
    ----------
    mc : record-array
        MC data, needs names ``'Run', 'Event'``.
    eheids : dict or record-array
        Needs names / keys ``'run_id', 'event_id``.

    Returns
    -------
    is_ehe_like : array-like, shape (len(mc),)
        Mask: ``True`` for each event in ``mc`` that is EHE like.
    """
    # Make combined IDs to easily match against EHE IDs with `np.isin`
    factor_mc = 10**np.ceil(np.log10(np.amax(mc["Event"])))
    _evids = np.atleast_1d(eheids["event_id"])
    factor_ehe = 10**np.ceil(np.log10(np.amax(_evids)))
    factor = max(factor_mc, factor_ehe)

    combined_mcids = (factor * mc["Run"] + mc["Event"]).astype(int)
    assert np.all(combined_mcids > factor)  # Is int overflow a thing here?

    _runids = np.atleast_1d(eheids["run_id"])
    combined_eheids = (factor * _runids + _evids).astype(int)
    assert np.all(combined_eheids > factor)

    # Check which MC event is tagged as EHE like
    is_ehe_like = np.in1d(combined_mcids, combined_eheids)
    print("  Found {} / {} EHE like events in MC".format(np.sum(is_ehe_like),
                                                          len(mc)))
    return is_ehe_like


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

source_type = "ehe" # "hese" or "ehe"

path_names = ["mc_no_{}".format(source_type), "my_mc_no_{}".format(source_type), "mc_no_{}_alert".format(source_type), "mc_{}_alert".format(source_type)]
local_names = ["check_{}_mc_ids".format(source_type), "check_my_{}_mc_ids".format(source_type), "check_{}_alert_mc_ids".format(source_type)]

off_data_outpath = os.path.join(PATHS.data, "data_offtime")
on_data_outpath = os.path.join(PATHS.data, "data_ontime")

mc_outpath = os.path.join(PATHS.data, "mc_no_{}".format(source_type))
mc_outpaths = []
for item in path_names:
	mc_outpaths.append(os.path.join(PATHS.data, item))

mc_unfiltered_outpath = os.path.join(PATHS.data, "mc_unfiltered")
out_paths = {"off": off_data_outpath, "on": on_data_outpath, "mc_1": mc_outpaths[0], "mc_2": mc_outpaths[1], "mc_3": mc_outpaths[2],
		"mc_unfiltered": mc_unfiltered_outpath, "mc_4": mc_outpaths[3]}
for _p in out_paths.values():
    if not os.path.isdir(_p):
        os.makedirs(_p)

# Load sources and lowest/highest lower/upper time window edge
sources = source_list_loader("all")
_dts0, _dts1 = time_window_loader("all")
dt0_min, dt1_max = np.amin(_dts0), np.amax(_dts1)

# Load runlists
runlists = runlist_loader("all")

# Load needed data and MC from PS track and add in one year of GFU sample
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

# Base MC is same for multiple samples, match names here
name2eheid_file = {
    #"IC79": "IC79.json.gz",
    "IC86_2011": "IC86_2011.json.gz",
    "IC86_2012-2014": "IC86_2012-2015.json.gz",
    "IC86_2015": "IC86_2012-2015.json.gz"
}

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

    # Remove EHE like events from MC
    is_ehe_like = []
    for item in local_names:
	_fname = os.path.join(PATHS.local, item, name2eheid_file[name]) 
    	#_fname = os.path.join(PATHS.local, "check_{}_mc_ids".format(source_type),
    	#                      name2eheid_file[name])
    	with gzip.open(_fname) as _file:
        	eheids = json.load(_file)
        	print("  Loaded EHE like MC IDs from :\n    {}".format(item))
    	is_ehe_like.append(remove_ehe_from_mc(mc, eheids))

    # Save, also in npy format
    print("  Saving on, off and non-EHE like MCs at:")
    out_arrs = {"off": exp[is_offtime], "on": exp_on, "mc_1": mc[~is_ehe_like[0]], "mc_2": mc[~is_ehe_like[1]], "mc_3": mc[~is_ehe_like[2]], "mc_unfiltered": mc, "mc_4": mc[is_ehe_like[2]]}
    for data_name in out_paths.keys():
        _fname = os.path.join(out_paths[data_name], name + ".npy")
        np.save(file=_fname, arr=out_arrs[data_name])
        print("    {:3s}: '{}'".format(data_name, _fname))
