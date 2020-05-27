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
"""
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
"""
# lambda background part
# Get sindec rate spline for each source, averaged over its time window
print(llhmod._log.INFO("Create time dep sindec splines."))
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
rate_bin_mids_off = rate_bin_mids
rates_off = rates
rates_std_off = rates_std
print("rate_rec: ",rate_rec)
print("rate: ", rates)
print(len(rates))
print(rate_bin_mids)
print(len(rate_bin_mids))
print(rates_std)
print("exp_off")
print(exp_off)
print(len(exp_off))


# Step 2: Cache fixed LLH args
# Cache expected nb for each source from allsky rate func integral
src_t = srcs["time"]
src_trange = np.vstack((srcs["dt0"], srcs["dt1"])).T
nb = spl_info["allsky_rate_func"].integral(
        src_t, src_trange, spl_info["allsky_best_params"])
assert len(nb) == len(src_t)
print(nb)
print(len(nb))
print(type(spl_info["allsky_rate_func"]))
print("params: ",spl_info["allsky_best_params"])
print(spl_info["allsky_rate_func"].fun(src_t,spl_info["allsky_best_params"]))
print(src_t)

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


allsky_off = allsky_rate
on_and_off_data = np.concatenate((exp_on,exp_off))
print("exp_on: ", len(exp_on))
print("all: ", len(on_and_off_data))
# for convinience
X = on_and_off_data

sin_dec_splines, spl_info = make_time_dep_dec_splines(
        X=X, srcs=srcs, run_list=runlist, sin_dec_bins=sin_dec_bins,
        rate_rebins=llhmod._spatial_opts["rate_rebins"],
        spl_s=llhmod._spatial_opts["spl_s"],
        n_scan_bins=llhmod._spatial_opts["n_scan_bins"])

ev_t = X["time"]
ev_sin_dec = np.sin(X["dec"])
src_t = srcs["time"]
src_trange = np.vstack((srcs["dt0"], srcs["dt1"])).T
sin_dec_bins = np.atleast_1d(sin_dec_bins)
rate_rebins = np.atleast_1d(llhmod._spatial_opts["rate_rebins"])

norm = np.diff(sin_dec_bins)
rate_rec = phys.make_rate_records(run_list=runlist, ev_runids=X["Run"])

rates, new_rate_bins, rates_std, _ = phys.rebin_rate_rec(
            rate_rec, bins=rate_rebins, ignore_zero_runs=True)
rate_bin_mids = 0.5 * (new_rate_bins[:-1] + new_rate_bins[1:])

allsky_rate = spl_info["allsky_rate_func"].fun(mjd,spl_info["allsky_best_params"])
plt.plot(mjd,allsky_rate*10**3, label="sine fit")
plt.errorbar(rate_bin_mids,rates*10**3, xerr=rate_bin_err, yerr=rates_std*10**3, fmt="", ls="none", label="exp_off_and_on_data monthly bins")
for time in src_t:
	plt.axvline(time, color="red")
plt.ylim(0,10)
plt.ylabel("Rate in mHz")
plt.xlabel("Time in MJD days")
plt.title("allsky_rate bg 2012-2014 on and off data")
plt.legend(loc="best")
plt.savefig("plot_stash/allsky_rate_bg_2012-2014_on_off.pdf")
plt.clf()

plt.errorbar(rate_bin_mids,rates*10**3, xerr=rate_bin_err, yerr=rates_std*10**3, fmt="", ls="none", label="exp_off_and_on_data monthly bins")
plt.errorbar(rate_bin_mids_off,rates_off*10**3, xerr=rate_bin_err, yerr=rates_std_off*10**3, fmt="", ls="none", label="exp_off_data monthly bins")
for time in src_t:
        plt.axvline(time, color="red")
plt.ylim(0,10)
plt.ylabel("Rate in mHz")
plt.xlabel("Time in MJD days")
plt.title("allsky_rate bg 2012-2014 on_off and off data bins")
plt.legend(loc="best")
plt.savefig("plot_stash/allsky_rate_bg_2012-2014_on_off_and_off_bins.pdf")
plt.clf()


plt.plot(mjd,allsky_rate*10**3, label="sine fit on and off")
plt.plot(mjd,allsky_off*10**3, label="sine fit off")
diff = abs(allsky_rate*10**3-allsky_off*10**3)
#plt.errorbar(rate_bin_mids,rates*10**3, xerr=rate_bin_err, yerr=rates_std*10**3, fmt="", ls="none", label="exp_off_and_on_data monthly bins")
#plt.ylim(0,10)
plt.yscale("log")
plt.ylabel("Rate in mHz")
plt.xlabel("Time in MJD days")
plt.title("allsky_rate bg 2012-2014 on_off and off data")
plt.legend(loc="best")
plt.savefig("plot_stash/allsky_rate_bg_2012-2014_on_off_and_off.pdf")
plt.clf()

plt.plot(mjd,diff)
plt.yscale("log")
plt.ylabel("abs(Rate in mHz)")
plt.xlabel("Time in MJD days")
plt.title("allsky_rate bg 2012-2014 on_off and off data diff")
plt.legend(loc="best")
plt.savefig("plot_stash/allsky_rate_bg_2012-2014_on_off_and_off_diff.pdf")
plt.clf()


def my_make_grid_interp_from_hist(hist, bins, edge_fillval,
                                     interp_col_log, force_y_asc): #todo: adjust for signal/background only
    """
    Create a 2D regular grind interpolator to describe a 2D histogram, "h_sig" or "h_bg". 
    The interpolation is done in the natural logarithm of the histogram.

    When the original histograms have empty entries a 2 step method of filling
    them is used: First all edge values in each x colum are filled. Then missing
    entries within each coumn are interpolated using existing and the new edge
    entries. Because this is only done in ``y`` direction, each histogram column
    (x bin) needs at least one entry.

    Parameters
    ----------
    hist : array-like, shape (len(x_bins), len(y_bins))
        Histogram for variables ``x, y`` for the background or signal distribution.
    bins : list of array-like
        Explicit x and y bin edges used to make the histogram.
    edge_fillval : string, optional
        Fill values to use when the background histogram has no entries at
        the higest and lowest bins per column:
        - 'minmax': Use the low/high global ratio vals at the bottom/top edges.
        - 'col': Next valid value in each colum from the top/bottom is used.
        - 'minmax_col': Like 'minmax' but use min/max value per bin.
        - 'min': Only the lowest global ratio value is used at all edges.
        Filling is always done in y direction only. Listed in order from
        optimistic to conservative. (default: 'minmax_col')
    interp_col_log : bool, optional
        If ``True``, remaining gaps after filling the edge values in the signal
        or background histogram are interpolated linearly in
        ``log(ratio)`` per column. Otherwise the interpolation is in linear
        space per column. (default: ``False``)
    force_y_asc : bool, optional
        If ``True``, assume that in each column the distribution ``y`` must be
        monotonically increasing. If it is not, a conservative approach is used
        and going from the top to the bottom edge per column, each value higher
        than its predecessor is shifted to its predecessor's value until we
        arrive at the bottom edge.
        Note: If ``True``, using 'min' in ``edge_fillval`` makes no sense, so a
        ``ValueError`` is thrown. (default: ``False``)

    Returns
    -------
    interp : scipy.interpolate.RegularGridInterpolator
        2D interpolator for the logarithm of the histogram:
        ``interp(x, y) = log(hist)(x, y)``. Exponentiate to obtain the
        original values. Interpolator returns 0, if points outside given
        ``bins`` domain are requested.
    """
    if edge_fillval not in ["minmax", "col", "minmax_col", "min"]:
        raise ValueError("`edge_fillval` must be one of " +
                         "['minmax'|'col'|'minmax_col'|'min'].")
    if edge_fillval == "min" and force_y_asc:
        raise ValueError("`edge_fillval` is 'min' and 'force_y_asc' is " +
                         "`True`, which doesn't make sense together.")

    # Create binmids to fit spline to bin centers
    x_bins, y_bins = map(np.atleast_1d, bins)
    x_mids, y_mids = map(lambda b: 0.5 * (b[:-1] + b[1:]), [x_bins, y_bins])
    nbins_x, nbins_y = len(x_mids), len(y_mids)

    # Check if hist shape fits to given binning
    #if h_bg.shape != h_sig.shape:
    #    raise ValueError("Histograms don't have the same shape.")
    if hist.shape != (nbins_x, nbins_y):
        raise ValueError("Hist shapes don't match with number of bins.")

    # Check if hists are normed and do so if they are not
    dA = np.diff(x_bins)[:, None] * np.diff(y_bins)[None, :]
    #if not np.isclose(np.sum(h_bg * dA), 1.):
    #    h_bg = h_bg / (np.sum(h_bg) * dA)
    if not np.isclose(np.sum(hist * dA), 1.):
        hist = hist / (np.sum(hist) * dA)
    #assert np.isclose(np.sum(h_bg * dA), 1.)
    assert np.isclose(np.sum(hist * dA), 1.)

    # Check that all x bins in the bg hist have at least one entry
    mask = (np.sum(hist, axis=1) <= 0.)
    if np.any(mask):
        raise ValueError("Got empty x bins, this must not happen. Empty " +
                         "bins idx:\n{}".format(np.arange(nbins_x)[mask]))

    # Step 1: Construct simple ratio where we have valid entries
    sig = np.ones_like(hist) - 1.  # Use invalid value for init
    mask = (hist > 0)
    sig[mask] = hist[mask]
    # Step 2: First fill all y edge values per column where no valid values
    # are, then interpolate missing inner ratios per column.
    if edge_fillval in ["minmax", "min"]:
        sig_min, sig_max = np.amin(sig[mask]), np.amax(sig[mask])
    for i in np.arange(nbins_x):
        if force_y_asc:
            # Rescale valid bins from top to bottom, so that b_i >= b_(i+1)
            mask = (sig[i] > 0)
            masked_sig = sig[i][mask]
            for j in range(len(masked_sig) - 1, 0, -1):
                if masked_sig[j] < masked_sig[j - 1]:
                    masked_sig[j - 1] = masked_sig[j]
            sig[i][mask] = masked_sig

    # Get invalid points in current column
    m = (sig[i] <= 0)

    if edge_fillval == "minmax_col":
        # Use min/max per slice instead of global min/max
        sig_min, sig_max = np.amin(sig[i][~m]), np.amax(sig[i][~m])

    # Fill missing top/bottom edge values, rest is interpolated later
    # Lower edge: argmax stops at first True, argmin at first False
    low_first_invalid_id = np.argmax(m)
    if low_first_invalid_id == 0:
        # Set lower edge with valid point, depending on 'edge_fillval'
        if edge_fillval == "col":
            # Fill with first valid ratio from bottom for this column
            low_first_valid_id = np.argmin(m)
            sig[i, 0] = sig[i, low_first_valid_id]
        elif edge_fillval in ["minmax", "minmax_col", "min"]:
            # Fill with global min or with min for this col
            sig[i, 0] = sig_min

    # Repeat with turned around array for upper edge
    hig_first_invalid_id = np.argmax(m[::-1])
    if hig_first_invalid_id == 0:
        if edge_fillval == "col":
            # Fill with first valid ratio from top for this column
            hig_first_valid_id = len(m) - 1 - np.argmin(m[::-1])
            sig[i, -1] = sig[i, hig_first_valid_id]
        elif edge_fillval == "min":
            # Fill also with global min
            sig[i, -1] = sig_min
        elif edge_fillval in ["minmax", "minmax_col"]:
            # Fill with global max or with max for this col
            sig[i, -1] = sig_max

    # Interpolate missing entries in the current column
    mask = (sig[i] > 0)
    _x = y_mids[mask]
    _y = sig[i, mask]
    if interp_col_log:
        col_interp = sci.interp1d(_x, np.log(_y), kind="linear")
        sig[i] = np.exp(col_interp(y_mids))
    else:
        col_interp = sci.interp1d(_x, _y, kind="linear")
        sig[i] = col_interp(y_mids)

    # Step 3: Construct a 2D interpolator for the log(hist).
    # Repeat values at the edges (y then x) to cover full bin domain, so the
    # interpolator can throw an error outside the domain
    sig_full = np.zeros((nbins_x + 2, nbins_y + 2), dtype=sig.dtype) - 1.
    for j, col in enumerate(sig):
        sig_full[j + 1] = np.concatenate([col[[0]], col, col[[-1]]])
    sig_full[0] = sig_full[1]
    sig_full[-1] = sig_full[-2]
    # Build full support points
    pts_x = np.concatenate((x_bins[[0]], x_mids, x_bins[[-1]]))
    pts_y = np.concatenate((y_bins[[0]], y_mids, y_bins[[-1]]))

    interp = sci.RegularGridInterpolator([pts_x, pts_y], np.log(sig_full),
                                         method="linear", bounds_error=False,
                                         fill_value=0.)
    return interp


