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

from tdepps.utils import fit_spl_to_hist, make_time_dep_dec_splines, spl_normed
from tdepps.utils import make_unique_time_dep_dec_splines
from tdepps.utils import make_equdist_bins

from matplotlib import pyplot as plt

from matplotlib import gridspec

print("hello world")

### functions

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



def sec2timestr(sec):
    sec = np.around(sec, decimals=3)
    d = int(sec / SECINDAY)
    sec -= d * SECINDAY
    h = int(sec / SECINHOUR)
    sec -= h * SECINHOUR
    m = int(sec / SECINMIN)
    s = sec - m * SECINMIN
    return "{:d}d : {:02d}h : {:02d}m : {:06.3f}s".format(d, h, m, s)



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



SECINMIN = 60.
SECINHOUR = 60. * SECINMIN
SECINDAY = 24. * SECINHOUR

all_srcs = []
for key in all_sample_names:
	key = key.replace(", ", "_")
        print(type(_loader.source_list_loader(key)[key]))
        all_srcs.extend(_loader.source_list_loader(key)[key])

for key in all_sample_names:
	dts = np.logspace(-1.3,2.3,5) * SECINDAY
	key = key.replace(", ", "_")
    	print("Working with sample {}".format(key))
	opts = _loader._common_loader(key,folder="saved_test_model/settings",info="settings")[key].copy()
        print("load off data")
        exp_off = _loader.off_data_loader(key)[key]
        print("load on data")
        exp_on = _loader.on_data_loader(key)[key]
        print("load mc")
        mc = _loader.mc_loader(source_type=source_type,names=key)[key]
        print("load runlist")
        runlist = _loader.runlist_loader(key)[key]
        # Process to tdepps format
	X = np.concatenate((exp_off,exp_on))

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
	
	spline_list = []
	spline_list2 = []
	
	for j,timewindow in enumerate(dts):
		print("timewindow: ", timewindow * 2.)
		srcs = phys.make_src_records_from_all_srcs(all_srcs, dt0=-timewindow, dt1=timewindow,
                                                X=X)
		
		sin_dec_splines, spl_info = make_time_dep_dec_splines(
                	X=X, srcs=srcs, run_list=runlists[key], sin_dec_bins=sin_dec_bins,
                	rate_rebins=spatial_opts["rate_rebins"],
                	spl_s=spatial_opts["spl_s"],
                	n_scan_bins=spatial_opts["n_scan_bins"],
                	ignore_zero_runs=False)
		
		unique_sin_dec_splines, spl_info_unique = make_unique_time_dep_dec_splines(
                        X=X, srcs=srcs, run_list=runlists[key], sin_dec_bins=sin_dec_bins,
                        rate_rebins=spatial_opts["rate_rebins"],
                        spl_s=spatial_opts["spl_s"],
                        n_scan_bins=spatial_opts["n_scan_bins"],
                        ignore_zero_runs=False)




		# Normalize sindec splines to be a PDF in sindec for sampling weights
        	def spl_normed_factory(spl, lo, hi, norm):
            		""" Renormalize spline, so ``int_lo^hi renorm_spl dx = norm`` """
            		return spl_normed(spl=spl, norm=norm, lo=lo, hi=hi)
		
        	lo, hi = sin_dec_bins[0], sin_dec_bins[-1]
        	sin_dec_pdf_splines = []
		unique_sin_dec_pdf_splines = []
        	for spl in sin_dec_splines:
            		sin_dec_pdf_splines.append(spl_normed_factory(spl, lo, hi, norm=1.))
		
		for u_spl in unique_sin_dec_splines:
			unique_sin_dec_pdf_splines.append(spl_normed_factory(u_spl, lo, hi, norm=1.))
	
        	# Make sampling CDFs to sample sindecs per source per trial
        	# First a PDF spline to estimate intrinsic data sindec distribution
        	ev_sin_dec = np.sin(X["dec"])
		inj_opts = opts["bg_inj_opts"]
        	_bins = make_equdist_bins(
            		ev_sin_dec, lo, hi, weights=None,
            		min_evts_per_bin=inj_opts["n_data_evts_min"])
        	# Spline is interpolating to cover the data densitiy as fine as possible
        	# because for resampling we divide by the initial densitiy.
        	hist = np.histogram(ev_sin_dec, bins=_bins, density=True)[0]
        	data_spl = fit_spl_to_hist(hist, bins=_bins, w=None, s=0)[0]
        	data_spl = spl_normed_factory(data_spl, lo, hi, norm=1.)
		
        	# Build sampling weights from PDF ratios
        	sample_w = np.empty((len(sin_dec_pdf_splines), len(ev_sin_dec)),
                	            dtype=float)
		unique_sample_w = np.empty((len(unique_sin_dec_pdf_splines), len(ev_sin_dec)), dtype=float)
        	_vals = data_spl(ev_sin_dec)
        	for i, spl in enumerate(sin_dec_pdf_splines):
            		sample_w[i] = spl(ev_sin_dec) / _vals
		
		for i, u_spl in enumerate(unique_sin_dec_pdf_splines):
			unique_sample_w[i] = u_spl(ev_sin_dec) / _vals
		
        	# Cache fixed sampling CDFs for fast random choice
        	CDFs = np.cumsum(sample_w, axis=1)
        	sample_CDFs = CDFs / CDFs[:, [-1]]
		unique_CDFs = np.cumsum(unique_sample_w, axis=1)
		unique_sample_CDFs = unique_CDFs / unique_CDFs[:, [-1]]
		x = np.linspace(-1,1,1000)
		"""
		plt.plot(np.sort(ev_sin_dec),sample_CDFs[0],label="example CDF")
		plt.xlabel(r'$\sin(\delta)$ (data)')
		plt.legend(loc='best')
		plt.savefig("plot_stash/sampling/CDF_0.pdf")
		
		plt.clf()
		"""
		spline_list.append(sin_dec_splines[0])
		if j==2:
			for i,spline in enumerate(sin_dec_splines):
                		plt.plot(x,spline(x)/(np.sum(spline(x))*2./len(x)),label = "source "+str(i))
        		plt.xlabel(r'$\sin(\delta)$ (spline)')
        		plt.legend(loc='best')
		        plt.savefig("plot_stash/sampling/sin_dec_spline_all_"+ str(timewindow*2/SECINDAY)+ ".pdf")
			plt.clf()
			special_spline = sin_dec_splines
			special_time = timewindow
		#plt.plot(x,sin_dec_splines[0](x))#,label="example sin_dec_spline")
	x = np.linspace(-1,1,1000)
	for i,spline in enumerate(spline_list):
		plt.plot(x,spline(x)/(np.sum(spline(x))*2./len(x)), label=str(dts[i]*2./SECINDAY)+" days")
	plt.xlabel(r'$\sin(\delta)$ (spline)')
        plt.legend(loc='best')
	plt.savefig("plot_stash/sampling/sin_dec_spline_0_all_timewindows.pdf")
	plt.clf()
	for i,spline in enumerate(sin_dec_splines):
		plt.plot(x,spline(x)/(np.sum(spline(x))*2./len(x)),label = "source "+str(i))
	plt.xlabel(r'$\sin(\delta)$ (spline)')
        plt.legend(loc='best')
        plt.savefig("plot_stash/sampling/sin_dec_spline_all.pdf")
	plt.clf()
        for i,spline in enumerate(special_spline):
                plt.plot(x,spline(x)/(np.sum(spline(x))*2./len(x)),label = "source "+str(i)+" low time")
		#plt.plot(x,sin_dec_splines[i](x)/(np.sum(sin_dec_splines[i](x))*2./len(x)),label = "source "+str(i)+" 400 days")
        plt.plot(x,sin_dec_splines[3](x)/(np.sum(sin_dec_splines[3](x))*2./len(x)),label = "source "+str(3)+" 400 days")

	plt.xlabel(r'$\sin(\delta)$ (spline)')
        plt.legend(loc='best')
        plt.savefig("plot_stash/sampling/sin_dec_spline_all_time_comp.pdf")
	plt.clf()
	plt.plot(x,sin_dec_splines[5](x)/(np.sum(sin_dec_splines[5](x))*2./len(x)),label = "source "+str(5)+" 400 days")
	plt.plot(x,sin_dec_splines[7](x)/(np.sum(sin_dec_splines[7](x))*2./len(x)),label = "source "+str(7)+" 400 days")
	plt.plot(x,special_spline[5](x)/(np.sum(special_spline[5](x))*2./len(x)),label = "source "+str(5)+" 6 days")
        plt.plot(x,special_spline[7](x)/(np.sum(special_spline[7](x))*2./len(x)),label = "source "+str(7)+" 6 days")
	plt.xlabel(r'$\sin(\delta)$ (spline)')
        plt.legend(loc='best')
        plt.savefig("plot_stash/sampling/sin_dec_spline_special_comp.pdf")
	"""
	plt.clf()
	plt.plot(np.sort(ev_sin_dec),sample_w[0],label="example weights")
	plt.xlabel(r'$\sin(\delta)$ (data)')
        plt.legend(loc='best')
	plt.savefig("plot_stash/sampling/sample_w_0.pdf")
	plt.clf()
	plt.plot(np.sort(ev_sin_dec),_vals)
	plt.xlabel(r'$\sin(\delta)$ (data)')
	plt.savefig("plot_stash/sampling/_vals.pdf")
	"""

plt.clf()
for i,cdf in enumerate(unique_sample_CDFs):
	plt.plot(np.sort(ev_sin_dec),cdf,label=str(i)+" unique CDF 400 days")
plt.xlabel(r'$\sin(\delta)$ (data)')
plt.legend(loc='best')
plt.savefig("plot_stash/sampling/unique_CDFs_400_d.pdf")
                
plt.clf()




for i,spline in enumerate(unique_sin_dec_pdf_splines):
        plt.plot(x,spline(x), label=str(i)+" 400 days")
plt.xlabel(r'$\sin(\delta)$ (spline)')
plt.legend(loc='best')
plt.savefig("plot_stash/sampling/unique_sin_dec_splines_400_d.pdf")
plt.clf()



fig = plt.figure(figsize=(6, 7)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[5, 2]) 
ax0 = plt.subplot(gs[0])
ax0.plot(x,sin_dec_splines[5](x)/(np.sum(sin_dec_splines[5](x))*2./len(x)),label = "source "+str(5)+" 400 days")
ax0.plot(x,sin_dec_splines[7](x)/(np.sum(sin_dec_splines[7](x))*2./len(x)),label = "source "+str(7)+" 400 days")
ax0.plot(x,special_spline[5](x)/(np.sum(special_spline[5](x))*2./len(x)),label = "source "+str(5)+" 6 days")
ax0.plot(x,special_spline[7](x)/(np.sum(special_spline[7](x))*2./len(x)),label = "source "+str(7)+" 6 days")
#ax0.set_ylim(0,1)
ax0.set_xlabel(r'$\sin(\delta)$ (spline)')
ax0.set_ylabel(r'PDF')
ax0.legend(loc='best')


ax1 = plt.subplot(gs[1])



mjd = np.linspace(np.amin(X["time"]),np.amax(X["time"]),1000)
allsky_rate = spl_info["allsky_rate_func"].fun(mjd,spl_info["allsky_best_params"])
ax1.plot(mjd,allsky_rate*10**3, label="sine fit")
ax1.axvline(srcs["time"][5], color="green")
ax1.axvline(srcs["time"][7], color="red")
ax1.axvspan(srcs["time"][5]-special_time/SECINDAY,srcs["time"][5]+special_time/SECINDAY,alpha=0.5,color="green")
ax1.axvspan(srcs["time"][7]-special_time/SECINDAY,srcs["time"][7]+special_time/SECINDAY,alpha=0.5,color="red")
ax1.axvspan(srcs["time"][5]-dts[-1]/SECINDAY,srcs["time"][5]+dts[-1]/SECINDAY,alpha=0.1,color="green")
ax1.axvspan(srcs["time"][7]-dts[-1]/SECINDAY,srcs["time"][7]+dts[-1]/SECINDAY,alpha=0.1,color="red")


ax1.set_ylabel("Rate in mHz")
ax1.set_xlabel("Time in MJD days")
ax1.legend(loc="best")
plt.savefig("plot_stash/sampling/allsky_rate_bg_2012-2014.pdf")
plt.clf()





print("sample_time: ",np.amin(X["time"]),np.amax(X["time"]))
print("src_5_time: ",srcs["time"][5])
print("src_7_time: ",srcs["time"][7])


print("fin :)")
