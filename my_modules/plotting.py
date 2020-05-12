import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tdepps.utils import make_equdist_bins, fit_spl_to_hist

def fluss(mc_sample, phi0, E0, gamma):
	"""
	calculates the flux
	
	Parameters
	----------
	mc_sample: dict
		mc sample
	phi0: double
		norm of flux
	E0: double
		energy
	gamma: double
		gamma
	
	Return
	------
	flux: double
		flux
	"""
	return phi0*(mc_sample["trueE"]/E0)**(-gamma)

def generate_mc_weight(mc_sample):
	"""
	Generates the mc weight
	
	Parameters
	----------
	mc_sample: dict
		mc sample
	
	Return
	------
	weight: list
		mc weights
	"""
        return mc_sample["ow"] * fluss(mc_sample,1,1,2)


def plot_energy_dec_hist(mc_1, mc_2, bins, path="", name=""):
	"""
	Plots fancy diagrams containing two 2D histograms of the energy and sin(dec_angle) originating from two mc samples.
	The titles and save names will be adjusted to the names within the samples.
	
	Parameters
	----------
	mc_1: dict
		dict of mc_samples without cleaning
	mc_2: dict
		dict of cleaned out mc_samples
	bins: int
		number of bins
	path: str (optional)
		outpath
	name: str (optional)
		additional save name

	Return
	------
	saves fancy plots :)
	"""
	for sample in mc_1.keys():
		x_1 = np.sin(mc_1[sample]["trueDec"])
		y_1 = np.log10(mc_1[sample]["trueE"])
		weight_1 = generate_mc_weight(mc_1[sample])
		x_2 = np.sin(mc_2[sample]["trueDec"])
	        y_2 = np.log10(mc_2[sample]["trueE"])
	        weight_2 = generate_mc_weight(mc_2[sample])
		plt.title(sample+", mc events")
		plt.subplot(221)
		plt.suptitle(sample+", mc events")
		ticks = [10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 10**0]
		plt.hist2d(x_1,y_1,bins = bins, range=[[-1,1],[2,9]], weights=weight_1, norm=LogNorm(), cmap=plt.cm.get_cmap("OrRd",10))
		plt.xlim(-1,1)
		plt.ylim(2,9)
		plt.xlabel(r'$\sin(\delta_\nu)$'+'\nalle mc events')
		plt.ylabel(r'$\mathrm{log}_{10}(E_\nu/\mathrm{GeV})$')
		plt.clim(10**(-5),10**0)
		cbar = plt.colorbar(ticks=ticks)
		cbar.ax.set_yticklabels([])
		plt.subplot(222)
		plt.hist2d(x_2,y_2,bins = bins, range=[[-1,1],[2,9]], weights=weight_2, norm=LogNorm(), cmap=plt.cm.get_cmap("OrRd",10))
		plt.xlim(-1,1)
		plt.ylim(2,9)
		plt.xlabel(r'$\sin(\delta_\nu)$'+'\nnur ehe alerts')
		plt.clim(10**(-5),10**0)
		plt.colorbar(ticks=ticks)
		plt.savefig(path+"/dec_energy_"+sample+name+".pdf")
		plt.clf()

def plot_weighted_hist_mc_no_ehe(mc, source_pos, bins, min_bin = None, mc_unfiltered = None, my_mc_no_ehe = None, mc_no_ehe_alert = None, log = False, histtype = 'step', path = "", name = ""):
        """
	Plots hist of mc data
	
	Parameters
	----------
	mc: dict
		mc data to be plotted
	source_pos: dict
		source pos (dec)
	bins: int
		-1 for equdist bin method or any integer for number of bins
	min_bin: int (optional)
		number of min events in bins for equdist bin method
	mc_unfiltered: dict (optional)
		unfiltered mc data
	my_mc_no_ehe: dict (optional)
		my filtered data
	log: bool (optional)
		True for log scale
	histtype: matplotlib statement (optional)
		sets histogram style
	path: str (optional)
		save path
	name: str (optional)
		additional save name
	
	Return
	------
	saves a plot :)
	"""
	for key in source_pos.keys():
		print("Plotting hist for "+key)
                w_sig = generate_mc_weight(mc[key])
                mc_sin_dec = np.sin(mc[key]["trueDec"])
		_bins = bins
		if (_bins < 0):
                        _bins = make_equdist_bins(mc_sin_dec, -1, 1, weights=w_sig, min_evts_per_bin=min_bin)
                        print("number of bins: ",len(_bins))
                else:
                        _bins = np.linspace(-1,1,bins)
		plt.hist(mc_sin_dec, _bins, weights=w_sig, histtype=histtype, log=log, label="mc_no_ehe")
                source_dec_x = np.sin(np.array(source_pos[key]))
                source_dec_y = []
                hist, edges = np.histogram(mc_sin_dec,_bins,weights=w_sig)

                for x_pos in source_dec_x:
                        source_dec_y.append(hist[np.where(_bins<x_pos)[0][-1]])
                plt.plot(source_dec_x,source_dec_y,"x", label="source_pos")
                if mc_unfiltered is not None:
                        mc_sin_dec = np.sin(mc_unfiltered[key]["trueDec"])
                        w_sig = generate_mc_weight(mc_unfiltered[key])
                        plt.hist(mc_sin_dec, _bins, weights=w_sig, histtype=histtype, log=log, label="mc")
                if my_mc_no_ehe is not None:
                        mc_sin_dec = np.sin(my_mc_no_ehe[key]["trueDec"])
                        w_sig = generate_mc_weight(my_mc_no_ehe[key])
                        plt.hist(mc_sin_dec, _bins, weights=w_sig, histtype=histtype, log=log, label= "my_mc_no_ehe")
		if mc_no_ehe_alert is not None:
                        mc_sin_dec = np.sin(mc_no_ehe_alert[key]["trueDec"])
                        w_sig = generate_mc_weight(mc_no_ehe_alert[key])
                        plt.hist(mc_sin_dec, _bins, weights=w_sig, histtype=histtype, log=log, label= "mc_no_ehe_alert")
                titles = ["PointSourceTracks","GFU"]
                if key == "IC86_2015":
                        plt.title("skylab_"+titles[1]+"_"+key+"_weighted")
                else:
                        plt.title("skylab_"+titles[0]+"_"+key+"_weighted")
                plt.xlabel(r"$\sin(\delta_\nu)$")
                plt.legend(loc='best')
                plt.savefig(path+"/weighted_"+key+name+".pdf")
                plt.clf()

def plot_weights(mc, source_pos, min_evts_per_bin=500, path="", name=""):
	"""
        Plots the final spline demonstrating the source weights
        
        Parameters
        ----------
        mc: dict
                mc data to be plotted
        source_pos: dict
                source pos (dec)
	min_evts_per_bin: int (optional)
		number of min events per bin
	path: str (optional)
		save path
	name: str (optional)
		additional save name
	
	Return
	------
	saves a plot :)	
	"""
	for key in source_pos.keys():
		w_sig = generate_mc_weight(mc[key])
		mc_sin_dec = np.sin(mc[key]["trueDec"])
		bins = make_equdist_bins(mc_sin_dec, -1, 1, weights=w_sig, min_evts_per_bin=min_evts_per_bin)
		print("made {} bins for allsky hist, {}".format(len(bins),key))
		hist = np.histogram(mc_sin_dec, bins=bins, weights=w_sig, density=False)[0]
		variance = np.histogram(mc_sin_dec, bins=bins, weights=w_sig**2, density=False)[0]
		dA = np.diff(bins)
		hist = hist / dA
		stddev = np.sqrt(variance) / dA
		weight = 1. / stddev
		mc_spline = fit_spl_to_hist(hist, bins=bins, w=weight, s=len(hist))[0]
		z = len(bins) / 2.
		x = np.linspace(-1,1,len(bins))
		y = mc_spline(x)/np.sum(mc_spline(x)) * z
		source_dec_x = np.sin(np.array(source_pos[key]))
		source_dec_y = mc_spline(source_dec_x)/np.sum(mc_spline(x)) * z
		plt.plot(x,y,label= r"mc[~alert]-spline")
		plt.plot(source_dec_x, source_dec_y, "d", label="weights")
		plt.title(key)
		plt.xlim(-1,1)
		plt.ylabel("PDF")
		plt.xlabel(r"$\sin(\delta_\nu)$")
		plt.legend(loc="best")
		plt.savefig(path+"/weights_"+key+name+".pdf")
		plt.clf()
