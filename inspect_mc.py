"""
script to inspect and plot mc data
"""

import numpy as np
from tdepps.utils import fit_spl_to_hist, make_equdist_bins
from matplotlib import pyplot as plt
import os as _os
from glob import glob as _glob
import json
import numpy as np

import _loader


import sys

#sys.path.append("../")


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
mc_no_ehe = load_mc_data(names, test_path + "mc_no_ehe")
my_mc_no_ehe = load_mc_data(names, test_path + "my_mc_no_ehe")
mc_no_ehe_alert = load_mc_data(names, test_path + "mc_no_ehe_alert")
mc_unfiltered = load_mc_data(names, test_path + "mc_unfiltered")
mc_ehe_alert = load_mc_data(names, test_path + "mc_ehe_alert")
source_pos = load_sources(source_path)
source_pos = {i:source_pos[i] for i in source_pos if i!="IC79"}
#source_pos = {list(source_pos.keys())[1]:list(source_pos.values())[1]} # just 2012-2014

source_pos = give_source_dec(source_pos)
print("plotting")
plot_weighted_hist_mc_no_ehe(mc_no_ehe, source_pos, bins=100, mc_unfiltered = mc_unfiltered, my_mc_no_ehe = my_mc_no_ehe, mc_no_ehe_alert = mc_no_ehe_alert, log = True, path="plot_stash")
plot_weighted_hist_mc_no_ehe(mc_no_ehe_alert, source_pos, bins = -1, min_bin = 500, path="plot_stash", name = "_no_alerts")
plot_weights(mc_no_ehe_alert, source_pos, path= "plot_stash")
#plot_energy_dec_hist(mc_unfiltered[names[0]], mc_ehe_alert[names[0]], bins = 50, title = names[0], path="plot_stash", name ="_"+names[0])
#plot_energy_dec_hist(mc_unfiltered[names[1]], mc_ehe_alert[names[1]], bins = 50, title = names[1], path="plot_stash", name ="_"+names[1])
#plot_energy_dec_hist(mc_unfiltered[names[2]], mc_ehe_alert[names[2]], bins = 50, title = names[2], path="plot_stash", name ="_"+names[2])
plot_energy_dec_hist(mc_unfiltered, mc_ehe_alert, bins=50, path="plot_stash")
