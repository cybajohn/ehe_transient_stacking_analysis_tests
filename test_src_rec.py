"""
to test the src distribution
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

from matplotlib import pyplot as plt

# lets play
dayinsec = 60.*60.*24.
days = 100.
dt0 = -days*dayinsec
dt1 = days*dayinsec



#load models one after another to save memory
sample_names = _loader.source_list_loader()
sample_names = sample_names[2:] # without IC79, IC86_2011
source_type = "ehe" # "ehe" or "hese"

all_srcs = []
for key in sample_names:
        print(type(_loader.source_list_loader(key)[key]))
        all_srcs.extend(_loader.source_list_loader(key)[key])

for key in sample_names:
        print("\n" + 80 * "#")
        print("# :: Setup for sample {} ::".format(key))
        #opts = _loader.settings_loader(key)[key].copy()
        #opts = _loader._common_loader(key,folder="saved_test_model/settings",info="settings")[key].copy()
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
        #srcs_rec = make_src_records(srcs, dt0=dt0, dt1=dt1)
        srcs_rec = phys.make_src_records_from_all_srcs(all_srcs, dt0=dt0, dt1=dt1,
                                                X=np.concatenate((exp_off,exp_on)))



print(len(srcs))
print(len(srcs_rec))

