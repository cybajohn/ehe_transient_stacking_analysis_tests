import numpy as np
from _paths import PATHS
import _loader


# Load files and build the models one after another to save memory
sample_names = _loader.source_list_loader()
sample_names = [sample_names[2]] # just 2012-2014
source_type = "ehe" # "ehe" or "hese"

all_srcs = []
for key in sample_names:
        print(type(_loader.source_list_loader(key)[key]))
        all_srcs.extend(_loader.source_list_loader(key)[key])

# just one source to have something to compare to skylab
#all_srcs = [all_srcs[5]]

srcs_time = []

for src in all_srcs:
	srcs_time.append(src["mjd"])

print(srcs_time)


print(srcs_time[2], srcs_time[6])

print(srcs_time[6]-srcs_time[2])

windows1 = np.linspace(10,100,15)
windows2 = np.linspace(120,500,5)
windows = np.concatenate((windows1,windows2))

print(windows*2)
