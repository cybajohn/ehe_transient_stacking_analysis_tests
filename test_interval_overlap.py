import numpy as np
import _loader
from tdepps.backend import bg_time_box
import matplotlib.pyplot as plt

sample_names = _loader.source_list_loader()
sample_names = sample_names[1:] # without IC79, IC86_2011
source_type = "ehe" # "ehe" or "hese"

all_srcs = []
for key in sample_names:
        print(type(_loader.source_list_loader(key)[key]))
        all_srcs.extend(_loader.source_list_loader(key)[key])
srcs_dt0 = np.zeros(shape=(len(all_srcs)))
srcs_dt1 = np.zeros(shape=(len(all_srcs)))
srcs_time = np.zeros(shape=(len(all_srcs)))

SECINDAY = 24. * 60. * 60.

days = 100

dt1 = days*SECINDAY
dt0 = -dt1 

for k,src in enumerate(all_srcs):
	srcs_time[k] = src["mjd"]
	srcs_dt0[k] = dt0
	srcs_dt1[k] = dt1

srcs_time = srcs_time.reshape(len(srcs_time),1)
srcs_dt_mjd = (np.vstack((srcs_dt0,srcs_dt1)).T / SECINDAY + srcs_time)




print(srcs_time)
print(srcs_dt_mjd)



"""
def merge_intervals(intervals):
	starts = intervals[:,0]
	ends = np.maximum.accumulate(intervals[:,1])
	valid = np.zeros(len(intervals) + 1,dtype=np.bool)
	valid[0] = True
	valid[-1] = True
	valid[1:-1] = starts[1:] >=ends[:-1]
	return np.vstack((starts[:][valid[:-1]], ends[:][valid[1:]])).T
"""

def merge_intervals(intervals):
	"""
	Merges list of intervals without overlap.
	Example: input: [[0,3],[2,6],[5,9],[10,14],[12,16]]
		 output: [[0,9],[10,16]]
	"""
	# sort if needed
	intervals.sort(key=lambda interval: interval[0])
	merged = [intervals[0]]
	for item in intervals:
		previous = merged[-1]
		if item[0] <= previous[1]:
			previous[1] = np.maximum(previous[1],item[1])
		else:
			merged.append(item)
	return merged

def get_unique_intervals(intervals):
	intervals.sort(axis=0)
	merged = [intervals[0]]
	#first = interval
	for item in intervals:
		if item[0] <= merged[-1][1]:
			merged[-1][1] = np.minimum(merged[-1][1],item[1])
		else:
			merged.append(item)
	return np.atleast_2d(merged).reshape(len(merged), 2)

def get_unique_intervals_v2(intervals):
	interval_bounds = intervals.reshape(len(intervals) * 2)
	interval_bounds.sort(axis=0)
	interval_mids = (interval_bounds[1:] + interval_bounds[:-1]) / 2. 
	return interval_mids

def get_unique_interval_hits(intervals):
	interval_bounds = intervals.reshape(len(intervals) * 2)
	interval_bounds.sort(axis=0)
	hit_intervals = np.vstack((interval_bounds[:-1],interval_bounds[1:])).T
	return hit_intervals	

times = [[0,3],[2,6],[5,9],[10,14],[12,16]]
times2 = [[0,1],[0,3],[2,6],[5,9],[10,14],[12,16],[15,16],[14,16]]

times0 = [[1,3],[2,4],[5,7]]
times4 = [[1,4],[3,6],[5,8],[9,12]]
"""
i want for times0: [1,2],[2,3],[3,4],[5,7]
i want for times4: [1,3],[3,4],[4,5],[5,6],[6,8],[9,12]
"""


times3 = np.array(times2)

print("times3: ", times3)
times3.sort(axis=0)
print("sorted: ", times3)

print(times2)
print(merge_intervals(times2))

print("get_unique_intervals(times)")
print(times0)
print(get_unique_intervals(np.array(times0)))

print("v2")
print(get_unique_intervals_v2(np.array(times0)))
# lists are weird, I mean okay, you say k is the list i
# but why doesnt this work on integers?
i = [1]
k = i
k[0] = 2
print(k,i)

tmp = get_unique_intervals_v2(srcs_dt_mjd)
print(bg_time_box(tmp, srcs_dt_mjd[:,0], srcs_dt_mjd[:,1]).T)
print(get_unique_intervals_v2(srcs_dt_mjd))
print(srcs_dt_mjd)
#print(bg_time_box(tmp, srcs_dt_mjd[:,0], srcs_dt_mjd[:,1]))
mask = bg_time_box(tmp, srcs_dt_mjd[:,0], srcs_dt_mjd[:,1]).T
my_intervals = []
for masks in mask:
	a = srcs_dt_mjd[masks.astype(np.bool)]
	if len(a)!= 0:
		print(a)
		low = np.amin(a)
		up = np.amax(a)
		my_intervals.append([low,up])

np.savetxt("plot_stash/data/time_intervals.txt",my_intervals, delimiter=",")	
print(my_intervals)
print(len(srcs_time))
np.savetxt("plot_stash/data/src_times.txt",srcs_time, delimiter=",")

hits = get_unique_interval_hits(srcs_dt_mjd)
print(hits)
#hits = hits[
mask2 = np.sum(mask,axis=1).astype(np.bool)
print(hits[mask2])

np.savetxt("plot_stash/data/time_interval_hits.txt", hits[mask2], delimiter=",")




