import numpy as np

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
