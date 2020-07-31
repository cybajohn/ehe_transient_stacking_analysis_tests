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

times = [[0,3],[2,6],[5,9],[10,14],[12,16]]
times2 = [[0,1],[0,3],[2,6],[5,9],[10,14],[12,16],[15,16],[14,16]]

print(times2)
print(merge_intervals(times2))

# lists are weird, I mean okay, you say k is the list i
# but why doesnt this work on integers?
i = [1]
k = i
k[0] = 2
print(k,i)
