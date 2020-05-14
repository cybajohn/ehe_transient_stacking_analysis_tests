"""
testscript to figure out how to filter events by example i3 files
"""

from __future__ import absolute_import
from icecube import icetray
#from icecube.filterscripts import filter_globals
from I3Tray import *
from icecube import icetray, dataclasses, dataio, filterscripts, filter_tools, trigger_sim, WaveCalibrator
from icecube import phys_services, DomTools
from icecube.filterscripts import filter_globals
from icecube.phys_services.which_split import which_split
#from icecube.filterscripts.filter_globals import EHEAlertFilter
from icecube import VHESelfVeto, weighting

from icecube.icetray import I3Units

import json
import numpy as np

import sys

#sys.path.append("../")
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from my_modules import MyEHEFilter, MyPreCalibration, MyEHECalibration, MyPreCalibration_IC86_2012, MyEHECalibration_IC86_2012, MyEHEFilter_IC86_2012, which_split, ehe_collector, EHEAlertFilter
from my_modules import push_test, push_test2, RunEHEAlertFilter, test_run_alert_filter, CheckFilter

from my_modules import HighQFilter

###### functions to inspect frames or other things

def CheckThings(frame, key):
	"""
	tray.AddModule(CheckThings, 'CheckThings',
			key = key)
	"""
	if not key in frame:
		print("frame does not contain {}".format(key))
		return False
	pass #do something

def check_data_equality(file1,file2):
	print("checking equality of files: ",file1,file2)
	with open(file1) as _file:
        	file1 = json.load(_file)
	with open(file2) as _file:
		file2 = json.load(_file)
	for key in file1:
		print("checking key: ",key)
		array1 = np.unique(file1[key])
		array2 = np.unique(file2[key])
		if not np.array_equal(array1,array2):
			print(key," is not equal")
	print("check finished")	
	


###### define infiles and outfiles

filelist = []
file_type = "IC86_2011"
if file_type == "IC79":
	gcd_file = "../../../../data/sim/IceCube/2010/filtered/level2/neutrino-generator/6359/00000-00999/GeoCalibDetectorStatus_IC79.55380_corrected.i3.gz"
	qp_file = "../../../../data/ana/IC79/level3-mu/sim/6308/Level3_IC79_nugen_numu.006308.004997.i3.bz2"
if file_type == "IC86_2012":
	qp_file = "../../../../data/ana/PointSource/IC86_2012_PS/files/sim/2012/neutrino-generator/11029/00000-00999/Final_v2_nugen_numu_IC86.2012.011029.000500.i3.bz2"
	gcd_file = "../../../../data/sim/IceCube/2012/filtered/level2/neutrino-generator/11029/00000-00999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz"

if file_type == "IC86_2011":
	qp_file = "../../../../data/sim/IceCube/2011/filtered/level2/neutrino-generator/9095/00000-00999/Level2_nugen_numu_IC86.2011.009095.000234.i3.bz2"
	gcd_file = "../../../../data/sim/IceCube/2011/filtered/level2/neutrino-generator/9095/00000-00999/GeoCalibDetectorStatus_IC86.55697_corrected_V2.i3.gz"
		

#files = '../../../data/exp/IceCube/2011/filtered/level2/0731/Level2_IC86.2011_data_Run00118514_Part00000085.i3.bz2'

filelist.append(gcd_file)
filelist.append(qp_file)

outfile = "results_{}".format(file_type)
"""
print("loading DomTools")
icetray.load("DomTools", False)

print("loading portia")
icetray.load("portia", False)

print("get hacked portia")
icetray.load("filterscripts", False)
"""
print("create tray")
tray = I3Tray()

print("read data")

tray.AddModule("I3Reader", "reader", Filenamelist=filelist)

tray.AddSegment(MyEHEFilter_IC86_2012,
		If = which_split(split_name='InIceSplit')
		)

"""

tray.AddSegment(MyEHEFilter,
		If = which_split(split_name='InIceSplit')
		)
"""
"""

tray.AddModule(test_run_alert_filter,
                outfilename = "run_ehe_alert_filter_test",
                If =  lambda f:RunEHEAlertFilter(f)
                )

"""
tray.AddSegment(EHEAlertFilter,
		If = which_split(split_name='InIceSplit')
		)

tray.AddModule(CheckFilter,
		filter_key = "EHEFilter_11",
		test_key = "MyEHEFilter",
		outfilename = outfile + "/FilterCheckPerModule",
		If = which_split(split_name='InIceSplit')
		)

tray.AddModule(HighQFilter, "my_HighQFilter",
		If = which_split(split_name='InIceSplit')
		)

tray.AddModule(weighting.get_weighted_primary, "weighted_primary",
                   If=lambda frame: not frame.Has("MCPrimary"))


tray.AddModule(ehe_collector,"collector",
		outfilename = outfile + "/ehe_test_stats",
		outfilename2 = outfile + "/my_ehe_test_stats",
		outfilename3 = outfile + "/alert_stats",
		If = which_split(split_name='InIceSplit')
		)



if file_type == "IC79":
	tray.AddModule('I3Writer', 'writer',
			Filename='filtertest/filtertest_79.i3.bz2')
if file_type == "IC86_2012":
	tray.AddModule('I3Writer', 'writer',
			Filename='filtertest/filtertest_2012.i3.bz2')

if file_type == "IC86_2011":
	tray.AddModule('I3Writer', 'writer',
			Filename='filtertest/filtertest_2011.i3.bz2')

tray.AddModule("TrashCan")

tray.Execute()

tray.Finish()

check_data_equality(outfile + "/ehe_test_stats",outfile + "/my_ehe_test_stats")

"""
i3f = dataio.I3File("filtertest/filtertest2.i3.bz2")
i = 0
k = 0
j = 0
while i3f.more():
	frame = i3f.pop_physics()
	j = j + 1
	f1 = frame["QFilterMask"]["EHEFilter_12"].condition_passed
	f2 = frame["MyEHEFilter"].value
	if f1:
		i = i + 1
		print(f2)
	if f2:
		k = k + 1
		#print(j)
print(i, k)
"""

