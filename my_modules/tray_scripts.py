"""
Collection of icetray modules, segments, classes and functions; primarily to construct filterscipts and modules to test these.
Some of them are unnecessary but I'll leave them here cause they give a breef insight in the workings of I3 modules.
"""

from icecube import icetray, dataclasses
from icecube.filterscripts import filter_globals
import json
import math

def which_split(split_name = None,split_names=[]):
	"""
	Function to select sub_event_streams on which modules shall run on
	
	Parameters
	----------
	split_name: str (optional)
		name of sub_event_stream
	split_names: list (optional)
		list of names of sub_event_streams
	
	Return
	------
	which_split: bool
		True if stream is of chosen type of sub_event_streams, else False
	"""
    	if split_name:
    		split_names.append(split_name)
    	def which_split(frame):
    	    	if len(split_names)==0:
    	        	print "Select a split name in your If...Running module anyway"
    	        	return True
    	    	if frame.Stop == icetray.I3Frame.Physics:
    	        	return (frame['I3EventHeader'].sub_event_stream in split_names)
    	    	else:
    	        	return False
    	return which_split



class CheckFilter(icetray.I3ConditionalModule):
	"""
	Checks and counts if filters were passed like previous ones in 'QFilterMask'
	and writes the results as a dict into a json file
	
	Parameters
	----------
	outfilename: str,
		name of outputfile
	filter_key: str,
		name of key in 'QFilterMask'
	test_key: str,
		name of test key
	
	Return
	------
	json file containing resulting dict
	"""
	def __init__(self,context):
		icetray.I3ConditionalModule.__init__(self, context)
		self.AddParameter("outfilename","outfilename","")
		self.AddParameter("filter_key","filter_key","")
		self.AddParameter("test_key","test_key","")
	def Configure(self):
		self.outfile = self.GetParameter("outfilename")
		self.filter_key = self.GetParameter("filter_key")
		self.test_key = self.GetParameter("test_key")
		self.filter_key_count, self.test_key_count, self.both_count = 0, 0, 0
	def Physics(self, frame):
		if frame.Has(self.test_key):
			if frame[self.test_key].value:
				self.test_key_count += 1
				if frame["QFilterMask"][self.filter_key].condition_passed:
					self.both_count += 1
			if frame["QFilterMask"][self.filter_key].condition_passed:
				self.filter_key_count += 1
			self.PushFrame(frame)
	def Finish(self):
		out_dict = {self.test_key:self.test_key_count, self.filter_key:self.filter_key_count, "both":self.both_count}
                with open(self.outfile, "w") as outf:
                        json.dump(out_dict, fp=outf, indent=2)
                print("Wrote output file to:\n ", self.outfile)
	
			
		

class PhysicsCopyTriggers(icetray.I3ConditionalModule):
	from icecube import dataclasses
	def __init__(self, context):
		icetray.I3ConditionalModule.__init__(self, context)
		self.AddOutBox("OutBox")
	def Configure(self):
		pass
	def Physics(self, frame):
		if frame.Has(filter_globals.qtriggerhierarchy):
			myth = frame.Get(filter_globals.qtriggerhierarchy)
		     	if frame.Has(filter_globals.triggerhierarchy):
			#	icetray.logging.log_error( "%s: triggers in frame, already run??" % self.name,
			#					 unit="PhysicsCopyTriggers" )
		  		pass
			else:
				frame.Put(filter_globals.triggerhierarchy, myth)
		else:
			icetray.logging.log_error( "%s: Missing QTriggerHierarchy input" % self.name, 
								unit="PhysicsCopyTriggers" )
		self.PushFrame(frame)



class rename_q_frame_key(icetray.I3ConditionalModule):
	"""
	module to rename 'OfflinePulses' to 'InIcePulses' in Q-Frames,
	unfortunatly there is no rename module for Q-frames, so here we go
	"""
	def __init__(self, ctx):
		icetray.I3ConditionalModule.__init__(self,ctx)
		self.AddParameter("old_key","old_key","")
                self.AddParameter("new_key","new_key","")
        def Configure(self):
                self.old_key = self.GetParameter("old_key")       
                self.new_key = self.GetParameter("new_key")      
	def DAQ(self, frame):
		if not frame.Has(self.new_key):
			frame[self.new_key] = frame[self.old_key]
		self.PushFrame(frame)

class rename_q_frame_hier(icetray.I3ConditionalModule):
        """
        module to rename 'OfflinePulses' to 'InIcePulses' in Q-Frames,
        unfortunatly there is no rename module for Q-frames, so here we go
        """
        def __init__(self, ctx):
                icetray.I3ConditionalModule.__init__(self,ctx)
        def DAQ(self, frame):
                if not frame.Has("DSTTriggers"):
                        frame["DSTTriggers"] = frame["I3TriggerHierarchy"]
                self.PushFrame(frame)


class rename_SubEventStreamName(icetray.I3ConditionalModule):
	"""
	module to rename the 'SubEventStreamName' to 'InIceSplit'
	"""
	def __init__(self,ctx):
		icetray.I3ConditionalModule.__init__(self,ctx)
	def Physics(self,frame):
		frame["I3EventHeader"].sub_event_stream = "InIceSplit"
		self.PushFrame(frame)
		#frame["I3EventHeader"].sub_event_stream = "InIceSplit"	
		#self.PushFrame(frame)

class push_test(icetray.I3ConditionalModule):
	def __init__(self,context):
		icetray.I3ConditionalModule.__init__(self,context)
	def Configure(self):
		self.some_number, self.another_number = 0,0
	def Physics(self,frame):
		self.some_number +=1
		if frame["QFilterMask"]["EHEFilter_12"].condition_passed:
		#	self.another_number +=1
			self.PushFrame(frame)
		#if not frame["QFilterMask"]["EHEFilter_12"].condition_passed:
		#	self.another_number +=1
		#	self.PushFrame(frame)
	def Finish(self):
		print("push in if: ",self.some_number)

class push_test2(icetray.I3ConditionalModule):
        def __init__(self,context):
                icetray.I3ConditionalModule.__init__(self,context)
        def Configure(self):
                self.some_number2 = 0
        def Physics(self,frame):
    		self.some_number2 +=1
                self.PushFrame(frame)
        def Finish(self):
                print("push in normal: ",self.some_number2)

class test_run_alert_filter(icetray.I3ConditionalModule):
	def __init__(self, context):
		icetray.I3ConditionalModule.__init__(self,context)
		self.AddParameter("outfilename","outfilename","")
	def Configure(self):
		self.frame_count = 0
		self.outfile = self.GetParameter("outfilename")
	def Physics(self,frame):
		self.frame_count += 1
		self.PushFrame
	def Finish(self):
		out_dict = {"frame_count":self.frame_count}
		with open(self.outfile, "w") as outf:
                        json.dump(out_dict, fp=outf, indent=2)
		print("Wrote output file to:\n ", self.outfile)
				
class ehe_collector(icetray.I3ConditionalModule):
	"""
	Testversion of ehe_collector, writes out keys of ehe, my_ehe and ehe alerts for comparision

	Parameters
	----------
	outfilename: str
		name of savefile for ehe
	outfilename2: str
		name of savefile for my ehe
	outfilename3: str
		name of savefile for my ehe alert
	
	Return
	------
	savefiles for the event types with energy, run_id, event_id
	"""
        def __init__(self, ctx):
                icetray.I3ConditionalModule.__init__(self, ctx)
                self.AddParameter("outfilename","outfilename","")
		self.AddParameter("outfilename2","outfilename2","")
		self.AddParameter("outfilename3","outfilename3","")

        def Configure(self):
                self.outfile = self.GetParameter("outfilename")
		self.second_outfile = self.GetParameter("outfilename2")
		self.alert_outfile = self.GetParameter("outfilename3")
                self.run_id = []
                self.event_id = []
                # extra stuff Torben saved
                self.energy = []        # only accessible through the prior use of weighting ["MCPrimary"]
		
		# for sanity checks
		self.my_run_id = []
		self.my_event_id = []
		self.my_energy = []
		
		# alert_filter
		self.alert_run_id = []
		self.alert_event_id = []
		self.alert_energy = []

        def Physics(self, frame):
                try:
                        ehe_like = frame["QFilterMask"]["EHEFilter_12"].condition_passed
                except KeyError:
                        try:
                                ehe_like = frame["FilterMask"]["EHEFilter_11"].condition_passed
                        except KeyError:
                                try:
                                        ehe_like = frame["FilterMask"]["EHEFilter_10"].condition_passed
                                except KeyError:
                                        print("Unable to find ehe_filter, setting it to true")
                                        ehe_like = True
		
		try:
			own_ehe_like = frame["MyEHEFilter"].value
		except KeyError:
			own_ehe_like = 0
		
		try:
			ehe_alert = frame["EHEAlertFilter"].value
		except KeyError:
			ehe_alert = 2 # for distinguishing reasons uwu
		
                if ehe_like:
                        evt_header = frame["I3EventHeader"]
                        prim = frame["MCPrimary"]

                        self.run_id.append(evt_header.run_id)
                        self.event_id.append(evt_header.event_id)

                        self.energy.append(prim.energy)

                        #self.PushFrame(frame)
		
		if own_ehe_like:
			evt_header = frame["I3EventHeader"]
			prim = frame["MCPrimary"]
			self.my_run_id.append(evt_header.run_id)
			self.my_event_id.append(evt_header.event_id)
			self.my_energy.append(prim.energy)
		
		if ehe_alert == 1:
			evt_header = frame["I3EventHeader"]
			prim = frame["MCPrimary"]
			self.alert_run_id.append(evt_header.run_id)
			self.alert_event_id.append(evt_header.event_id)
			self.alert_energy.append(prim.energy)
		
		self.PushFrame(frame)

        def Finish(self):
                out_dict = {"energy": self.energy, "run_id": self.run_id, "event_id": self.event_id}
                out_dict2 = {"energy": self.my_energy, "run_id": self.my_run_id, "event_id": self.my_event_id}
		out_dict3 = {"energy": self.alert_energy, "run_id": self.alert_run_id, "event_id": self.alert_event_id}
		with open(self.outfile, "w") as outf:
                        json.dump(out_dict, fp=outf, indent=2)
		with open(self.second_outfile, "w") as outf:
			json.dump(out_dict2, fp=outf, indent=2)
		with open(self.alert_outfile, "w") as outf:
			json.dump(out_dict3, fp=outf, indent=2)
                print("Wrote output file to:\n ", self.outfile, self.second_outfile, self.alert_outfile)


@icetray.traysegment
def MyPreCalibration_IC86_2012(tray, name="", If=lambda f:True):
	"""
	Segment I cobbled together to generate necessary keys for the ehe calibration,
	works for IC86_2011 to GFU 2015 sample.
	Somehow IC79 still refuses to bow to my will...
	"""
	from icecube import icetray, dataclasses
	from icecube.filterscripts import filter_globals

	def rename(frame):
                """ Rename for older files not having 'InIcePulses' key """
                if not frame.Has("InIcePulses"): # not python-ish, but understandable
                        frame["InIcePulses"] = frame["OfflinePulses"]
                return True
	
	def CheckIfNoDSTTriggers(frame):
		"""
		Checks if frame has 'DSTTriggers'
		"""
		if frame.Has("DSTTriggers"):
			return False
		return True
	
	
	def TriggerPacker(frame):
        	"""
        	Create a compressed representation of the trigger hierarchy,
        	using the position of the TriggerKey in
        	I3DetectorStatus::triggerStatus to identify each trigger.
		'I3TriggerHierarchy' is old these days
        	"""
        	triggers = frame["I3TriggerHierarchy"]
        	status = frame['I3DetectorStatus']
        	packed = dataclasses.I3SuperDSTTriggerSeries(triggers, status)
        	frame['DSTTriggers'] = packed
	
	DAQ = [icetray.I3Frame.DAQ]
	
	# TriggerPacker for those who really care
	#tray.AddModule(TriggerPacker, "TriggerPacker",
	#		Streams = DAQ,
	#		If = CheckIfNoDSTTriggers
	#		)
	
	# rename triggerhierarchy, we can create real dsttriggers
	# but since it is just a compact version of the old triggers
	# just renaming the old one works just as fine
	tray.AddModule(rename_q_frame_key, "Rename_Trigger_Hier",
			old_key = "I3TriggerHierarchy",
			new_key = "DSTTriggers",
			If = CheckIfNoDSTTriggers
			)
	
	tray.AddModule(rename, "rename_offline_pulses"
			)
	
        # old files name 'InIcePulses' 'OfflinePulses', rename them in the q-frame
        tray.AddModule(rename_q_frame_key, "rename_pulses",
			old_key = "OfflinePulses",
			new_key = "InIcePulses"
			)

	# I3 portia only works on 'InIceSplit' + we need the time window, which is
	# never really mentioned where to get it from, but the last line will do the trick
	tray.AddModule("I3TriggerSplitter", "InIceSplit",
			SubEventStreamName = "InIceSplit",
			TrigHierName = "DSTTriggers",
			InputResponses = ["InIcePulses"],
			OutputResponses = ["SplitInIcePulses"],
			#InputResponses = ["InIceDSTPulses", "InIcePulses"],
			#OutputResponses = ["SplitInIceDSTPulses", "SplitInIcePulses"],
			WriteTimeWindow = True,
			)
	
	# Lets try this here:
	# TriggerCheck important for EHEFlag, setting the things we need
        tray.AddModule("TriggerCheck_13", "TriggerChecker",
                        I3TriggerHierarchy = "DSTTriggers",
                        InIceSMTFlag = "InIceSMTTriggered")
	
	
	# CleanInIceRawData
        icetray.load("DomTools", False)
        tray.AddModule("I3DOMLaunchCleaning", "I3LCCleaning",
                        InIceInput = "InIceRawData",
                        InIceOutput = "CleanInIceRawData",
                        IceTopInput = "IceTopRawData",
                        IceTopOutput = "CleanIceTopRawData",
#                       If = which_split(split_names = ["InIceSplit", "nullsplit"])
                        )

	# seeding which might be unnecessary here
	from icecube.icetray import I3Units	
	from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService
    	seededRTConfig = I3DOMLinkSeededRTConfigurationService(
                        ic_ic_RTRadius              = 150.0*I3Units.m,
                        ic_ic_RTTime                = 1000.0*I3Units.ns,
                        treat_string_36_as_deepcore = False,
                        useDustlayerCorrection      = False,
                        allowSelfCoincidence        = True
                     	)

	tray.AddModule('I3SeededRTCleaning_RecoPulseMask_Module', 'North_seededrt',
       			InputHitSeriesMapName  = 'SplitInIcePulses',
        		OutputHitSeriesMapName = 'SRTInIcePulses',
        		STConfigService        = seededRTConfig,
        		SeedProcedure          = 'HLCCoreHits',
        		NHitsThreshold         = 2,
        		MaxNIterations         = 3,
        		Streams                = [icetray.I3Frame.Physics],
        		If = If
    			)


# segment copied and modified from https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/meta-projects/combo/trunk/filterscripts/python/offlineL2/level2_EHE_Calibration.py
@icetray.traysegment
def MyEHECalibration_IC86_2012(tray, name="", inPulses = 'CleanInIceRawData',
                   outATWD = 'EHECalibratedATWD_Wave', outFADC = 'EHECalibratedFADC_Wave',
		   PreIf = lambda f: True,
                   If = lambda f: True,):
	from icecube import icetray, dataclasses, DomTools, ophelia, WaveCalibrator
	from icecube.icetray import OMKey
	
	tray.AddSegment(MyPreCalibration_IC86_2012, "GeneratingKeysForCalibration",
			If = PreIf)
	#**************************************************************
        ### Run WaveCalibrator w/o droop correction. DeepCore DOMs are omitted.
   	### Split waveforms up into two maps FADC and ATWD (highest-gain unsaturated channel)
   	#***
	#***********************************************************
	# temporal. should not be needed in the actual script
	#tray.AddModule("I3EHEEventSelector", name + "inicePframe",setCriteriaOnEventHeader = True, If=If)
	# This may not be needed if the frame have HLCOfflineCleanInIceRawData
	# Actually, we like to have the bad dom cleaned launches here
	tray.AddModule( "I3LCCleaning", name + "OfflineInIceLCCleaningSLC",
	                InIceInput = inPulses, 
	                InIceOutput = "HLCOfflineCleanInIceRawData",  # ! Name of HLC-only DOMLaunches
	                InIceOutputSLC = "SLCOfflineCleanInIceRawData",  # ! Name of the SLC-only DOMLaunches
	                If = If,)
	#**************************************************************
	# removing Deep Core strings
	#**************************************************************
	tray.AddModule("I3DOMLaunchCleaning", name + "LaunchCleaning",
	                InIceInput = "HLCOfflineCleanInIceRawData",
	                InIceOutput = "HLCOfflineCleanInIceRawDataWODC",
	                CleanedKeys = [OMKey(a,b) for a in range(79, 87) for b in range(1, 61)],
	                IceTopInput = "CleanIceTopRawData", #nk: Very important! otherwise it re-cleans IT!!!
	                IceTopOutput = "CleanIceTopRawData_EHE", #nk: okay so this STILL tries to write out IceTop.. give different name
	                If = If,)
	
	#***********************************************************
	# Calibrate waveforms without droop correction
	#***********************************************************
	tray.AddModule("I3WaveCalibrator", name + "calibrator",
	                Launches="HLCOfflineCleanInIceRawDataWODC",
	                Waveforms="EHEHLCCalibratedWaveforms",
	                ATWDSaturationMargin=123, # 1023-900 == 123
	                FADCSaturationMargin=0,
	                CorrectDroop=False,
	                WaveformRange="", #nk: don't write out Calibrated Waveform Range... already written with default name by Recalibration.py
	                If = If, )
	
	tray.AddModule("I3WaveformSplitter", name + "split",
	                Input="EHEHLCCalibratedWaveforms",
	                HLC_ATWD = outATWD,
	                HLC_FADC = outFADC,
			SLC = "EHECalibratedSLC",
	                Force=True,
	                PickUnsaturatedATWD=True,
	                If = If, )

def SelectOMKeySeedPulse(omkey_name, pulses_name, seed_name):
	def do(f):
        	om = f[omkey_name][0]
        	f[seed_name] = dataclasses.I3RecoPulseSeriesMapMask(f, pulses_name, lambda omkey, p_idx, p: omkey == om and p_idx == 0)
    	return do

@icetray.traysegment
def MyEHEFilter_IC86_2012(tray, name="",
                   	inATWD = 'EHECalibratedATWD_Wave', inFADC = 'EHECalibratedFADC_Wave',
		   	PreIf = lambda f: True,
		   	CalibIf = lambda f: True,
                   	If = lambda f: True):
	tray.AddSegment(MyEHECalibration_IC86_2012, "EHECalibration",
			PreIf = PreIf,
			If = CalibIf
			)
        from icecube.icetray import I3Units
    	from icecube import STTools
    	#icetray.load("libSeededRTCleaning");
    	# Create a SeededRT configuration object with the standard RT settings.
    	# This object will be used by all the different SeededRT modules of this EHE
    	# hit cleaning tray segment.
    	from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService
    	seededRTConfigEHE = I3DOMLinkSeededRTConfigurationService(
    		ic_ic_RTRadius              = 150.0*I3Units.m,
    		ic_ic_RTTime                = 1000.0*I3Units.ns,
		treat_string_36_as_deepcore = False,
  		useDustlayerCorrection      = True, # EHE use the dustlayer correction!
    	    	allowSelfCoincidence        = True
    	)
    	#***********************************************************
    	# portia splitter
    	# This module takes the splitted start time and end time and makes split DOM map
    	#***********************************************************
    	tray.AddModule("I3PortiaSplitter", name + "EHE-SplitMap-Maker",
    	              	DataReadOutName="HLCOfflineCleanInIceRawDataWODC",
    	                SplitDOMMapName="splittedDOMMap",
    	                SplitLaunchTime=True, 
    	                TimeWindowName = "TriggerSplitterLaunchWindow",
    	                If = If
			)
    	#***************************************************************                                   
    	#     Portia Pulse process  with the split DOM map for SeededRT seed
    	#***************************************************************                                   
    	tray.AddModule("I3Portia", name + "pulseSplitted",
    	                SplitDOMMapName = "splittedDOMMap",
    	                OutPortiaEventName = "EHEPortiaEventSummary",
    	                ReadExternalDOMMap=True,
    	                MakeIceTopPulse=False,
    	                ATWDPulseSeriesName = "EHEATWDPulseSeries",
    	                ATWDPortiaPulseName = "EHEATWDPortiaPulse",
    	                ATWDWaveformName = inATWD,
    	                ATWDBaseLineOption = "eheoptimized",
    	                FADCBaseLineOption = "eheoptimized",
    	                ATWDThresholdCharge = 0.1*I3Units.pC,
    	                ATWDLEThresholdAmplitude = 1.0*I3Units.mV,
    	                UseFADC = True,
    	                FADCPulseSeriesName = "EHEFADCPulseSeries",
    	                FADCPortiaPulseName = "EHEFADCPortiaPulse",
    	                FADCWaveformName = inFADC,
    	                FADCThresholdCharge = 0.1*I3Units.pC,
    	                FADCLEThresholdAmplitude = 1.0*I3Units.mV,
    	                MakeBestPulseSeries = True,
    	                BestPortiaPulseName = "EHEBestPortiaPulse",
    	                PMTGain = 10000000,
    	                If = which_split(split_name='InIceSplit')
			)
   
    	tray.AddModule("I3PortiaEventOMKeyConverter", name + "portiaOMconverter",
    	                InputPortiaEventName = "EHEPortiaEventSummary",
    	                OutputOMKeyListName = "LargestOMKey",
    	                If = If
    	                )

    	#***************************************************************
    	#     EHE SeededRTCleaning for DOMmap
    	#***************************************************************
    	#---------------------------------------------------------------------------
    	# The splittedDOMMap frame object is an I3MapKeyVectorDouble
    	# object. In order to use STTools in the way SeededRTCleaning did it, we
    	# need to convert this into an I3RecoPulseSeriesMap where the time of each
    	# pulse is set to the double value of the splittedDOMMap frame object.
    	def I3MapKeyVectorDouble_to_I3RecoPulseSeriesMap(f):
    	    	i3MapKeyVectorDouble = f['splittedDOMMap']
    	    	i3RecoPulseSeriesMap = dataclasses.I3RecoPulseSeriesMap()
    	    	for (k,l) in i3MapKeyVectorDouble.items():
    	    		pulses = dataclasses.I3RecoPulseSeries()
    	    	    	for d in l:
    	    	        	p = dataclasses.I3RecoPulse()
    	    	        	p.time = d
    	    	        	pulses.append(p)
    	    	    	i3RecoPulseSeriesMap[k] = pulses
    	    	f['splittedDOMMap_pulses'] = i3RecoPulseSeriesMap
    	"""
	tray.AddModule(I3MapKeyVectorDouble_to_I3RecoPulseSeriesMap, name+'IsolatedHitsCutSRT_preconvert',
    			If = If
    			)
    	# Also we need the seed pulse, which is the first pulse of the OM which is
    	# named "LargestOMKey".
    	tray.AddModule(SelectOMKeySeedPulse(
    	        	omkey_name  = 'LargestOMKey',
    	        	pulses_name = 'splittedDOMMap_pulses',
    	        	seed_name   = 'IsolatedHitsCutSRT_seed'
    	    		),
    	    		name+'IsolatedHitsCutSRT_seed_pulse_selector',
    	    		If = If
    			)
    	# Do SeededRT cleaning.
    	tray.AddModule('I3SeededRTCleaning_RecoPulse_Module', name+'IsolatedHitsCutSRT',
    	    		InputHitSeriesMapName  = 'splittedDOMMap_pulses',
    	    		OutputHitSeriesMapName = 'splittedDOMMapSRT_pulses',
    	    		STConfigService        = seededRTConfigEHE,
    	    		SeedProcedure          = 'HitSeriesMapHitsFromFrame',
    	    		SeedHitSeriesMapName   = 'IsolatedHitsCutSRT_seed',
    	    		MaxNIterations         = -1,
    	    		Streams                = [icetray.I3Frame.Physics],
    	    		If = If
    			)
    	# Convert the resulting I3RecoPulseSeriesMap back into an
    	# I3MapKeyVectorDouble object.
    	def I3RecoPulseSeriesMap_to_I3MapKeyVectorDouble(f):
    		i3RecoPulseSeriesMap = f['splittedDOMMapSRT_pulses']
    	    	i3MapKeyVectorDouble = dataclasses.I3MapKeyVectorDouble()
    	    	for (k,l) in i3RecoPulseSeriesMap.items():
    	        	doubles = [p.time for p in l]
    	        	i3MapKeyVectorDouble[k] = doubles
    	    	f['splittedDOMMapSRT'] = i3MapKeyVectorDouble
    	tray.AddModule(I3RecoPulseSeriesMap_to_I3MapKeyVectorDouble, name+'IsolatedHitsCutSRT_postconvert',
    	    		If = If
    			)
	"""
	# Lets try this, maybe it works (hopefully)
	# make EHE decision based on 10**3 npe
        tray.AddModule("I3FilterModule<I3EHEFilter_13>","EHEfilter",
                        TriggerEvalList=["InIceSMTTriggered"],
                        DecisionName    = "MyEHEFilter",
                        DiscardEvents   = False,
                        PortiaEventName = "EHEPortiaEventSummary",
                        Threshold       = pow(10,3.0),
                        If = which_split(split_name='InIceSplit')
                        )
	"""
	tray.AddModule(CheckFilter,"CheckFilter",
			outfilename = "FilterCheckPerModule",
			filter_key = "EHEFilter_12",
			test_key = "MyEHEFilter",
			If = which_split(split_name='InIceSplit')
			)
	"""
# function and segment below are copied from 
# https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/meta-projects/combo/trunk/filterscripts/python/ehealertfilter.py
#  EHEAlertFilter -- Only run if NPE threshold high enough                             
# Note: this threshold higher than EHEFilter   
def RunEHEAlertFilter(frame):
    	if 'EHEPortiaEventSummary' not in frame:
    	    	return False
    	# Check EHE filter first, it's low rate to start...bool in frame here         
    	try:
		ehefilterflag = frame["QFilterMask"]["EHEFilter_12"].condition_passed
    	except KeyError:
		try:
			ehefilterflag = frame["FilterMask"]["EHEFilter_11"].condition_passed
		except KeyError:
			try:
				ehefilterflag = frame["FilterMask"]["EHEFilter_10"].condition_passed
			except KeyError:
				return False
	if not ehefilterflag:
    	    	return False
    	npe    = frame['EHEPortiaEventSummary'].GetTotalBestNPEbtw()
    	if math.isnan(npe): return False
    	if npe <= 0:        return False
    	lognpe = math.log10(npe)
    	return lognpe >= 3.6 


@icetray.traysegment
def EHEAlertFilter(tray, name, 
                   pulses         = 'CleanedMuonPulses',
                   portia_pulse   = 'EHEBestPortiaPulse',   # Maybe this should be 'Pole'
                   portia_summary = 'EHEPortiaEventSummary',
                   split_dom_map  = 'splittedDOMMap',
                   If = lambda f: True):
   
	# Some necessary stuff
	from icecube import dataclasses, linefit
    	from icecube.icetray import OMKey, I3Units
    	from icecube.filterscripts import filter_globals
    	icetray.load("filterscripts",False)
    	icetray.load("portia",False)
    	icetray.load("ophelia",False)
    	from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService
    	"""
	# Get the largest OM Key in the frame
    	tray.AddModule("I3PortiaEventOMKeyConverter", name + "portiaOMconverter",
    	               InputPortiaEventName = portia_summary,
    	               OutputOMKeyListName = "LargestOMKey",
    	               If = (If and (lambda f:RunEHEAlertFilter(f)))
    	               )
	"""
    	# Static time window cleaning
    	tray.AddModule("I3EHEStaticTWC", name + "portia_static",
    	               InputPulseName = portia_pulse,
    	               InputPortiaEventName = portia_summary,
    	               outputPulseName = name + "BestPortiaPulse_BTW",
    	               TimeInterval = 500.0 * I3Units.ns, #650, now no interval cut
    	               TimeWidnowNegative =  -2000.0 * I3Units.ns,
    	               TimeWindowPositive = 6400.0 * I3Units.ns,
    	               If = (If and (lambda f:RunEHEAlertFilter(f)))
    	               )
    	# Convert portia pulses
    	tray.AddModule("I3OpheliaConvertPortia", name + "portia2reco",
    	               InputPortiaPulseName = name + "BestPortiaPulse_BTW",
    	               OutputRecoPulseName = name + "RecoPulseBestPortiaPulse_BTW",
    	               If = (If and (lambda f:RunEHEAlertFilter(f)))
    	               )
    	# Run delay cleaning
    	tray.AddModule("DelayCleaningEHE", name + "DelayCleaning",
    	               InputResponse = name + "RecoPulseBestPortiaPulse_BTW",
    	               OutputResponse = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay",
    	               Distance = 200.0 * I3Units.m, #156m default
    	               TimeInterval = 1800.0 * I3Units.ns, #interval 1.8msec
    	               TimeWindow = 778.0 * I3Units.ns, #778ns default
    	               If = (If and (lambda f:RunEHEAlertFilter(f)))
    	               )
    	seededRTConfigEHE = I3DOMLinkSeededRTConfigurationService(
    	    ic_ic_RTRadius              = 150.0*I3Units.m,
    	    ic_ic_RTTime                = 1000.0*I3Units.ns,
    	    treat_string_36_as_deepcore = False,
    	    useDustlayerCorrection      = True, # EHE use the dustlayer correction!
    	    allowSelfCoincidence        = True
    	    )
   
    	tray.AddModule('I3SeededRTCleaning_RecoPulse_Module', name+'Isolated_DelayClean',
    	               InputHitSeriesMapName  = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay",
    	               OutputHitSeriesMapName = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay_RT",
    	               STConfigService        = seededRTConfigEHE,
    	               SeedProcedure          = 'HLCCOGSTHits',
    	               MaxNIterations         = -1, # Infinite.
    	               Streams                = [icetray.I3Frame.Physics],
    	               If = (If and (lambda f:RunEHEAlertFilter(f)))
    	               )                   
    	# Huber fit
    	tray.AddModule("HuberFitEHE", name + "HuberFit",
    	               Name = "HuberFit",
    	               Distance = 180.0 * I3Units.m, #153m default
    	               InputRecoPulses = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay_RT",
    	               If = (If and (lambda f:RunEHEAlertFilter(f)))
    	               )
    	# Debiasing of pulses
    	tray.AddModule("DebiasingEHE", name + "Debiasing",
    	               OutputResponse = name + "debiased_BestPortiaPulse_CleanDelay",
    	               InputResponse = name + "RecoPulseBestPortiaPulse_BTW_CleanDelay_RT",
    	               Seed = "HuberFit",
    	               Distance = 150.0 * I3Units.m,#116m default
    	               If = (If and (lambda f:RunEHEAlertFilter(f)))
    	               )
   
    	# Convert back to portia pulses to be fed to ophelia
    	tray.AddModule("I3OpheliaConvertRecoPulseToPortia", name + "reco2portia",
    	               InputRecoPulseName = name + "debiased_BestPortiaPulse_CleanDelay",
    	               OutputPortiaPulseName = name + "portia_debiased_BestPortiaPulse_CleanDelay",
    	               If = (If and (lambda f:RunEHEAlertFilter(f)))
    	               )
    	# Run I3EHE First guess module and get final result
    	tray.AddModule("I3EHEFirstGuess", name + "reco_improvedLinefit",
    	               MinimumNumberPulseDom = 8,
    	               InputSplitDOMMapName = split_dom_map,
    	               OutputFirstguessName = "PoleEHEOphelia_ImpLF", # Final result
    	               OutputFirstguessNameBtw = name + "OpheliaBTW_ImpLF", # Don't Use
    	               InputPulseName1 = name + "portia_debiased_BestPortiaPulse_CleanDelay",
    	               ChargeOption = 1,
    	               LCOption =  False, 
    	               InputPortiaEventName =portia_summary,
    	               OutputParticleName = "PoleEHEOpheliaParticle_ImpLF", # Final Result
    	               OutputParticleNameBtw = name + "OpheliaParticleBTW_ImpLF", # Don't Use
    	               NPEThreshold = 0.0,
    	               If = (If and (lambda f:RunEHEAlertFilter(f)))
    	               )
   
    	# Run the alert filter
    	tray.AddModule("I3FilterModule<I3EHEAlertFilter_15>","EHEAlertFilter",
    	               TriggerEvalList= ["InIceSMTTriggered"],
    	               DecisionName = "EHEAlertFilter",
    	               DiscardEvents = False,
    	               PortiaEventName = portia_summary,
    	               EHEFirstGuessParticleName = "PoleEHEOpheliaParticle_ImpLF",
    	               EHEFirstGuessName = "PoleEHEOphelia_ImpLF",
    	               If = (If and (lambda f: 'PoleEHEOphelia_ImpLF' in f) # First Guess can fail
    	                     and (lambda f:RunEHEAlertFilter(f)))
    	               )
    	# Again for Heartbeat
    	tray.AddModule("I3FilterModule<I3EHEAlertFilter_15>","EHEAlertFilterHB",
    	               TriggerEvalList = ["InIceSMTTriggered"],
    	               DecisionName = "EHEAlertFilterHB",
    	               DiscardEvents = False,
    	               PortiaEventName = portia_summary,
    	               EHEFirstGuessParticleName = "PoleEHEOpheliaParticle_ImpLF",
    	               EHEFirstGuessName = "PoleEHEOphelia_ImpLF",
    	               Looser = True, # For Heartbeat ~ 4 events / day
    	               #Loosest = True, # For PnF testing ~ 4 events / hour
    	               If = (If and (lambda f: 'PoleEHEOphelia_ImpLF' in f) # First Guess can fail
    	                     and (lambda f:RunEHEAlertFilter(f)))
    	               )




@icetray.traysegment
def MyPreCalibration(tray, name="", If=lambda f:True):
	def rename(frame):
		""" Rename for older files not having 'InIcePulses' key """
		if not frame.Has("InIcePulses"): # not python-ish, but understandable
			frame["InIcePulses"] = frame["OfflinePulses"]
		return True
	
	# rename SubEventStream for trigger
	#tray.AddModule(rename_SubEventStreamName, "rename_sub_event_stream")

	# call the trigger something else for the Q frame
	# filter_globals has default values like triggerhierarchy and qtriggerhierarchy
	tray.AddModule("Rename", name + '_trigrename', 
			keys=[filter_globals.triggerhierarchy,
			filter_globals.qtriggerhierarchy]
			)
	
	# old files name 'InIcePulses' 'OfflinePulses', rename them in the q-frame
	tray.AddModule(rename_q_frame_key, "rename_q_frame_key")

	"""
	tray.AddModule("Rename", "rename_pulses",
			keys=["OfflinePulses","InIcePulses"])
		
	tray.AddModule(rename, "RenameToInIcePulses"
			)
	"""
	# all we need is TriggerSplitterLaunchWindow
	# this splits the frames again, maybe we can do something about it
		
	tray.AddModule("I3TriggerSplitter", "InIceTriggerSplit",
			SubEventStreamName = "InIceSplit",	# name the split will have, setting it manually cause WARN >:(
			TrigHierName=filter_globals.qtriggerhierarchy,
     		    	# Note: example takes the SuperDST pulses
			# SuperDST is compressed InIce version, so idc
                    	InputResponses=["InIcePulses"],
                    	OutputResponses=["SplitUncleanedInIcePulses"],
     		    	WriteTimeWindow = True  # Needed for EHE
                   	)
		
	# rename again for new splits, there has to be another way though
	#tray.AddModule(rename_SubEventStreamName, "rename_sub_event_stream_again")
	#tray.AddModule(rename_SubEventStreamName, "rename_duh")
	
	# This part is taken from OnlineCalibration to
	# generate CleanInIceRawData
	# url: https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/meta-projects/combo/trunk/DomTools/resources/examples/exampleI3DOMLaunchCleaning.py
	icetray.load("DomTools", False)
	tray.AddModule("I3DOMLaunchCleaning", "I3LCCleaning",
			InIceInput = "InIceRawData",
			InIceOutput = "CleanInIceRawData",
			IceTopInput = "IceTopRawData",
			IceTopOutput = "CleanIceTopRawData",
#			If = which_split(split_names = ["InIceSplit", "nullsplit"])
			)
	
	# TriggerChecker needs I3Triggerhierarchy in nullsplit, so we copy it
	tray.AddModule(PhysicsCopyTriggers, "nullTriggerCopy",
			If = which_split(split_name="nullsplit"))
	
	# TriggerCheck important for EHEFlag, setting the things we need
	tray.AddModule("TriggerCheck_13", "TriggerChecker",
			I3TriggerHierarchy = filter_globals.triggerhierarchy,
			InIceSMTFlag = "InIceSMTTriggered")
	
@icetray.traysegment
def MyEHECalibration(tray, name, If=lambda f:True, PreIf=lambda f:True):
	from icecube import dataclasses, WaveCalibrator
	from icecube.filterscripts import filter_globals
	
	# pre calibration part
	tray.AddSegment(MyPreCalibration, "GeneratingKeysForCalibration",
			If = PreIf)

	# real calibration part taken from ehefilter.py
	# skip deepcore '_noDC'
	tray.AddModule("I3OMSelection<I3DOMLaunchSeries>", "skip_DeepCore",
			InputResponse = filter_globals.CleanInIceRawData,
			OmittedStrings = [79,80,81,82,83,84,85,86],
			OutputResponse = 'CleanInIceRawData_noDC',
			If = If
			)

	tray.AddModule("I3LCCleaning", "CleaningSLC",
			InIceInput = "CleanInIceRawData_noDC",
			InIceOutput = "HLCCleanInIceRawData_noDC",
			InIceOutputSLC = "SLCCleanInIceRawData_noDC",
			If = If
			)
	
	tray.AddModule("I3WaveCalibrator", "EHEWaveCalibrator",
			CorrectDroop= False,
			Launches = "HLCCleanInIceRawData_noDC",
			Waveforms = "EHECalibratedWaveforms",
			Errata = "CalibratedErrata",	# Name, containing the OMs for which the calibration failed in one or more waveforms
			WaveformRange = "EHECalibratedWaveformRange",
			If = If
			)
	
	tray.AddModule("I3WaveformSplitter", "EHEWaveformSplitter",
			Input = "EHECalibratedWaveforms",
			PickUnsaturatedATWD = True,
			HLC_ATWD = "EHECalibratedATWD",
			HLC_FADC = "EHECalibratedFADC",
			SLC = "EHECalibratedGarbage",
			Force = True,
			If = If
			)

@icetray.traysegment
def MyEHEFilter(tray, name, If=lambda f:True, CalIf=lambda f:True, PreIf=lambda f:True):
	"""
	My EHEFilter
	"""
	from icecube.icetray import I3Units
	from icecube.filterscripts import filter_globals
	
	icetray.load("portia", False)
	icetray.load("filterscripts", False) 

	tray.AddSegment(MyEHECalibration, "EHECalibration",
			If = CalIf,
			PreIf = PreIf
			)
	
	tray.AddModule("I3PortiaSplitter", "EHE-SplitMap-Maker",
			DataReadOutName = "HLCCleanInIceRawData_noDC",
#			SplitDOMMapName = "splittedDOMMap",	# default 'SplittedDOMMap'
#			SubEventStreamNames = "InIceSplit",	# Name of the P-Frame: default 'InIceSplit'; ToDo: Rename frames
			SplitLaunchTime = True,
			TimeWindowName = "TriggerSplitterLaunchWindow",	# generated by 'I3TriggerSplitter' in MyPreCalibration
			If = If
			)
	
	tray.AddModule("I3Portia", "pulseSplitted",
#			SplitDOMMapName = "splittedDOMMap",
        		OutPortiaEventName = "PoleEHESummaryPulseInfo",
                   	ReadExternalDOMMap=True,
                   	MakeIceTopPulse=False,
                   	ATWDPulseSeriesName = "EHEATWDPulseSeries",
                   	ATWDPortiaPulseName = "EHEATWDPortiaPulse",
                   	ATWDWaveformName = "EHECalibratedATWD",
                   	ATWDBaseLineOption = "eheoptimized",
                   	FADCBaseLineOption = "eheoptimized",
                   	ATWDThresholdCharge = 0.1*I3Units.pC,
                   	ATWDLEThresholdAmplitude = 1.0*I3Units.mV,
                   	UseFADC = True,
                   	FADCPulseSeriesName = "EHEFADCPulseSeries",
                   	FADCPortiaPulseName = "EHEFADCPortiaPulse",
                   	FADCWaveformName = "EHECalibratedFADC",
                   	FADCThresholdCharge = 0.1*I3Units.pC,
                   	FADCLEThresholdAmplitude = 1.0*I3Units.mV,
                   	MakeBestPulseSeries = True,
                   	BestPortiaPulseName = "EHEBestPortiaPulse",
                   	PMTGain = 10000000,
                   	If = If )
	
	# make EHE decision based on 10**3 npe
	tray.AddModule("I3FilterModule<I3EHEFilter_13>","EHEfilter",
			TriggerEvalList=["InIceSMTTriggered"],
			DecisionName    = "MyEHEFilter",
                   	DiscardEvents   = False,
                   	PortiaEventName = "PoleEHESummaryPulseInfo",
                   	Threshold       = pow(10,3.0),
                   	If = If
                   	)


## copied and modified from https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/projects/filterscripts/trunk/python/highqfilter.py
# High Charge Filter Traysegment - 2017/Pass2
@icetray.traysegment
def HighQFilter(tray, name, pulses='SplitInIcePulses', If=lambda f: True, PreIf=lambda f: True):
    	"""
    	Traysegment for a high charge filter (formerly EHE).
    	"""   
    	# load needed libs, the "False" suppresses any "Loading..." messages
    	#from icecube.filterscripts import filter_globals
    	#icetray.load("filterscripts",False) 
    	from icecube import VHESelfVeto

	
    	HighQFilter_threshold = 1000.0 #from the website
   
    	TriggerEvalList = ["InIceSMTTriggered"] # work on SMT8 triggers
    	def If_with_triggers(frame):
        	if not If(frame):
            		return False
        	for trigger in TriggerEvalList:
            		if frame[trigger].value:
                		return True
        	return False
   
   	# apply the veto
    	tray.AddModule('HomogenizedQTot', name+'_qtot_total',
        		Pulses=pulses,
        		Output='HESE_CausalQTot',
        		If = If_with_triggers)
    	tray.AddModule("I3FilterModule<I3HighQFilter_17>",
                   	name+"HighQFilter",
                   	MinimumCharge = HighQFilter_threshold,
                   	ChargeName = 'HESE_CausalQTot',
                   	TriggerEvalList = TriggerEvalList,
                   	DecisionName = "HighQFilter",
                   	If = If)


