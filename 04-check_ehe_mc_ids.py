# coding:utf-8

"""
Apply EHE filter to sim files and store runIDs, eventIDs and MC primary
energies for all events surviving the filter.
The IDs are checked against the final level MCs to sort out any EHE like events
for sensitivity calulations.
"""

from __future__ import division, print_function

import json
import argparse

from I3Tray import *
from icecube import icetray, dataclasses, dataio
from icecube import DomTools, weighting
from icecube import VHESelfVeto

# just import everything...
from icecube import icetray, dataclasses, dataio, filterscripts, filter_tools, trigger_sim, WaveCalibrator
from icecube import phys_services, DomTools
from icecube.filterscripts import filter_globals
from icecube.phys_services.which_split import which_split
#from icecube.filterscripts.filter_globals import EHEAlertFilter
from icecube import VHESelfVeto, weighting

from icecube.icetray import I3Units


import sys

#sys.path.append("../")
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from my_modules import MyEHEFilter_IC86_2012, which_split, ehe_collector, EHEAlertFilter, HighQFilter



class hese_collector(icetray.I3ConditionalModule):
    """
    Collect run ID, event ID and MC primary energy for events surviving the
    HESE filter.
    """

    def __init__(self, ctx):
        icetray.I3ConditionalModule.__init__(self, ctx)
        self.AddParameter("outfilename", "outfilename", "")

    def Configure(self):
        # HESE total charge cut: wiki.icecube.wisc.edu/index.php/HESE-7year#Summary_of_Cuts_and_Parameters
        self.minimum_charge = 6000.

        self.outfile = self.GetParameter("outfilename")
        self.run_id = []
        self.event_id = []
        # Just save some extra stuff
        self.qtot = []
        self.energy = []

    def Physics(self, frame):
        # If HESE veto is passed, VHESelfVeto variable is False.
        # Also VHESelfVeto doesn't always write the key, which means the event
        # is vetoed.
        try:
            hese_veto = frame["VHESelfVeto"].value
        except KeyError:
            hese_veto = True
        if not hese_veto:
            # Not HESE vetoed: Candidate event
            qtot = frame["HESE_CausalQTot"].value
            if qtot >= self.minimum_charge:
                # Also over QTot cut: Winner
                evt_header = frame["I3EventHeader"]
                prim = frame["MCPrimary"]

                self.run_id.append(evt_header.run_id)
                self.event_id.append(evt_header.event_id)
                self.energy.append(prim.energy)
                self.qtot.append(qtot)

                self.PushFrame(frame)

    def Finish(self):
        out_dict = {"energy": self.energy, "run_id": self.run_id,
                    "event_id": self.event_id, "qtot": self.qtot}
        with open(self.outfile, "w") as outf:
            json.dump(out_dict, fp=outf, indent=2)
        print("Wrote output file to:\n  ", self.outfile)




def main(in_files, out_file_1, out_file_2, out_file_3, out_file_4, gcd_file, source_type):
    files = []
    files.append(gcd_file)
    if not isinstance(in_files, list):
        in_files = [in_files]
    files.extend(in_files)

    tray = I3Tray()

    # Read files
    tray.AddModule("I3Reader", "reader", Filenamelist=files)

  
    def rename(frame):
        """ Rename for older files not having 'InIcePulses' key """
        if not frame.Has('InIcePulses'):
            frame['InIcePulses'] = frame['OfflinePulses']
        return True

    if source_type == "hese":
	print("checking for hese as requested")
    	# Rename frames if InIcePulses
    	tray.AddModule(rename, 'rename')    
	
	# Create correct MCPrimary for energy
	tray.AddModule(weighting.get_weighted_primary, "weighted_primary",
                   If=lambda frame: not frame.Has("MCPrimary"))

		
    	##########################################################################
    	# Following code from hesefilter.py
    	# Prepare Pulses
    	tray.AddModule("I3LCPulseCleaning", "cleaning_HLC",
        	           OutputHLC="InIcePulsesHLC",
                	   OutputSLC="",
                 	   Input="InIcePulses",
                	   If=lambda frame: not frame.Has("InIcePulsesHLC"))
	
        # Apply HESE filter
   	tray.AddModule("VHESelfVeto",
        	          "selfveto",
                	  Pulses="InIcePulsesHLC")
	
   	# Add CausalQTot frame
   	tray.AddModule('HomogenizedQTot', 'qtot_causal',
        	          Pulses="InIcePulses",
                	  Output='HESE_CausalQTot',
                 	  VertexTime='VHESelfVetoVertexTime')
   	##########################################################################
        tray.AddModule(hese_collector, "hese_collector",
			  outfilename = out_file_1,
			  outfilename_2 = out_file_2,
			  outfilename_3 = out_file_3)
    if source_type == "ehe":
	print("checking for ehe as requested")
	
	tray.AddSegment(MyEHEFilter_IC86_2012,
        	        If = which_split(split_name='InIceSplit')
               		)
	tray.AddSegment(EHEAlertFilter,
			If = which_split(split_name='InIceSplit')
			)
	
	tray.AddSegment(HighQFilter, "my_HighQFilter",
                	If = which_split(split_name='InIceSplit')
                	)
	
	# Create correct MCPrimary for energy
	tray.AddModule(weighting.get_weighted_primary, "weighted_primary",
        	        If=lambda frame: not frame.Has("MCPrimary"))

	
        tray.AddModule(ehe_collector, "ehe_collector",
			outfilename = out_file_1,
			outfilename2 = out_file_2,
			outfilename3 = out_file_3,
			outfilename4 = out_file_4,
			If = which_split(split_name='InIceSplit') # returns empty dicts, cause no InIceSplits yet
			)
    
    tray.AddModule("TrashCan", "NacHsart")
    tray.Execute()
    tray.Finish()



###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    # Parse options and call `main`
    parser = argparse.ArgumentParser(description="Check EHE or HESE filter")
    parser.add_argument("--infiles", type=str)
    parser.add_argument("--gcdfile", type=str)
    parser.add_argument("--outfile_1", type=str)
    parser.add_argument("--outfile_2", type=str)
    parser.add_argument("--outfile_3", type=str)
    parser.add_argument("--outfile_4", type=str)

    args = parser.parse_args()

    in_files = args.infiles.split(",")
    gcd_file = args.gcdfile
    out_file_1 = args.outfile_1
    out_file_2 = args.outfile_2
    out_file_3 = args.outfile_3
    out_file_4 = args.outfile_4
    
    source_type = "ehe" # "ehe" or "hese"
    
    main(in_files, out_file_1, out_file_2, out_file_3, out_file_4, gcd_file, source_type)
                                           
