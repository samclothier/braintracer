"""
Copyright (C) 2021-2023  Sam Clothier

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import importlib # import other braintracer files using relative path, agnostic to directory inheritance
bt_path = '.'.join(__name__.split('.')[:-1]) # get module path (folder containing this file)
btf = importlib.import_module(bt_path+'.file_management')
bt = importlib.import_module(bt_path+'.analysis')
btp = importlib.import_module(bt_path+'.plotting')
import numpy as np



def custom_rois(areas):
    areas_title = 'Custom merged ROIs'
    area_idxs = bt.get_area_info(areas)[1]
    return area_idxs, areas_title

def summary_regions():
    areas_title = 'Summary regions'
    areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN','IO']
    area_idxs = bt.get_area_info(areas)[1]
    return area_idxs, areas_title

# Anterograde areas
#region anterograde

def inputs_antero_MF(threshold, select_norm, fluorescence):
    datasets = [i for i in bt.datasets if i.fluorescence == fluorescence]
    areas_title = "MF Inputs (anterograde)"
    areas = ['P', 'LRN', 'VNC', 'ICB']
    areas = list(map(lambda x: [bt.children_from(x, depth=0)[0]] + bt.children_from(x, depth=0)[1], areas))
    areas = [item for sublist in areas for item in sublist]
    area_idxs = [ i for i in areas if bt.area_predicate(i, threshold, select_norm, datasets) ]
    return area_idxs, areas_title

def inputs_antero_MF_roi():
    areas_title = "MF Inputs (anterograde)"
    #         <----------------------Pons---------------------------->   <------------------------------------------Medulla------------------------------------------>   <-------Midbrain------->  <-----Cortex------->
    areas = ['TRN', 'PRNc', 'PG', 'PGRN', 'PSV', 'PB', 'V', 'CS', 'NLL', 'LRN', 'VNC', 'ICB', 'PARN', 'VCO', 'MDRN', 'SPVI', 'SPVO', 'GRN', 'PRP', 'IRN', 'MARN', 'DCO', 'IC', 'MRN', 'SAG', 'PAG', 'AUD', 'ORB', 'ECT']
    area_idxs = bt.get_area_info(areas)[1]
    return area_idxs, areas_title

def inputs_antero_MF_roi_crop():
    areas_title = "MF Inputs (anterograde)"
    #         <----------------------Pons---------------------->   <------------------------------------------Medulla------------------------------------------>  <----Midbrain--->
    areas = ['TRN', 'PRNc', 'PG', 'PGRN', 'PSV', 'PB', 'V', 'CS', 'LRN', 'VNC', 'ICB', 'PARN', 'VCO', 'MDRN', 'SPVI', 'SPVO', 'GRN', 'PRP', 'IRN', 'MARN', 'DCO', 'IC', 'MRN', 'PAG','IO']
    area_idxs = bt.get_area_info(areas)[1]
    return area_idxs, areas_title

def inputs_antero_CF(split):
    split_options = ['rc', 'ml', 'both']
    if split == split_options[0]:
        io_areas = ['Rostral IO', 'Caudal IO']
    elif split == split_options[1]:
        io_areas = ['Medial IO', 'Lateral IO']
    elif split == split_options[2]:
        io_areas = ['Rostral-medial IO', 'Rostral-lateral IO', 'Caudal-medial IO', 'Caudal-lateral IO']
    else:
        return None

    areas_title = f"CF Inputs (anterograde)"
    return io_areas, areas_title

def cerebellar_cortex_antero():
    areas_title = "Cbx (anterograde)"
    parent, children = bt.children_from('CBX', depth=2)
    cbx_area_idxs = children
    return cbx_area_idxs, areas_title

def mono_outputs_antero():
    areas_title = "Monosynaptic Outputs"
    parent, children = bt.children_from('CBN', depth=0)
    cb_area_idxs = children
    return cb_area_idxs, areas_title

def di_outputs_antero(threshold, select_norm, fluorescence):
    datasets = [i for i in bt.datasets if i.fluorescence == fluorescence]
    areas_title = "Disynaptic Outputs"
    area_idxs = bt.children_from('root', depth=0)[1]
    area_idxs = np.array([ i for i in area_idxs if bt.area_predicate(i, threshold, select_norm, datasets) ])
    area_idxs = area_idxs[area_idxs != 83] # remove IO and regions providing input
    area_idxs = [ i for i in area_idxs if bt.get_area_info(i)[0][0][0].isupper() ] # remove tracts (they start lower-case in the atlas)
    areas_to_remove = ['CB', 'P', 'LRN', 'VNC', 'ICB']
    areas_to_remove = list(map(lambda x: [bt.children_from(x, depth=0)[0]] + bt.children_from(x, depth=0)[1], areas_to_remove))
    areas_to_remove = [item for sublist in areas_to_remove for item in sublist]
    area_idxs = list(filter(lambda x: x not in areas_to_remove, area_idxs)) # remove areas above
    return area_idxs, areas_title

def di_outputs_antero_roi(threshold, select_norm, fluorescence): # threshold = 0.1
    datasets = [i for i in bt.datasets if i.fluorescence == fluorescence]
    areas_title = "Disynaptic Outputs (ROIs)"
    areas = ['TH', 'SCs', 'SCm', 'RN', 'PAG', 'VTA']
    area_idxs = list(map(lambda x: [bt.children_from(x, depth=0)[0]] + bt.children_from(x, depth=0)[1], areas))
    area_idxs = [item for sublist in area_idxs for item in sublist]
    area_idxs = [ i for i in area_idxs if bt.area_predicate(i, threshold, select_norm, datasets) ]
    return area_idxs, areas_title

#endregion


# Retrograde areas
#region Retrograde

def mono_inputs_retro():
    areas_title = "Monosynaptic Inputs"
    parent, children = bt.children_from('IO', depth=0)
    io_area_idxs = [parent] + children
    return io_area_idxs, areas_title

def di_inputs_retro(threshold, select_norm, fluorescence):
    datasets = [i for i in bt.datasets if i.fluorescence == fluorescence]
    areas_title = "Disynaptic Inputs"
    area_idxs = np.array(bt.children_from('root', depth=0)[1])
    area_idxs = area_idxs[area_idxs != 83] # remove IO
    area_idxs = [ i for i in area_idxs if bt.get_area_info(i)[0][0][0].isupper() ] # remove tracts (they start lower-case in the atlas)
    cb_areas = bt.children_from('CB', depth=0)[1]
    area_idxs = list(filter(lambda x: x not in cb_areas, area_idxs)) # remove areas in cerebellum
    dcn_areas = bt.children_from('CBN', depth=0)[1] # add DCN back in
    area_idxs = area_idxs + dcn_areas
    area_idxs = [ i for i in area_idxs if bt.area_predicate(i, threshold, select_norm, datasets) ]
    return area_idxs, areas_title

def di_inputs_retro_roi():
    areas_title = "Disynaptic Inputs (ROIs)"
    areas = ['PH', 'ZI', 'SCm', 'MRN', 'PF', 'MM', 'VTA', 'IPN', 'PRNc', 'MV', 'IP', 'SPIV', 'SPVO', 'SPVI', 'PARN']
    area_idxs = bt.get_area_info(areas)[1]
    return area_idxs, areas_title

#endregion