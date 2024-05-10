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
btp = importlib.import_module(bt_path+'.plotting')

import matplotlib.pyplot as plt
import shortuuid as uid
import pandas as pd
import numpy as np
from IPython.display import clear_output
from collections import Counter
from tqdm.notebook import tqdm
from matplotlib import colors
from itertools import chain
from matplotlib import cm

datasets		= []
atlas			= btf.get_atlas()
reference		= btf.get_reference()
area_indexes	= btf.open_file('structures.csv')

postsyn_region			= None # You must set a starter region to use some features
postsyn_ch				= '' # You must set the channel(s) containing starter cells (postsynaptics)
presyn_ch				= '' # And the channel(s) containing input cells (presynaptics)
presyn_regions_exclude	= [] # Presynaptic cells are the total cells in presyn_ch - (postsyn_region + presyn_regions_exclude)
channel_colours			= ['r','g','b']
network_name			= 'Unet'
grouped					= True
debug					= False

# Dataset class
#region Dataset
class Dataset:

	def __init__(self, name, group, channels, fluorescence=False, skimmed=False, starters=None, atlas_25=False, starter_pedestal_norm=0, starter_normaliser=1):
		'''
		Initialise a dataset object.
		'''
		self.name, self.group, self.channels, self.fluorescence, self.skimmed, self.starter_pedestal_norm, self.starter_normaliser = name, group, channels, fluorescence, skimmed, starter_pedestal_norm, starter_normaliser
		self.true_postsynaptics = starters
		global postsyn_region
		datasets.append(self)

		def preprocess_coords(ch):
			if self.fluorescence:
				name = f'binary_registered_skimmed_{self.name}_{ch}.npy' if self.skimmed else f'binary_registered_{self.name}_{ch}.npy'
			else:
				name = f'cells_{self.name}_{network_name}_{ch}.csv'

			cell_coords = btf.open_file(name, atlas_25=atlas_25) #TODO: check if atlas 25 works with anterograde pipeline
			cells_by_area_raw = self.__count_cells(cell_coords) # used in generate_zoom_plot()
			cells_by_area = self.__propagate_cells_through_inheritance_tree(cells_by_area_raw) # used in many plots
			return cell_coords, cells_by_area, cells_by_area_raw
		
		self.cell_coords, self.cells_by_area, self.cells_by_area_raw = {}, {}, {} # index dicts by channel name
		for channel in self.channels:
			self.cell_coords[channel], self.cells_by_area[channel], self.cells_by_area_raw[channel] = preprocess_coords(channel)

		if debug:
			try:
				for channel in self.channels:
					_, _, starter_cells = get_area_info([postsyn_region], self, self.cells_by_area[channel])
					print(f'{self.name} ({self.group}): {starter_cells[0]} channel {channel} cells in {self.name} {postsyn_region}, out of {self.num_cells(channel)} total channel {channel} cells.')
			except Exception:
				print('Starter region not known for cell count assessment. Use shorthand region notation, e.g. IO for inferior olivary complex.')
	# end of init

	def __count_cells(self, cell_coords):
		'''
		returns the number of cells in each brain region
		'''
		x_vals, y_vals, z_vals = cell_coords[0], cell_coords[1], cell_coords[2]
		counter = Counter()
		for idx, z in enumerate(z_vals):
			x = x_vals[idx]
			y = y_vals[idx]
			area_index = _get_area_index(z, y, x)
			counter.setdefault(area_index, 0)
			counter[area_index] = counter[area_index] + 1
		if debug:
			total_cells = sum(counter.values()) # <-- if total cells needs to be verified
			print(f'Cells in channel (before manipulation): {total_cells}')
		counter = counter.most_common()
		return counter

	def __propagate_cells_through_inheritance_tree(self, original_counter):
		'''
		quantifies numbers of cells for parent brain areas
		'''
		new_counter = Counter()
		for idx, count in original_counter:
			try:
				id_path = area_indexes.loc[idx].structure_id_path
			except KeyError:
				pass #print(f'Index {idx} does not exist.')
			else:
				id_path_list = id_path.split('/')
				id_path_list = [i for i in id_path_list if i != ''] # remove current indexes
				for i in id_path_list: # propagate lowest area count through all parent areas
					area_index = int(i)
					new_counter.setdefault(area_index, 0)
					new_counter[area_index] += count
		return new_counter

	def _set_channels(self, ch): # helper for functions that take channel=None argument
		channels = self.channels if ch == None else ch
		if isinstance(channels, str): # convert single channel name to list
			assert channels in self.channels, f'Specified channel name does not exist in this dataset ({self.name}).'
			channels = [channels]
		assert isinstance(channels, list), 'Failed to recognise channels. Please provide a list, e.g. [1, 2, 3], or select all channels with None.'
		return channels

	def num_cells(self, channel=None):
		'''
		Return total number of cells in a given channel.
		:channel: str: channel name. If None, returns sum of all channels
		'''
		if channel == None:
			return sum([len(self.cell_coords[channel][0]) for channel in self.channels])
		else:
			if self.fluorescence != True: # if dataset has corrected starters then account for in cell total
				return len(self.cell_coords[channel][0]) - self.num_cells_in(postsyn_region, postsyn_ch) + self.postsynaptics()
			else:
				return len(self.cell_coords[channel][0])

	def num_cells_in(self, area, channel=None, left=None, include_children=False):
		'''
		Gets the number of cells in a given brain area.
		WARNING: Used by internal functions before propagation; use only to query raw data
		'''
		channels = self._set_channels(channel)
		area_idx = get_area_info([area])[1] if area != 0 else 0 # if we are checking counts outside of brain, don't need to fetch index (0 is not available in hierarchy)
		if include_children:
			parent, children = children_from(area_idx, depth=0)
			area_idx = [parent] + children
		return sum([len(_get_cells_in(area_idx, self, channel=ch, left=left)[0]) for ch in channels])

	def show_coronal_section(self, channels=None, section=750, cells_pm=0):
		'''
		Show a coronal section of the atlas with dataset points plotted on top.
		'''
		channels = self._set_channels(channels)

		plt.suptitle(f'{self.name} Slice {str(section)} Caudal View')
		plt.imshow(atlas[section,:,:], norm=colors.LogNorm())

		for i, ch in enumerate(channels):
			cells_z_coords = np.array(self.cell_coords[ch][2])
			cells_idxs = np.array(np.where((cells_z_coords>=section-cells_pm)*(cells_z_coords<=section+cells_pm)))[0]
			for ci in cells_idxs: # plot x and y values of cells at these indexes
				plt.scatter(self.cell_coords[ch][0][ci], self.cell_coords[ch][1][ci], c=channel_colours[i], s=0.8)


	def presynaptics(self): # Presynaptic cells are the total cells inside atlas in presyn_ch - (postsyn_region + presyn_regions_exclude)
		presyn_cells = self.num_cells(presyn_ch) - self.num_cells_in(0) - self.postsynaptics()
		for region in presyn_regions_exclude:
			presyn_cells = presyn_cells - self.num_cells_in(region, presyn_ch, include_children=True)
		return presyn_cells

	def postsynaptics(self):
		if self.true_postsynaptics is not None:
			return self.true_postsynaptics
		return self.num_cells_in(postsyn_region, postsyn_ch, include_children=True)

	def project_slices(self, region, figsize=(10,6)):
		start, end = region
		subtracted_stack_files = btf.open_transformed_brain(self)

		stack = []
		for i in tqdm(range(start, end)):
			im = np.load(subtracted_stack_files[i])
			stack.append(im)
		stack = np.array(stack)
		proj = np.sum(stack, axis=0)
		binary_proj = np.where(proj > 0, 1, 0)
		print(proj.shape, proj.dtype)
		
		f, ax = plt.subplots(figsize=figsize)
		ax.imshow(binary_proj)
		plt.imsave(f'olive_proj_{self.name}.jpeg', binary_proj, cmap=cm.gray)

	def show_slice_sequence(self, region, figsize=(10,6), save=False):
		def bin_array(data, axis, binstep, binsize, func=np.nanmean):
			data = np.array(data)
			dims = np.array(data.shape)
			argdims = np.arange(data.ndim)
			argdims[0], argdims[axis]= argdims[axis], argdims[0]
			data = data.transpose(argdims)
			data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
			data = np.array(data).transpose(argdims)
			return data
		
		start, end = region
		subtracted_stack_files = btf.open_transformed_brain(self)
		seq_id = uid.uuid()
		
		for i in tqdm(range(start, end)):
			im = np.load(subtracted_stack_files[i])
			im = im.astype(int)
			res = bin_array(im, 0, 25, 25, func=np.sum)
			res = bin_array(res, 1, 25, 25, func=np.sum)

			clear_output(wait=True)
			plt.show()
			plt.figure(figsize=figsize)
			plt.axis('off')
			plt.imshow(res, cmap='binary', vmin=0, vmax=625)
			if save:
				btf.save(f'video_{self.name}_frame_{i}', as_type='png', dpi=80, vID=seq_id)

	def get_cells_in_hemisphere(self, area): #, left=True):
		# to query raw data:
		num_total = self.num_cells_in(area, ch1=True, left=None)
		num_left = self.num_cells_in(area, ch1=True, left=True)
		num_right = self.num_cells_in(area, ch1=True, left=False)

		# to get real answer:
		parent, children = children_from(area, depth=0)
		areas = [parent] + children
		true_num_total = len(_get_cells_in(areas, self, ch1=True, left=None)[0])
		true_num_left = len(_get_cells_in(areas, self, ch1=True, left=True)[0])
		true_num_right = len(_get_cells_in(areas, self, ch1=True, left=False)[0])

		return (num_total, num_left, num_right), (true_num_total, true_num_left, true_num_right)

	def assess_performance(self, gt_name, xy_tol=10, z_tol=10):
		'''
		Legacy function: Compare cell coordinates in a channel (1 at the moment) to a file containing ground truth cells.
		Generate ground truth coordinates in atlas space - downsampled_channel_0 is channel 1, downsampled_standard
		'''
		gt_cells = btf.open_file(gt_name)[0]
		gt_cells[0] = list(map(lambda x: atlas.shape[2]-x, gt_cells[0])) # flip cells x coord along the midline
		matching_gt_idxs = []
		matching_cf_idxs = []
		for gt_idx, Z in enumerate(gt_cells[2]):
			X = gt_cells[0][gt_idx]
			Y = gt_cells[1][gt_idx]
			for cf_idx, z in enumerate(self.ch1_cells[2]):
				x = self.ch1_cells[0][cf_idx]
				y = self.ch1_cells[1][cf_idx]
				if (Z-z_tol <= z <= Z+z_tol) & (X-xy_tol <= x <= X+xy_tol) & (Y-xy_tol <= y <= Y+xy_tol):
					matching_gt_idxs.append(gt_idx)
					matching_cf_idxs.append(cf_idx)
					break
		total_gt_cells = len(gt_cells[2])
		gt_matched_cells = len(matching_gt_idxs)
		cf_pos_cells = len(self.ch1_cells[2])

		prop_detected_by_cf = (gt_matched_cells / total_gt_cells) * 100
		false_positives = cf_pos_cells - gt_matched_cells
		prop_true = (gt_matched_cells / (false_positives + gt_matched_cells)) * 100
		print(f'{gt_matched_cells} matches found ({prop_detected_by_cf:.2f}% of ground truth cells)')
		print(f'{false_positives} false positives ({prop_true:.2f}% true positive rate)')

		fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
		fig.suptitle(f'{self.name} {gt_name}')
		def plot_dist(ax, axis, xlabel='Distance / um', legend=False):
			pos_cf, gt_ar = self.ch1_cells[axis], gt_cells[axis]
			y, x, _ = ax.hist([pos_cf, gt_ar], label=('Positives','Ground Truth'), bins=50)
			if legend:
				ax.legend()
				ax.text(0.02, 0.76, f'{prop_detected_by_cf:.2f}% ground truth matched ({gt_matched_cells})', ha='left', transform=ax.transAxes, color='g', fontsize=8)
				ax.text(0.02, 0.72, f'{100-prop_true:.2f}% false positives ({false_positives})', ha='left', transform=ax.transAxes, color='r', fontsize=8)
				ax.text(0.02, 0.62, f'Ztol={z_tol}um, XYtol={xy_tol}um', ha='left', transform=ax.transAxes, color='k', fontsize=8)
			ax.set_xlabel(xlabel)
			ax.set_ylabel('Cell count')
			ax.set_xlim(0, atlas.shape[axis])
		plot_dist(ax1, 2, xlabel='Distance from caudal end / um')
		plot_dist(ax2, 0, xlabel='Distance from image left / um', legend=True)
		plot_dist(ax3, 1, xlabel='Distance from image top / um')

#endregion


def validate_dimensions(dataset, atlas_25, display=False):
	'''
	Legacy function: Validate image dimensions.
	'''
	def check_dataset_size(datasets):
		for first_dataset in datasets:
			for second_dataset in datasets:
				if len(first_dataset) != len(second_dataset):
					if debug:
						print('Datasets are not the same length in z-axis.')
					return False
				if first_dataset[0].shape != second_dataset[0].shape:
					if debug:
						print('Datasets are not the same shape in xy plane.')
					return False
		return True
	def set_data_dims(datasets, squeeze=False):
		im_sets = datasets
		if squeeze:
			return list(map(lambda x: np.squeeze(x), im_sets))
		return im_sets

	if atlas_25:
		print('Warning: Dataset channel 1 is not in the same coordinate space as the 10um reference atlas. Cells being scaled up, but skipping dimension validation.')
	atlas_scaled = atlas * 2.5 if atlas_25 else atlas
	dataset_scaled = dataset.transform * 2.5 if atlas_25 else dataset.transform
	im_sets = set_data_dims([atlas_scaled, dataset_scaled])
	
	if not atlas_25: ## TODO: make check work for 25um atlas
		try:
			assert check_dataset_size(im_sets), 'Failed dimension validation.'
		except AssertionError as e:
			if debug:
				print(e, 'Attempting np.squeeze...')
			im_sets = set_data_dims([atlas_scaled, dataset_scaled], squeeze=True)
			assert check_dataset_size(im_sets), 'Failed dimension validation.'

	if display:
		print(f'Stack resolutions: {list(map(lambda x: x.shape, im_sets))}')
		fig, axes = plt.subplots(1, len(im_sets), figsize=(8,5))
		for idx, ax in enumerate(axes):
			dist = int(len(im_sets[idx])*0.7)
			ax.imshow(im_sets[idx][dist], norm=colors.LogNorm())

def _get_area_index(z, y, x):
	'''
	get the index referring to the brain area in which a cell is located
	'''
	im = atlas[z]
	if x < im.shape[1] and y < im.shape[0]: # not <= because index is (shape - 1)
		area_index = int(im[y,x])
		### USE atlas.structure_from_coords
		#print(btf.atlas.structure_from_coords((z, y, x), as_acronym=True), area_index) # this works, but returns name 'IO'
	else:
		print('Warning: Point out of bounds')
		return 0
	if area_index >= 0:
		return area_index
	else:
		print('Warning: Area index is < 0')
		return 0

def _get_cells_in(region, dataset, channel, left=None):
	'''
	Returns coordinates of cells within a defined region.
	region: can be a list of area indexes, numpy array of a 3D area, or a tuple containing the coordinates bounding a cube
	If you only need the number of cells in a region, use dataset.cells_by_area[ch][area]
	'''
	def hemisphere_predicate(hemi):
		if left == None:
			return hemi == 'left' or hemi == 'right' or hemi == None # return true regardless of hemisphere
		elif left == True:
			return hemi == 'left'
		elif left == False:
			return hemi == 'right'
	def is_in_region(z, y, x):
		if isinstance(region, list):
			areas = region
			return _get_area_index(z, y, x) in areas
		elif isinstance(region, int):
			area = region
			return _get_area_index(z, y, x) == area
		elif isinstance(region, tuple):
			(x_min, x_max), (y_min, y_max), (z_min, z_max) = region
			return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max
		elif isinstance(region, np.ndarray):
			dilation = region
			return dilation[z,y,x] == 1
		else:
			print(f'Unable to identify region type {type(region)} for returning cell coordinates.')
	
	cells_x, cells_y, cells_z = [], [], []
	cell_coords = dataset.cell_coords[channel]
	cls_x, cls_y, cls_z, cls_h = cell_coords[0], cell_coords[1], cell_coords[2], cell_coords[3]
	for idx, z in enumerate(cls_z):
		x = cls_x[idx]
		y = cls_y[idx]
		if is_in_region(z,y,x) and hemisphere_predicate(cls_h[idx]):
			cells_x.append(x)
			cells_y.append(y)
			cells_z.append(z)
	return cells_x, cells_y, cells_z

def children_from(parent, depth):
	'''
	parent: choose parent area
	depth: number of steps down the inheritance tree to fetch child areas from. 0 returns all children.
	'''
	parent, _, _ = get_area_info(parent) # get full name in case short name provided
	parent = area_indexes.loc[area_indexes['name']==parent[0]].index[0]
	parents = [parent]
	if depth == 0:
		children = []
		for i in range(10):
			parents = list(map(lambda x: area_indexes.loc[area_indexes['parent_structure_id']==x].index.values.tolist(), parents))
			parents = list(chain.from_iterable(parents))
			if parents: # if the list is not empty
				children.append(parents)
		children = list(chain.from_iterable(children))
	else:
		for i in range(depth):
			parents = list(map(lambda x: area_indexes.loc[area_indexes['parent_structure_id']==x].index.values.tolist(), parents))
			parents = list(chain.from_iterable(parents))
		children = parents
	return parent, children

def get_area_info(codes, dataset=None, channels=None): # TODO: create functions where requested representation type is returned and starting type is not specified
	'''
	Returns area full-names, area indexes, and area cell count for a dataset given short-letter codes or area indexes.
	:dataset: Optionally specify the dataset to fetch cells
	:channels: If providing dataset, optionally specify which channels to fetch total for each area from
	'''
	if not isinstance(codes, (list, np.ndarray)):
		codes = [codes]
	if isinstance(codes[0], (int, np.int32, np.int64)):
		names = area_indexes.loc[codes, 'name'].tolist()
		idxes = codes
	elif isinstance(codes[0], str):
		try:
			names = list(map(lambda x: area_indexes.loc[area_indexes['acronym']==x, 'name'].item(), codes))
			idxes = list(map(lambda x: area_indexes.loc[area_indexes['acronym']==x, 'name'].index[0], codes))
		except ValueError:
			names = list(map(lambda x: area_indexes.loc[area_indexes['name']==x, 'name'].item(), codes))
			idxes = list(map(lambda x: area_indexes.loc[area_indexes['name']==x, 'name'].index[0], codes))
	else:
		print('Unknown area reference format.')
	# for each area index, get the number of cells in the given areas in the given channels
	cells = list(map(lambda x: sum([dataset.cells_by_area[ch][int(x)] for ch in dataset._set_channels(channels)]), idxes)) if dataset is not None else None
	return names, idxes, cells

def get_area_acronyms(codes):
	if not isinstance(codes, (list, np.ndarray)):
		codes = [codes]
	assert isinstance(codes[0], (int, np.int32, np.int64)), 'Please provide area code(s).'
	return area_indexes.loc[codes, 'acronym'].tolist()

def _cells_in_areas_in_datasets(areas, datasets, channels, normalisation='presynaptics', log=False):
	cells_list = []
	for dataset in datasets: # titles will thus be set by the final dataset
		data_type = 'pixel' if dataset.fluorescence else 'cell'
		_, _, cells = get_area_info(areas, dataset, channels)
		if normalisation == 'total': # normalise to total cells in selected channels
			axis_title = f'% {data_type}s'
			cells = list(map(lambda x: (x / dataset.num_cells(channel=channels)) * 100, cells))
		elif normalisation == 'presynaptics': # by default, normalise to the number of presynaptics
			axis_title = f'% presynaptic {data_type}s'
			cells = list(map(lambda x: (x / dataset.presynaptics()) * 100, cells))
		elif normalisation == 'postsynaptics': # normalise to inputs per postsynaptic cell
			axis_title = f'Inputs / postsynaptic {data_type}'
			try:
				cells = list(map(lambda x: x / dataset.postsynaptics(), cells))
			except ZeroDivisionError:
				print(f'Dividing by zero f{data_type}s. Ensure postsynaptic correction has been applied.')
		elif normalisation == 'custom_division':
			axis_title = f'{data_type} / cerebellar px (division normalised)'
			cells = list(map(lambda x: (x / dataset.starter_normaliser), cells))
		elif normalisation == 'custom_pedestal':
			axis_title = f'{data_type} / cerebellar px (pedestal normalised)'
			cells = list(map(lambda x: (x - (dataset.starter_pedestal_norm * x)) / dataset.starter_normaliser, cells))
		else:
			if debug:
				print(f'Normalisation set to {normalisation}, defaulting to {data_type} count.')
			axis_title = f'# {data_type}s'
		if log:
			cells = list(map(lambda x: np.log(x), cells))
			axis_title = f'log({axis_title})'
		cells_list.append(cells)
	return cells_list, axis_title

def _get_extra_cells(codes, original_counter):
	'''
	Check if there are extra cells assigned to the parent area rather than the children.
	'''
	names = area_indexes.loc[codes, 'name'].tolist()
	try:
		cells = list(map(lambda x: [item for item in original_counter if item[0] == x][0][-1], codes))
		if debug:
			print('There were additional cells in the parent area')
	except IndexError:
		if debug:
			print('The zoomed in parent area has no additional cells in the parent area')
		cells = list(map(lambda x: x*0, codes))
	return names, cells

def area_predicate(area, threshold, normalisation, datasets):
    dataset_cells = _cells_in_areas_in_datasets(area, datasets, 'r', normalisation=normalisation)[0]
    mean = np.mean(np.array(dataset_cells), axis=0)[0] # calculate mean number of cells in each area
    area_has_threshold = mean > threshold
    any_child_has_threshold = False
    for child in children_from(area, depth=0)[1]:
        dataset_cells = _cells_in_areas_in_datasets(child, datasets, 'r', normalisation=normalisation)[0]
        mean = np.mean(np.array(dataset_cells), axis=0)[0] # calculate mean number of cells in each area
        if mean > threshold:
            any_child_has_threshold = True
    return area_has_threshold and not any_child_has_threshold



### STATS
#region Stats
# from scipy.stats import tukey_hsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

def anova(channel, fluorescence, areas, parametric, norm=None):
	df = get_stats_df(channel, fluorescence, areas, normalisation=norm)
	print(df.head())
	df['Cells'] = stats.rankdata(df['Cells'])
	print(df.head())
	if parametric:
		model = ols('Cells ~ C(Area) * C(Dataset)', data=df).fit()
	else:
		model = ols('Cells ~ C(Area) + C(Dataset)', data=df).fit()
	return sm.stats.anova_lm(model, typ=2)

def mwu(channel, fluorescence, area, norm=None):
	df = get_stats_df(channel, fluorescence, [area], normalisation=norm)
	LS_group = df.loc[df['Dataset']=='LS', 'Cells'].tolist()
	LV_group = df.loc[df['Dataset']=='LV', 'Cells'].tolist()
	return stats.mannwhitneyu(LS_group, LV_group)

def get_stats_df(channel, fluorescence, areas, normalisation=None, ):
	dsets = [i for i in datasets if i.fluorescence == fluorescence]
	dataset_cells, _ = _cells_in_areas_in_datasets(areas, dsets, channel, normalisation=normalisation)
	dataset_groups = [i.group for i in dsets]

	num_datasets = len(dsets)
	num_areas = len(areas)
	area_labels = areas * num_datasets
	groups = []
	for group in dataset_groups:
		groups = groups + [f'{group}']*num_areas
	cells = []
	for counts in dataset_cells:
		cells = cells + counts
	
	column_titles = ['Area', 'Dataset', 'Cells']
	df = pd.DataFrame(zip(area_labels, groups, cells), columns=column_titles)
	return df
#endregion