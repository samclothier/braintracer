import pickle, time, cv2
import braintracer.file_management as btf
import braintracer.plotting as btp
import matplotlib.pyplot as plt
import shortuuid as uid
import pandas as pd
import numpy as np
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from IPython.display import clear_output
from sklearn import linear_model
from collections import Counter
from skimage import morphology
from tqdm.notebook import tqdm
from matplotlib import colors
from itertools import chain
from scipy import ndimage
from matplotlib import cm

datasets = []
area_indexes = btf.get_lookup_df()
atlas = btf.get_atlas()
reference = btf.get_reference()
network_name = 'Unet'
fluorescence = False
starter_region = None
starter_ch1 = False
grouped = True
debug = False

class Dataset:
	def __init__(self, name, group, sig, bg, starters=None, atlas_25=False, fluorescence=False, ignore_autofluorescence=False, modify_starter=False, fluorescence_parameters=False):
		self.name, self.group, self.sig, self.bg, self.fluorescence = name, group, sig, bg, fluorescence
		self.atlas = None # becomes used if the atlas is modified or fluorescence analysed
		if not fluorescence:
			self.ch1_cells = btf.open_file(f'cells_{self.name}_{network_name}_{self.sig[0]}.csv', atlas_25=atlas_25)
			#self.ch2_cells = btf.open_file(f'cells_{self.name}_{network_name}_{self.sig[-1]}.csv', atlas_25=atlas_25)
			self.raw_ch1_cells_by_area = self.__count_cells(self.ch1_cells)
			#self.raw_ch2_cells_by_area = self.__count_cells(self.ch2_cells)
			self.ch1_cells_by_area = self.__propagate_cells_through_inheritance_tree(self.raw_ch1_cells_by_area)
			#self.ch2_cells_by_area = self.__propagate_cells_through_inheritance_tree(self.raw_ch2_cells_by_area)
		datasets.append(self)
		self.flr_by_area = None
		self.area_volumes = None
		self.true_postsynaptics = starters
		if fluorescence:
			try:
				self.flr_by_area = btf.open_file(f'fluorescence_{self.name}.pkl')
				print(f'Successfully opened saved fluorescence data for {self.name}')
			except (OSError, IOError) as e:
				print(f'Failed to open saved fluorescence data for {self.name}. Analysing fluorescence...')
				if fluorescence_parameters:
					self.analyse_fluorescence(correct_x_flip=True, check_after=400)
				else:
					self.analyse_fluorescence()
				if self.flr_by_area != None:
					btf.save(f'fluorescence_{self.name}', 'pkl', file=self.flr_by_area)
					print(f'Saved fluorescence data for {self.name}.')
				else:
					print(f'Fluorescence data for {self.name} not saved because fluorescence not quantified correctly.')
		global starter_region
		if modify_starter: # starter region is always the global starter region
			if starter_region is None:
				print('Starter region unknown. Define it with bt.starter_region = \'IO\'')
			with btf.HiddenPrints():
				self.atlas = btf.get_atlas() #TODO: allow custom starter region modification
			self.adapt_starter_area((452+175, 452+225), (627+90, 627+125), (1098+100, 1098+180))
		if debug:
			try:
				_, _, IO_cells1 = get_area_info([starter_region], self.ch1_cells_by_area)
				_, _, IO_cells2 = get_area_info([starter_region], self.ch2_cells_by_area)
				print(f'{self.name} ({self.group}): {IO_cells1[0]} ch1 cells in {self.name} {starter_region}, out of {self.num_cells(ch1=True)} total ch1 cells.')
				print(f'{self.name} ({self.group}): {IO_cells2[0]} ch2 cells in {self.name} {starter_region}, out of {self.num_cells(ch1=False)} total ch2 cells.')
			except Exception:
				print('Starter region not known for cell count assessment. Use shorthand region notation, e.g. IO for inferior olivary complex.')

	def show_coronal_section(self, slice_frac=(500, 1000), cells_pm=0, ch1=None):
		'''
		Show a coronal section of the atlas with dataset points plotted on top.
		:slice_frac: 2-item tuple: first value is slice in raw data; second value is total number of slice in raw data
		:ch1: boolean: None is both channels; True is channel 1; False is channel 2
		'''
		raw_slice_num, raw_len = slice_frac[0], slice_frac[1]
		frac = raw_slice_num / raw_len
		atlas_len = 1255 # atlas len is 1320 but registration cut-off is about 1250
		ds_slice_num = atlas_len - int(atlas_len*frac)
		plt.suptitle(f'{self.name} Slice {str(ds_slice_num)} Caudal View')
		plt.imshow(atlas[ds_slice_num,:,:], norm=colors.LogNorm())
		ch1_inslice = np.array(self.ch1_cells[2])
		ch2_inslice = np.array(self.ch2_cells[2])
		layer_ch1_idxs = np.array(np.where((ch1_inslice>=ds_slice_num-cells_pm)*(ch1_inslice<=ds_slice_num+cells_pm)))[0]
		layer_ch2_idxs = np.array(np.where((ch2_inslice>=ds_slice_num-cells_pm)*(ch2_inslice<=ds_slice_num+cells_pm)))[0]
		if ch1 != False:
			for i in layer_ch1_idxs:
				plt.scatter(self.ch1_cells[0][i], self.ch1_cells[1][i], c='r', s=0.8)
		if ch1 != True:
			for i in layer_ch2_idxs:
				plt.scatter(self.ch2_cells[0][i], self.ch2_cells[1][i], c='g', s=0.8)

	def num_cells(self, ch1=None):
		if ch1 == True:
			return len(self.ch1_cells[0])
		elif ch1 == False:
			return len(self.ch2_cells[0])
		else:
			return len(self.ch1_cells[0]) + len(self.ch2_cells[0])


	def __count_cells(self, cell_coords):
		'''
		return the number of cells in each brain region
		'''
		x_vals, y_vals, z_vals = cell_coords[0], cell_coords[1], cell_coords[2]
		counter = Counter()
		for idx, z in enumerate(z_vals):
			x = x_vals[idx]
			y = y_vals[idx]
			area_index = _get_area_index(self, z, x, y)
			counter.setdefault(area_index, 0)
			counter[area_index] = counter[area_index] + 1
		if debug:
			total_cells = sum(counter.values()) # <-- if total cells needs to be verified
			print(f'Cells in channel (before manipulation): {total_cells}')
		counter = counter.most_common()
		return counter

	def __propagate_cells_through_inheritance_tree(self, original_counter):
		'''
		quantify numbers of cells for parent brain areas
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

	def num_cells_in(self, area, ch1=None):
		'''
		get the number of cells in a given brain area
		WARNING: Used by internal functions before propagation; use only to query raw data
		'''
		area_idx = get_area_info([area])[1]
		if ch1 == None:
			return len(_get_cells_in(area_idx, self, ch1=True)[0]) + len(_get_cells_in(area_idx, self, ch1=False)[0])
		else:
			return len(_get_cells_in(area_idx, self, ch1=ch1)[0])

	def presynaptics(self):
		red_cells = self.num_cells(ch1=True)
		IO_red = self.num_cells_in('IO', ch1=True)
		CB_red = self.num_cells_in('CBX', ch1=True)
		presynaptics = red_cells - (IO_red + CB_red)
		return presynaptics
	def postsynaptics(self):
		if self.true_postsynaptics is not None:
			return self.true_postsynaptics
		return self.num_cells_in(starter_region, ch1=starter_ch1)

	def adapt_starter_area(self, x_bounds, y_bounds, z_bounds):
		z_min, z_max = z_bounds
		x_min, x_max = x_bounds
		y_min, y_max = y_bounds
		with btf.HiddenPrints():
			self.atlas = btf.get_atlas()
		_, area_index, _ = get_area_info([starter_region])
		self.atlas[z_min:z_max,y_min:y_max,x_min:x_max] = area_index

	def assess_performance(self, gt_name, xy_tol=10, z_tol=10):
		gt_cells = btf.open_file(gt_name)[0]
		gt_cells[0] = list(map(lambda x: atlas.shape[2]-x, gt_cells[0])) # flip cells x coord along the midline
		# please generate ground truth coordinates in atlas space - downsampled_channel_0 is channel 1, downsampled_standard
		# compare channel 1 cell coordinates to ground truth cells
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

	def check_fluorescence(self, num, correct_z_flip=False, correct_x_flip=False, restrict_to_idx=None):
		self.atlas = np.array(btf.open_file(f'atlas_{self.name}_r.tiff'))[0]
		if correct_x_flip:
			self.atlas = np.flip(self.atlas, axis=2)
		if correct_z_flip:
			self.atlas = np.flip(self.atlas, axis=0)
		print(atlas.shape)
		
		subtracted_stack_files = btf.open_transformed_brain(self)
		num_slices = len(subtracted_stack_files)
		print(num_slices)
		im = np.load(subtracted_stack_files[num])
		print(im.shape)

		# get values to scale pixels into registered atlas space
		z_scalar = self.atlas.shape[0] / num_slices
		y_scalar = self.atlas.shape[1] / im.shape[0]
		x_scalar = self.atlas.shape[2] / im.shape[1]
		print(z_scalar,y_scalar,x_scalar)

		z_coords, y_coords, x_coords = np.array([]), np.array([]), np.array([])
		atlas_z = int((num_slices - num) * z_scalar) - 1 # pick the closest atlas slice, -1 to start from zero
		print(atlas_z)
		y_pxs, x_pxs = np.nonzero(im) # get coordinates of 1s in this slice
		x_coords = np.append(x_coords, list(map(lambda x: int(x * x_scalar), x_pxs)))
		y_coords = np.append(y_coords, list(map(lambda y: int(y * y_scalar), y_pxs)))
		z_coords = np.append(z_coords, [atlas_z] * len(x_pxs)) # add all the repeating z coordinates for this slice
		coords = [list(x_coords.astype(int)), list(y_coords.astype(int)), list(z_coords.astype(int))]

		x_vals, y_vals, z_vals = coords[0], coords[1], coords[2]
		x_IO, y_IO, z_IO = np.array([]), np.array([]), np.array([])
		seen_vals = []
		for idx, z in enumerate(z_vals):
			x = x_vals[idx]
			y = y_vals[idx]
			area_index = _get_area_index(self, z, x, y)
			if area_index == restrict_to_idx:
				x_IO = np.append(x_IO, x)
				y_IO = np.append(y_IO, y)

		atlas_im = self.atlas[atlas_z]
		f, ax = plt.subplots(figsize=(10, 10))
		ax.imshow(atlas_im, cmap='Greens', vmax=1000)
		if restrict_to_idx is None:
			ax.scatter(coords[0], coords[1], s=0.2)
		else:
			ax.scatter(x_IO, y_IO, s=0.2)

	def analyse_fluorescence(self, correct_z_flip=False, correct_x_flip=False, check_after=200):
		self.atlas = np.array(btf.open_file(f'atlas_{self.name}_r.tiff'))[0]
		if correct_x_flip:
			self.atlas = np.flip(self.atlas, axis=2)
		if correct_z_flip:
			self.atlas = np.flip(self.atlas, axis=0)
		print(self.atlas.shape)
		
		subtracted_stack_files = btf.open_transformed_brain(self)
		num_slices = len(subtracted_stack_files)
		print(num_slices)
		first_im = np.load(subtracted_stack_files[0])
		print(first_im.shape)

		# get values to scale pixels into registered atlas space
		z_scalar = self.atlas.shape[0] / num_slices
		y_scalar = self.atlas.shape[1] / first_im.shape[0]
		x_scalar = self.atlas.shape[2] / first_im.shape[1]
		print(z_scalar,y_scalar,x_scalar)

		z_coords, y_coords, x_coords = np.array([]), np.array([]), np.array([])
		for z, im in enumerate(tqdm(subtracted_stack_files)):
			# take reverse of z
			atlas_z = int((num_slices - z) * z_scalar) - 1 # pick the closest atlas slice, -1 to start from zero
			y_pxs, x_pxs = np.nonzero(np.load(im)) # get coordinates of 1s in this slice
			x_atlas = list(map(lambda x: int(x * x_scalar), x_pxs))
			y_atlas = list(map(lambda y: int(y * y_scalar), y_pxs))
			x_coords = np.append(x_coords, x_atlas)
			y_coords = np.append(y_coords, y_atlas)
			z_coords = np.append(z_coords, [atlas_z] * len(x_pxs)) # add all the repeating z coordinates for this slice
			if check_after == z:
				atlas_im = np.where(self.atlas[atlas_z] > 0, 1, 0)
				f, ax = plt.subplots(figsize=(10, 10))
				ax.imshow(atlas_im, cmap='Greens', interpolation=None)
				ax.scatter(x_atlas, y_atlas, s=1)
		coords = [list(x_coords.astype(int)), list(y_coords.astype(int)), list(z_coords.astype(int))]
		
		raw_fluorescence_by_area = self.__count_cells(coords)
		fluorescence_by_area = self.__propagate_cells_through_inheritance_tree(raw_fluorescence_by_area)
		self.flr_by_area = fluorescence_by_area

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

	def get_tot_fluorescence(self, area_idxs):
		if any(n < 0 for n in self.flr_by_area):
			print(f'Warning: Brain regions have {sum(n < 0 for n in self.flr_by_area)} negative fluorescence values.')
		_, area_idxs, _ = get_area_info(area_idxs) # make sure string area names are indexes

		return [self.flr_by_area[int(i)] for i in area_idxs]
		# TODO: get area volumes with API


class _Results: # singleton object
	_instance = None
	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(_Results, cls).__new__(cls)
			# Put any initialization here.
			cls.summary_cells = []
			cls.nrmdltn_cells = []
		return cls._instance
results = _Results()

def validate_dimensions(dataset, atlas_25, display=False):
	'''
	validate image dimensions
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
	

def _raw_to_downsampled(raw_dim, downsampled_dim, cell_coords):
	'''
	downsample points from raw coordinates
	'''
	x_vals, y_vals, z_vals = cell_coords[0], cell_coords[1], cell_coords[2]
	def x_to_downsampled(x):
		return downsampled_dim[1] - int(x * (downsampled_dim[1] / raw_dim[0])) # x inverted, results in float
	def y_to_downsampled(y):
		return int(y * (downsampled_dim[0] / raw_dim[1])) # also results in float
	def z_to_downsampled(z):
		return downsampled_dim[2] - int(z * (downsampled_dim[2] / raw_dim[2])) # z inverted
	x_vals = list(map(x_to_downsampled, x_vals))
	y_vals = list(map(y_to_downsampled, y_vals))
	z_vals = list(map(z_to_downsampled, z_vals))
	return x_vals, y_vals, z_vals

def _get_area_index(dataset, z, x, y):
	'''
	get the index referring to the brain area in which a cell is located
	'''
	im = atlas[z] if dataset.atlas is None else dataset.atlas[z] # happens when a dataset's atlas has been modified
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

def _get_cells_in(region, dataset, ch1=True):
	'''
	return coordinates of cells within a defined region.
	region: can be a list of area indexes, numpy array of a 3D area, or a tuple containing the coordinates bounding a cube
	'''
	def is_in_region(z, x, y):
		if isinstance(region, list):
			areas = region
			return _get_area_index(dataset, z, x, y) in areas
		elif isinstance(region, int):
			area = region
			return _get_area_index(dataset, z, x, y) == area
		elif isinstance(region, tuple):
			(x_min, x_max), (y_min, y_max), (z_min, z_max) = region
			return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max
		elif isinstance(region, np.ndarray):
			dilation = region
			return dilation[z,y,x] == 1
		else:
			print(f'Unable to identify region type {type(region)} for returning cell coordinates.')
	cells_x, cells_y, cells_z = [], [], []
	cell_coords = dataset.ch1_cells if ch1 else dataset.ch2_cells
	cls_x, cls_y, cls_z = cell_coords[0], cell_coords[1], cell_coords[2]
	for idx, z in enumerate(cls_z):
		x = cls_x[idx]
		y = cls_y[idx]
		if is_in_region(z,x,y):
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

def get_area_info(codes, new_counter=None):
	'''
	return area full-names, area indexes, and area cell count given short-letter codes or area indexes
	'''
	if not isinstance(codes, (list, np.ndarray)):
		codes = [codes]
	if isinstance(codes[0], (int, np.int32, np.int64)):
		names = area_indexes.loc[codes, 'name'].tolist()
		idxes = codes
		if new_counter is not None:
			cells = list(map(lambda x: new_counter[int(x)], idxes))
	elif isinstance(codes[0], str):
		try:
			names = list(map(lambda x: area_indexes.loc[area_indexes['acronym']==x, 'name'].item(), codes))
			idxes = list(map(lambda x: area_indexes.loc[area_indexes['acronym']==x, 'name'].index[0], codes))
		except ValueError:
			names = list(map(lambda x: area_indexes.loc[area_indexes['name']==x, 'name'].item(), codes))
			idxes = list(map(lambda x: area_indexes.loc[area_indexes['name']==x, 'name'].index[0], codes))
		if new_counter is not None:
			cells = list(map(lambda x: new_counter[int(x)], idxes))
	else:
		'Unknown area reference format.'
	if new_counter == None:
		cells = None
	return names, idxes, cells

def _get_extra_cells(codes, original_counter):
	'''
	check if there are extra cells assigned to the parent area rather than the children
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

def _project(ax, dataset, area, padding, ch1, s, contour, colours=['r','g'], axis=0, dilate=False, all_cells=False):
	'''
	plot a coronal or horizontal projection of a brain region with cells superimposed
	'''
	projection, (x_min, y_min, z_min), (x_max, y_max, z_max) = get_projection(area, padding, dataset=dataset, dilate=dilate, all_cells=all_cells, axis=axis)
	
	if dataset.atlas is None: # don't show modified areas
		if contour:
			ax.contour(projection, colors='k')
			ax.set_aspect('equal')
		else:
			ax.imshow(projection)

	def show_cells(channel, colour):
		if all_cells:
			region = (x_min, x_max), (y_min, y_max), (z_min, z_max)
		elif dilate:
			region = atlas_ar
		else:
			parent, children = children_from(area, depth=0)
			areas = [parent] + children
			region = areas
		X_r, Y_r, Z_r = _get_cells_in(region, dataset, ch1=channel)
		X_r = [x-x_min for x in X_r]
		Y_r = [y-y_min for y in Y_r]
		Z_r = [z-z_min for z in Z_r]
		channel_label = 'Channel 1' if channel else 'Channel 2'
		if axis == 0:
			ax.scatter(X_r, Y_r, color=colour, s=s, label=channel_label, zorder=10)
		elif axis == 1:
			ax.scatter(X_r, Z_r, color=colour, s=s, label=channel_label, zorder=10)
		else:
			pass
	if ch1 == None:
		show_cells(True, colours[0])
		show_cells(False, colours[1])
	elif ch1 == True:
		show_cells(True, colours[0])
	else:
		show_cells(False, colours[1])
	return x_min, y_min, z_min

def get_projection(area, padding, dataset=None, dilate=False, all_cells=False, axis=0):
	parent, children = children_from(area, depth=0)
	areas = [parent] + children
	
	if dataset is not None:
		atlas_to_project = atlas if dataset.atlas is None else dataset.atlas # happens when a dataset's atlas has been modified
	else:
		atlas_to_project = atlas
	if all_cells:
		atlas_to_project = atlas
	atlas_ar = np.isin(atlas_to_project, areas)
	if dilate:
		struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
		atlas_ar = ndimage.binary_dilation(atlas_ar, struct, iterations=10)
	nz = np.nonzero(atlas_ar)
	z_min = nz[0].min() - padding
	y_min = nz[1].min() - padding
	x_min = nz[2].min() - padding
	z_max = nz[0].max() + padding+1
	y_max = nz[1].max() + padding+1
	x_max = nz[2].max() + padding+1
	if debug:
		print('x:'+str(x_min)+' '+str(x_max)+' y:'+str(y_min)+' '+str(y_max)+' z:'+str(z_min)+' '+str(z_max))
	perimeter = atlas_ar[z_min : z_max,
						 y_min : y_max,
						 x_min : x_max]
	projection = perimeter.any(axis=axis)
	projected_area = projection.astype(int)
	return projected_area, (x_min, y_min, z_min), (x_max, y_max, z_max)



### STATS
# from scipy.stats import tukey_hsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
def anova(areas, norm='postsynaptics'):
	df = get_stats_df(areas, normalisation=norm)
	print(df.head())
	df['Cells'] = stats.rankdata(df['Cells'])
	print(df.head())
	model = ols('Cells ~ C(Area) + C(Dataset)', data=df).fit()
	return sm.stats.anova_lm(model, typ=2)
def mwu(area, norm='postsynaptics'):
	df = get_stats_df([area], normalisation=norm)
	LS_group = df.loc[df['Dataset']=='LS', 'Cells'].tolist()
	LV_group = df.loc[df['Dataset']=='LV', 'Cells'].tolist()
	return stats.mannwhitneyu(LS_group, LV_group)

def get_stats_df(areas, normalisation='postsynaptics'):
	if not fluorescence:
		dataset_cells, _ = btp._cells_in_areas_in_datasets(areas, datasets, normalisation=normalisation)
	else:
		dataset_cells, _ = btp._fluorescence_by_area_across_fl_datasets(areas, datasets, normalisation=normalisation)
	dataset_groups = [i.group for i in datasets]

	num_datasets = len(datasets)
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