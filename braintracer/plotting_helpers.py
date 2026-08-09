"""
Copyright (C) 2021-2025  Sam Clothier

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

import numpy as np
import matplotlib.colors as clrs
import matplotlib.pyplot as plt
from fastcluster import linkage
import seaborn as sns
import pandas as pd
from itertools import chain
from scipy import signal

import importlib
bt_path = '.'.join(__name__.split('.')[:-1]) # get module path (folder containing this file)
bt = importlib.import_module(bt_path+'.analysis')


# colour helpers
class RGB(np.ndarray):
	@classmethod
	def from_str(cls, rgbstr):
		return np.array([
			int(rgbstr[i:i+2], 16)
			for i in range(1, len(rgbstr), 2)
		]).view(cls)
	def __str__(self):
		self = self.astype(np.uint8)
		return '#' + ''.join(format(n, 'x') for n in self)

def get_angle_for_cmap(g1, g2): # simpler version of the grad_to_rgb function for determining probability_map_overlap cbar
	angle = np.arctan2(g1, g2)
	angle = angle % (2 * np.pi)
	if angle < 0:
		angle += 2 * np.pi
	angle = angle * 1.5 + np.pi
	rgb_space = clrs.hsv_to_rgb((angle / 2 / np.pi, 1, 1))
	return rgb_space

def set_group_colours(group1='#ED008C', group2='#1E74BD', arbitrary='#08A045'):
	try: # when this function is called before datasets are imported, group names are not yet known
		group1_name, group2_name = get_bt_groups()[0], get_bt_groups()[1]
	except:
		group1_name, group2_name = 'Group 1', 'Group 2'

	mix_name = f'{group1_name} to {group2_name}'
	cmap_arbtry = clrs.LinearSegmentedColormap.from_list('Arbitrary cmap', ['#FFFFFF', arbitrary])
	cmap_group1 = clrs.LinearSegmentedColormap.from_list(group1_name, ['#FFFFFF', group1])
	cmap_group2 = clrs.LinearSegmentedColormap.from_list(group2_name, ['#FFFFFF', group2])
	group_midpoint = (RGB.from_str(group1) + RGB.from_str(group2)) / 2
	
	global cmap_group1_to_group2, cmap_midpoint_of_both_groups, cmap_groups_mixed_for_density_maps, csolid_group, cmaps_group
	cmap_group1_to_group2 = clrs.LinearSegmentedColormap.from_list(mix_name, [group1, '#FFFFFF', group2])
	cmap_midpoint_of_both_groups = clrs.LinearSegmentedColormap.from_list('Global', ['#FFFFFF', str(group_midpoint)])
	default_cmap = group2 == '#1E74BD' and group1 == '#ED008C'
	cmap_groups_mixed_for_density_maps = clrs.LinearSegmentedColormap.from_list(mix_name, [get_angle_for_cmap(0, 1), get_angle_for_cmap(1, 1), get_angle_for_cmap(1, 0)])
	csolid_group = [group1, group2]
	cmaps_group = [cmap_arbtry, cmap_group1, cmap_group2] # the first of these will be used for plots with only one group


# plotting helpers
def draw_plot(ax, datasets, areas, values, axis_title, fig_title, horizontal=False, l_space=None, b_space=None):
	if ax is None:
		f, ax = plt.subplots(figsize=(8,6))
		f.subplots_adjust(left=l_space, bottom=b_space)
		f.set_facecolor('white')
	if bt.grouped:
		groups = [i.group for i in datasets]
		df = plot_grouped_points(ax, values, groups, areas, axis_title, is_horizontal=horizontal)
	else:
		dataset_names = [i.name for i in datasets]
		a_names, d_names, values = prep_for_sns(areas, dataset_names, values)
		column_titles = ['Area', 'Dataset', axis_title]
		df = pd.DataFrame(zip(a_names, d_names, values), columns=column_titles)
		x = column_titles[2] if horizontal else column_titles[0]
		y = column_titles[0] if horizontal else column_titles[2]
		sns.barplot(x=x, y=y, hue=column_titles[1], data=df, ax=ax)
	if not horizontal:
		ax.set(xlabel=None)
		ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
	grid_orientation = 'x' if horizontal else 'y'
	ax.grid(axis=grid_orientation)
	ax.set_title(fig_title)
	return df

# other helpers
def map_for_data(ax1_data, ax2_data, lower_lim, saturation_multiplier, colour_set="magenta-cyan"):
	if lower_lim is not None:
		ax1_data[ax1_data < lower_lim] = 0
		ax2_data[ax2_data < lower_lim] = 0
	gradabs = np.add(ax1_data, ax2_data) / 2 # average to get labelling magnitude
	
	# enable saturation scaling
	max_abs = np.max(gradabs)
	print(f'Upper limit: {max_abs}')
	gradabs = gradabs * saturation_multiplier
	gradabs[gradabs > max_abs] = max_abs

	# calculate angle component
	gradang = np.arctan2(ax1_data, ax2_data)

	def grad_to_rgb(angle, absolute):
		"""Get the rgb value for the given `angle` and the `absolute` value
		angle : float - The angle in radians
		absolute : float - The absolute value of the gradient
		-------
		array_like
			The rgb value as a tuple with values [0..1]
		"""
		# normalize angle
		angle = angle % (2 * np.pi)
		if angle < 0:
			angle += 2 * np.pi

		# angle * x - controls the range on hsv colour wheel. lower is tighter. x=2 is opposites on colour wheel.
		# np.pi * x - controls the offset of the range on hsv colour wheel. [-pi,+pi]
		if colour_set == "magenta-cyan":
			angle = angle * 1.5 + np.pi
		else: # magenta-green
			angle = angle * 1.5 + np.pi
			cutoff = 1.55
			if angle > cutoff * np.pi:
				angle = 2 * np.pi - (angle - cutoff * np.pi)
			else:
				angle = cutoff * np.pi - angle
		rgb_space = clrs.hsv_to_rgb((angle / 2 / np.pi, 
									absolute / max_abs, 
									1)) #absolute / max_abs))
		return rgb_space

	grad = np.array(list(map(grad_to_rgb, gradang.flatten(), gradabs.flatten())))
	grad = grad.reshape(tuple(list(gradabs.shape) + [3]))
	
	return grad

def prep_for_sns(area_names, dataset_names, dataset_cells):
	#area_names = list(chain.from_iterable(area_names))
	num_datasets = len(dataset_names)
	num_areas = len(area_names)
	names = area_names*num_datasets
	datasets = []
	for name in dataset_names:
		datasets = datasets + [f'{name}']*num_areas
	cells = []
	for counts in dataset_cells:
		cells = cells + counts
	return names, datasets, cells

def compress_into_groups(group_names, dataset_cells):
	groups = get_bt_groups()
	group1 = []
	group2 = []
	for idx, group in enumerate(group_names): # lists of dataset cells by group
		if group == groups[0]:
			group1.append(dataset_cells[idx])
		else:
			group2.append(dataset_cells[idx])
	num_group1 = len(group1)
	num_group2 = len(group2)
	total1 = group1[0]
	total2 = group2[0]
	for idx, cells in enumerate(group1):
		if idx != 0:
			total1 = [x+y for x,y in zip(total1, cells)]
	for idx, cells in enumerate(group2):
		if idx != 0:
			total2 = [x+y for x,y in zip(total2, cells)]
	group1 = [x/num_group1 for x in total1]
	group2 = [x/num_group2 for x in total2]
	cells = [group1]
	cells.append(group2)
	datasets = groups
	return datasets, cells

def group_points(names, uncompressed_cells, groups):
	first_names = names[:len(uncompressed_cells[0])]
	area_name = []
	dataset_name = []
	dataset_cell = []
	for idx, cells in enumerate(uncompressed_cells):
		area_name.append(first_names)
		dataset_name.append([groups[idx]]*len(cells))
		dataset_cell.append(cells)
	area_name = list(chain.from_iterable(area_name))
	dataset_name = list(chain.from_iterable(dataset_name))
	dataset_cell = list(chain.from_iterable(dataset_cell))
	return area_name, dataset_name, dataset_cell

def plot_grouped_points(ax, dataset_cells, group_names, area_names, axis_title, is_horizontal):
	pre_compressed_dataset_cells = dataset_cells
	dataset_names, dataset_cells = compress_into_groups(group_names, dataset_cells)
	names, datasets, cells = prep_for_sns(area_names, dataset_names, dataset_cells)
	titles = ['Area', 'Dataset', axis_title]
	df = pd.DataFrame(zip(names, datasets, cells), columns=titles)
	area_name, dataset_name, dataset_cell = group_points(names, pre_compressed_dataset_cells, group_names)
	points_df = pd.DataFrame(zip(area_name, dataset_name, dataset_cell), columns=titles)
	if is_horizontal:
		sns.barplot(x=titles[2], y=titles[0], hue=titles[1], hue_order=dataset_names, palette=csolid_group, data=df, ax=ax)
		sns.stripplot(x=titles[2], y=titles[0], hue=titles[1], hue_order=dataset_names, palette=csolid_group, dodge=True, edgecolor='w', linewidth=0.5, data=points_df, ax=ax)
	else:
		sns.barplot(x=titles[0], y=titles[2], hue=titles[1], hue_order=dataset_names, palette=csolid_group, data=df, ax=ax)
		sns.stripplot(x=titles[0], y=titles[2], hue=titles[1], hue_order=dataset_names, palette=csolid_group, dodge=True, edgecolor='w', linewidth=0.5, data=points_df, ax=ax)
	display_legend_subset(ax, (2,3))
	return df

def display_legend_subset(ax, idx_tup):
	handles, labels = ax.get_legend_handles_labels()
	ax.legend([handle for i,handle in enumerate(handles) if i in idx_tup],
				[label for i,label in enumerate(labels) if i in idx_tup])

def get_bt_groups():
	groups = list(dict.fromkeys([dataset.group for dataset in bt.datasets])) # get unique values
	if len(groups) not in [0,2]:
		print('Warning: Comparison plots should only be generated for two dataset groups.')
	return groups

def colours_from_labels(names):
	datasets = [next((d for d in bt.datasets if d.name==i), None) for i in names] # get datasets by ordered labels
	groups = [i.group for i in datasets] # get their groups and return list of row/column label colours
	return list(map(lambda x: csolid_group[0] if x == get_bt_groups()[0] else csolid_group[1], groups)) # list of colours

def seriation(Z,N,cur_index):
	'''
		input:
			- Z is a hierarchical tree (dendrogram)
			- N is the number of points given to the clustering process
			- cur_index is the position in the tree for the recursive traversal
		output:
			- order implied by the hierarchical tree Z
			
		seriation computes the order implied by a hierarchical tree (dendrogram)
	'''
	if cur_index < N:
		return [cur_index]
	else:
		left = int(Z[cur_index-N,0])
		right = int(Z[cur_index-N,1])
		return (seriation(Z,N,left) + seriation(Z,N,right))
	
def compute_serial_matrix(dist_mat,method="ward"):
	'''
		input:
			- dist_mat is a distance matrix
			- method = ["ward","single","average","complete"]
		output:
			- seriated_dist is the input dist_mat,
			  but with re-ordered rows and columns
			  according to the seriation, i.e. the
			  order implied by the hierarchical tree
			- res_order is the order implied by
			  the hierarhical tree
			- res_linkage is the hierarhical tree (dendrogram)
		
		compute_serial_matrix transforms a distance matrix into 
		a sorted distance matrix according to the order implied 
		by the hierarchical tree (dendrogram)
	'''
	N = len(dist_mat)
	flat_dist_mat = dist_mat #squareform(dist_mat)
	res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
	res_order = seriation(res_linkage, N, N + N-2)
	seriated_dist = np.zeros((N,N))
	a,b = np.triu_indices(N,k=1)
	seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
	seriated_dist[b,a] = seriated_dist[a,b]
	
	return seriated_dist, res_order, res_linkage

def get_bins(dim, size):
	atlas_res = 10
	if dim == 2: # z 1320
		num_slices = len(bt.atlas.annotation)
	elif dim == 1: # y 800
		num_slices = bt.atlas.annotation[0].shape[0]
	elif dim == 0: # x 1140
		num_slices = bt.atlas.annotation[0].shape[1]
	bin_size = int(size / atlas_res)
	return len([i for i in range(0, num_slices + bin_size, bin_size)]) # return num bins

def fetch_groups(fluorescence):
	groups = get_bt_groups()
	dataset_selection = [i for i in bt.datasets if i.fluorescence == fluorescence]
	datasets1 = [i for i in dataset_selection if i.group == groups[0]]
	datasets2 = [i for i in dataset_selection if i.group == groups[1]]
	datasets = datasets1 + datasets2
	num_datasets_in_group1 = len(datasets1)
	return datasets, num_datasets_in_group1

def get_matrix_data(channel, area_func, postprocess_for_scatter=False, fluorescence=False, value_norm=None, sort_matrix=True):
	datasets, num_g1 = fetch_groups(fluorescence)
	assert len(datasets) > 0, 'No datasets found!'
	if postprocess_for_scatter == False:
		print('Warning: This function does not sort, even if postprocess_for_scatter=False')
	
	area_idxs, areas_title = area_func
	area_labels = bt.get_area_info(area_idxs)[0]
	dataset_cells, axis_title = bt._cells_in_areas_in_datasets(area_labels, datasets, channel, normalisation=value_norm)
	
	dataset_cells = np.array(dataset_cells)

	num_g1 = num_g1 # take the mean of the group cols for sorting
	collapsed_g1 = np.mean(dataset_cells[0:num_g1,:], axis=0)
	collapsed_g2 = np.mean(dataset_cells[num_g1:,:], axis=0)
	dataset_cells_for_sorting = np.concatenate([[collapsed_g1, collapsed_g2]], axis=1)

	if postprocess_for_scatter: # provide only the mean of each group
		dataset_cells = dataset_cells_for_sorting
	
	if not postprocess_for_scatter and sort_matrix:
		#sum_side = np.mean(dataset_cells, axis=0)

		LS_labelling, LV_labelling = dataset_cells_for_sorting[0], dataset_cells_for_sorting[1]
		perc_deviation = ((LV_labelling - LS_labelling) / (LS_labelling + LV_labelling)) * 100
		order_weighting = [LV_labelling[i] if (x >=0) else -LS_labelling[i] for i, x in enumerate(perc_deviation)] # sort primarily by selectivity direction and then by quantity of labelling

		sort_order = np.array(order_weighting).argsort()
		dataset_cells = dataset_cells[:, sort_order[::-1]]
		area_labels = [area_labels[i] for i in reversed(sort_order)]
	
	return area_labels, dataset_cells, datasets, areas_title, axis_title


# don't plot the outline, just return the data
def probability_map_data(channel, fluorescence, area_num=None, binsize=200, sigma=None, padding=0, three_dimensions=False, axis=0, exclude_subregions=None):
	atlas_res = 10
	assert binsize % atlas_res == 0, f'Binsize must be a multiple of atlas resolution ({atlas_res}um) to display correctly.'
	assert axis in [0, 1, 2], 'Must provide a valid axis number 0-2.'
	if area_num is None:
		area_num = 997

	_, min_bounds, max_bounds = get_projection(area_num, padding=padding, axis=axis)
	
	if three_dimensions:
		ax1_data = get_density_map_3D(channel, fluorescence, area_num, binsize, sigma, get_bt_groups()[0], exclude_subregions=exclude_subregions)
		ax2_data = get_density_map_3D(channel, fluorescence, area_num, binsize, sigma, get_bt_groups()[1], exclude_subregions=exclude_subregions)
	else:
		ax1_data = get_density_map(channel, area_num, axis, atlas_res, binsize, sigma, get_bt_groups()[0], min_bounds, max_bounds, fluorescence, exclude_subregions=exclude_subregions)
		ax2_data = get_density_map(channel, area_num, axis, atlas_res, binsize, sigma, get_bt_groups()[1], min_bounds, max_bounds, fluorescence, exclude_subregions=exclude_subregions)
	return ax1_data, ax2_data

def get_density_map(channel, area, axis, atlas_res, binsize, sigma, group, min_bounds, max_bounds, fluorescence, exclude_subregions=None):
	(px_min, py_min, pz_min), (px_max, py_max, pz_max) = min_bounds, max_bounds

	hist_list = []
	datasets_to_bin = [d for d in bt.datasets if d.group == group and d.fluorescence == fluorescence]
	assert len(datasets_to_bin) > 0, 'Could not find any datasets matching group and fluorescence setting.'

	for d in datasets_to_bin:
		points = d.get_points_from_area(channel=channel, area=area, exclude_subregions=exclude_subregions)

		x_bins, y_bins, z_bins = get_bins(0, binsize), get_bins(1, binsize), get_bins(2, binsize)
		hist, _ = np.histogramdd(points, bins=(x_bins, y_bins, z_bins), range=((0,1140),(0,800),(0,1320)), density=False)
		hist = get_sigma_smoothed_3D_array(hist, sigma)
		hist = convert_3D_array_to_probability_density_values(hist)
		hist = np.sum(hist, axis=axis) # take the maximum projection of the distribution

		scale = int(binsize / atlas_res) ## make ready for plotting
		hist = hist.repeat(scale, axis=0).repeat(scale, axis=1) # multiply up to the atlas resolution
		at_shp = bt.atlas.annotation.shape
		if axis == 2:
			hist = hist[hist.shape[0]-at_shp[2] :, hist.shape[1]-at_shp[1] :] # correct the misalignment created by repeating values during scale up, by removing the first values
			hist = hist[px_min : px_max, py_min : py_max] # crop the axes of the binned data that were scaled up to atlas resolution
		elif axis == 1:
			hist = hist[hist.shape[0]-at_shp[2] :, hist.shape[1]-at_shp[0] :]
			hist = hist[px_min : px_max, pz_min : pz_max]
		else:
			hist = hist[hist.shape[0]-at_shp[1] :, hist.shape[1]-at_shp[0] :]
			hist = hist[py_min : py_max, pz_min : pz_max] 

		hist_list.append(hist)
	all_hists = np.array(hist_list) # get cell distributions for each dataset, ready for plotting
	av_im = np.median(all_hists, axis=0) # get the median cell distribution across datasets
	av_im = av_im if axis == 0 else av_im.T # side-on orientation does not need axis swapping
	
	return av_im

def get_density_map_3D(channel, fluorescence, area, binsize, sigma, group, exclude_subregions=None):
	# get the density map voxels unflattened and untransformed for calculating the spatial selectivity index
	hist_list = []
	datasets_to_bin = [d for d in bt.datasets if d.group == group and d.fluorescence == fluorescence]
	assert len(datasets_to_bin) > 0, 'Could not find any datasets matching group and fluorescence setting.'
	for d in datasets_to_bin:
		points = d.get_points_from_area(channel=channel, area=area, exclude_subregions=exclude_subregions)

		x_bins, y_bins, z_bins = get_bins(0, binsize), get_bins(1, binsize), get_bins(2, binsize)
		hist, _ = np.histogramdd(points, bins=(x_bins, y_bins, z_bins), range=((0,1140),(0,800),(0,1320)), density=False)
		hist = get_sigma_smoothed_3D_array(hist, sigma)

		hist_list.append(hist)
	all_hists = np.array(hist_list) # get cell distributions for each dataset, ready for plotting
	av_im = np.median(all_hists, axis=0) # get the median cell distribution across datasets
	av_im = convert_3D_array_to_probability_density_values(av_im)
	return av_im

def replace_areas_with_combined_area(areas_to_combine, area_labels, dataset_cells=None, do_not_merge=False,
									 correlations=None, corr_channel=None, corr_fl=None, corr_gradient=None, corr_sigma=None):
	for new_area_name, old_area_codes in areas_to_combine.items():
		old_area_names = bt.get_area_info(old_area_codes)[0]
		print(old_area_names)
		indexes_to_remove = [area_labels.index(name) for name in old_area_names]

		area_labels = [ label for i, label in enumerate(area_labels) if i not in indexes_to_remove ]
		area_labels.append(new_area_name)

		if dataset_cells is not None: # sum cell counts for areas being merged. But to maintain ability to display child regions set do_not_merge=True
			combined_sum = np.array([np.sum(dataset_cells[:, indexes_to_remove], axis=1)])
			if do_not_merge:
				for i in indexes_to_remove:
					dataset_cells[:, i] = combined_sum
			else:
				dataset_cells = np.delete(dataset_cells, indexes_to_remove, axis=1)
				dataset_cells = np.concatenate((dataset_cells, combined_sum.T), axis=1)

		if correlations is not None:
			corr = get_corr_index_mult(corr_channel, corr_fl, old_area_codes, corr_gradient, corr_sigma)
			if do_not_merge:
				for i in indexes_to_remove:
					correlations[i] = corr
			else:
				correlations = [ corr for i, corr in enumerate(correlations) if i not in indexes_to_remove ]
				correlations.append(corr)

	return area_labels, dataset_cells, correlations




# matrix sorting helpers

def selectivity_index(g1_labelling, g2_labelling):
	divisor = g1_labelling + g2_labelling
	if np.sum(divisor) == 0:
		return 0
	return ((g2_labelling - g1_labelling) / divisor)

def colours_from_SI_array(SIs):
	def get_colour(x):
		return csolid_group[0] if x <= 0 else csolid_group[1]
	return np.array([get_colour(xi) for xi in SIs])

def calculate_SI_from_means(g1_matrix, g2_matrix):
	group1_mean_cells_by_area = np.mean(g1_matrix, axis=0)
	group2_mean_cells_by_area = np.mean(g2_matrix, axis=0)
	
	avgs = selectivity_index(group1_mean_cells_by_area, group2_mean_cells_by_area)
	colours = colours_from_SI_array(avgs)
	
	return avgs, colours

def calculate_SI_with_errors(g1_matrix, g2_matrix):
	SI_combos_array = []
	for i in range(0, g1_matrix.shape[1]):
		group1_area_vals, group2_area_vals = g1_matrix[:,i], g2_matrix[:,i]
		SI_combinations = []
		for group1_val in group1_area_vals:
			for group2_val in group2_area_vals:
				SI_combinations.append(selectivity_index(group1_val, group2_val))
		SI_combos_array.append(SI_combinations)
		
	avgs = [np.mean(i) for i in SI_combos_array]
	stes = [bt.ste(i) for i in SI_combos_array] # convert standard deviation to error
	colours = colours_from_SI_array(avgs)
	
	return avgs, stes, colours

def get_sorting_from_SI(dataset_cells, fluorescence, mean=False):
	_, num_g1 = fetch_groups(fluorescence) # get sum across each group
	group1_cells_by_area, group2_cells_by_area = dataset_cells[0:num_g1,:], dataset_cells[num_g1:,:]
	
	if mean:
		avgs, _ = calculate_SI_from_means(group1_cells_by_area, group2_cells_by_area)
	else:
		avgs, _, _ = calculate_SI_with_errors(group1_cells_by_area, group2_cells_by_area)
	
	# sort primarily by selectivity direction and then by quantity of labelling
	group1_mean_cells_by_area = np.mean(group1_cells_by_area, axis=0)
	group2_mean_cells_by_area = np.mean(group2_cells_by_area, axis=0)
	order_weighting = [group2_mean_cells_by_area[i] if (x >=0) else -group1_mean_cells_by_area[i] for i, x in enumerate(avgs)]
	sort_order = np.array(order_weighting).argsort()
	return sort_order

def point_in_boxes(LS_point, LV_point, gradient):
	return (LV_point <= LS_point * gradient) or (LS_point <= LV_point * gradient)

def get_corr_index(LS_data, LV_data, gradient):
	bool_mask = [point_in_boxes(s, v, gradient) for s, v in zip(LS_data.flatten(), LV_data.flatten())]
	sum_in_boxes = bool_mask.count(True)
	sum_in_centre = bool_mask.count(False)
	index = sum_in_boxes / (sum_in_boxes + sum_in_centre)
	return index

def get_corr_indexes(channel, fluorescence, area_idxs, gradient, sigma):
	coefs = []
	for area in area_idxs:
		LS_data, LV_data = probability_map_data(channel, fluorescence, area_num=area, binsize=50, sigma=sigma, three_dimensions=True)
		LS_data, LV_data = remove_corner_points(LS_data, LV_data, percentile=bt.spatial_segregation_calculation_threshold)
		coef = get_corr_index(LS_data, LV_data, gradient)
		coefs.append(coef)
	return coefs

def get_corr_index_mult(channel, fluorescence, areas, gradient, sigma):
	LS_data, LV_data = probability_map_data(channel, fluorescence, area_num=areas, binsize=50, sigma=sigma, three_dimensions=True)
	LS_data, LV_data = remove_corner_points(LS_data, LV_data, percentile=bt.spatial_segregation_calculation_threshold)
	return get_corr_index(LS_data, LV_data, gradient)

def remove_corner_points(ax1_data, ax2_data, percentile):
	max_xy = min(np.max(ax1_data), np.max(ax2_data))
	idxs_below_thresh1 = np.argwhere(ax1_data.flatten() < max_xy * percentile)
	idxs_below_thresh2 = np.argwhere(ax2_data.flatten() < max_xy * percentile)
	idxs_to_remove = np.intersect1d(idxs_below_thresh1, idxs_below_thresh2)
	ax1_data = np.delete(ax1_data.flatten(), idxs_to_remove)
	ax2_data = np.delete(ax2_data.flatten(), idxs_to_remove)
	return ax1_data, ax2_data

def get_SI_and_corrs(channel, area_func, fluorescence, norm, sigma, threshold=None, gradient=0.1, areas_to_combine=None):
    area_labels, dataset_cells, _, _, _ = get_matrix_data(channel, area_func=area_func, postprocess_for_scatter=False, sort_matrix=False, fluorescence=fluorescence, value_norm=norm)
    correlations = get_corr_indexes(channel, fluorescence, area_labels, gradient=gradient, sigma=sigma)
    
    if areas_to_combine is not None:
        area_labels, dataset_cells, correlations = replace_areas_with_combined_area(areas_to_combine, area_labels, dataset_cells=dataset_cells, 
                                                                                        correlations=correlations, corr_channel=channel, corr_fl=fluorescence, corr_gradient=gradient, corr_sigma=sigma)
    if threshold is not None:
        area_labels, dataset_cells, correlations = delete_matrix_and_corr_entries_where_below_threshold(area_labels, dataset_cells, correlations, threshold)
    
    _, num_g1 = fetch_groups(fluorescence) # get sum across each group
    group1_mean_cells_by_area, group2_mean_cells_by_area = dataset_cells[0:num_g1,:], dataset_cells[num_g1:,:]
    SI_avgs, _, _ = calculate_SI_with_errors(group1_mean_cells_by_area, group2_mean_cells_by_area)
    
    return SI_avgs, correlations

def delete_matrix_and_corr_entries_where_below_threshold(area_labels, dataset_cells, correlations, threshold):
    sum_regions = np.sum(dataset_cells, axis=0)
    thresh = np.max(sum_regions) * threshold
    thresh = np.max(sum_regions) * threshold
    idxs_to_del = np.argwhere(sum_regions < thresh)
    dataset_cells = np.delete(dataset_cells, idxs_to_del, axis=1)
    area_labels = np.delete(area_labels, idxs_to_del, axis=0)
    correlations = np.delete(correlations, idxs_to_del, axis=0)
    return area_labels.tolist(), dataset_cells, correlations


# MARK: creating projections for plots

def plot_area_projection(ax, area, padding, axis=0, projcol='k', alpha=0.1):
	projection, (x_min, y_min, z_min), (x_max, y_max, z_max) = get_projection(area, padding=padding, axis=axis)
	ax.contour(projection, colors=projcol, alpha=alpha)
	ax.set_aspect('equal')
	if padding != None: # add padding
		ax.set_xlim(ax.get_xlim()[0] - padding, ax.get_xlim()[1] + padding)
		ax.set_ylim(ax.get_ylim()[0] - padding, ax.get_ylim()[1] + padding)
	if axis != 1:
		ax.invert_yaxis()

	return (x_min, y_min, z_min), (x_max, y_max, z_max)

def project_with_cells(ax, dataset, area, padding, s, channels=None, axis=0, all_cells=False):
	'''
	Plot a coronal or horizontal projection of a brain region with cells superimposed.
	'''
	_, (x_min, y_min, z_min), (x_max, y_max, z_max) = plot_area_projection(ax, area, padding, axis=2)
	def show_cells(ch, colour):
		if all_cells:
			region = (x_min, x_max), (y_min, y_max), (z_min, z_max)
		else:
			parent, children = bt.children_from(area, depth=0)
			areas = [parent] + children
			region = areas
		X_r, Y_r, Z_r = bt._vectorised_get_cells_in(region, dataset, ch)
		X_r = [x-x_min for x in X_r]
		Y_r = [y-y_min for y in Y_r]
		Z_r = [z-z_min for z in Z_r]
		channel_label = f'Channel {ch}'
		if axis == 2: # don't plot any cells if axis is not 0 or 1
			ax.scatter(X_r, Y_r, color=colour, s=s, label=channel_label, zorder=10)
		elif axis == 1:
			ax.scatter(X_r, Z_r, color=colour, s=s, label=channel_label, zorder=10)
	
	channels = dataset._set_channels(channels)
	for i, channel in enumerate(channels):
		show_cells(channel, bt.channel_colours[i])

	return x_min, y_min, z_min

def plot_dataset_projection(ax, data, sum_rather_than_max_bool, axis, areas, padding, vmin, vmax, cmap, logmax):
	f = ax.get_figure()
	plot_area_projection(ax, 997, padding=padding, axis=axis)
	if areas is not None:
		for i in areas:
			plot_area_projection(ax, i, padding=padding, axis=axis)
	
	if not logmax:
		if sum_rather_than_max_bool:
			data_projection = np.sum(data, axis=2-axis)
			title = f'#px along axis={axis}'
		else:
			data_projection = np.max(data, axis=2-axis)
			title = f'max px value along axis={axis}'
	else:
		if sum_rather_than_max_bool:
			data_projection = np.log(data.sum(axis=2-axis))
			title = f'log (#px along axis={axis})'
		else:
			data_projection = np.log(data.max(axis=2-axis))
			title = f'log (max px value along axis={axis})'
	data_projection = data_projection.astype(int)
	if axis == 0:
		data_projection = data_projection.T
	im = ax.imshow(data_projection, vmin=vmin, vmax=vmax, cmap=cmap)
	f.colorbar(im, label=title)
	plt.axis('off')

def get_projection(area, padding=None, axis=0):
	if isinstance(area, list):
		areas = []
		for i in area:
			parent, children = bt.children_from(i, depth=0)
			areas = areas + [parent] + children
	else:
		parent, children = bt.children_from(area, depth=0)
		areas = [parent] + children
	
	atlas_ar = np.isin(bt.atlas.annotation, areas)

	nz = np.nonzero(atlas_ar) # indices where areas are present
	z_min, y_min, x_min = nz[0].min(), nz[1].min(), nz[2].min()
	z_max, y_max, x_max = nz[0].max()+1, nz[1].max()+1, nz[2].max()+1
	
	if padding is not None:
		z_min, y_min, x_min = z_min - padding, y_min - padding, x_min - padding
		z_max, y_max, x_max = z_max + padding, y_max + padding, x_max + padding
	if (z_max > atlas_ar.shape[0]) or (y_max > atlas_ar.shape[1]) or (x_max > atlas_ar.shape[2]):
		print('Watch out! Remove padding for areas that touch the edge of the atlas.')
	if (z_min < 0) or (y_min < 0) or (x_min < 0):
		print('Watch out! Remove padding for areas that touch the edge of the atlas.')
	if bt.debug:
		print('x:'+str(x_min)+' '+str(x_max)+' y:'+str(y_min)+' '+str(y_max)+' z:'+str(z_min)+' '+str(z_max))
	if padding is not None:
		atlas_ar = atlas_ar[z_min : z_max,
							 y_min : y_max,
							 x_min : x_max]

	projection = atlas_ar.any(axis=2-axis)
	projection = projection.astype(int)
	projection = projection.T if axis == 0 else projection # side-on orientation does not need axis swapping
	return projection, (x_min, y_min, z_min), (x_max, y_max, z_max)

def get_sigma_smoothed_3D_array(array, sigma):
	if sigma == 0:
		return array
	x, y, z = np.arange(-3,4,1), np.arange(-3,4,1), np.arange(-3,4,1) # coordinate arrays -- make sure they include (0,0)!
	xx, yy, zz = np.meshgrid(x,y,z)
	kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
	smoothed_array = signal.convolve(array, kernel, mode='same')
	return smoothed_array

def convert_3D_array_to_probability_density_values(array):
	if array.sum() != 0:
		density_array = array / array.sum()
		return density_array
	else:
		print(f'No data in this area. {array.sum()} total across all voxels.')
		return np.zeros_like(array)