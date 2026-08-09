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

import importlib
bt_path = '.'.join(__name__.split('.')[:-1]) # get module path (folder containing this file)
btf = importlib.import_module(bt_path+'.file_management')
bta = importlib.import_module(bt_path+'.area_lists')
bt = importlib.import_module(bt_path+'.analysis')
helpers = importlib.import_module(bt_path+'.plotting_helpers')

import plotly
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import plotly.graph_objs as go
import brainglobe_heatmap as bgh
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Affine2D
from itertools import chain
from matplotlib import cm
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
from myterial import white
bgh.heatmaps.settings.ROOT_COLOR = white
helpers.set_group_colours() # default


def custom_plot(channel, fluorescence, area_names, title='Custom plot', normalisation=None, log=False, horizontal=True, ax=None):
	area_labels, _, _ = bt.get_area_info(area_names)
	datasets = [i for i in bt.datasets if i.fluorescence == fluorescence]
	dataset_cells, axis_title = bt._cells_in_areas_in_datasets(area_names, datasets, channel, normalisation, log)
	helpers.draw_plot(ax, datasets, area_labels, dataset_cells, axis_title, fig_title=title, horizontal=horizontal, l_space=0.3)
	if bt.debug:
		print(dataset_cells)
		percentages = [f'{sum(dataset):.1f}% ' for dataset in dataset_cells]
		print(', '.join(percentages)+'cells are within brain boundaries and in non-tract and non-ventricular areas')
	btf.save(f'customPlot_{title}_ch={channel}_log={log}', as_type='pdf')

def summary_plot(channel, log=False, norm=None, horizontal=True, ax=None):
	summary_areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN']
	custom_plot(channel, summary_areas, title='Whole brain', normalisation=norm, log=log, horizontal=horizontal, ax=ax)

"""
pmap_params = (channel, fluorescence, area_num, pmap_binsize, sigma)
"""
def pmap_corr_scatter_binned(pmap_params, nbins=30, gradient=0.1, vbounds=(None, None)):
	f, ax = plt.subplots(1,1, figsize=(5,5))
	
	ax1_data, ax2_data = helpers.probability_map_data(pmap_params[0], 
														pmap_params[1], 
														area_num=pmap_params[2], 
														binsize=pmap_params[3],
														sigma=pmap_params[4], padding=0, three_dimensions=True)
	ax1_data, ax2_data = helpers.remove_corner_points(ax1_data, ax2_data, bt.spatial_segregation_calculation_threshold)

	max_xy = max(np.max(ax1_data), np.max(ax2_data))
	bins_xy = np.linspace(0, max_xy, nbins + 1)

	hist, _, _, pc = ax.hist2d(ax1_data, ax2_data, bins=(bins_xy, bins_xy), cmap='Greys', vmin=vbounds[0], vmax=vbounds[1]);
	
	ax.set_xlim(0, max_xy * 1.05)
	ax.set_ylim(0, max_xy * 1.05)
	ax.axline((0, 0), (max_xy, max_xy), linestyle=(0, (5, 10)), c='orange') # add y=x line
	xaxis = [0, ax.get_xlim()[1]]
	shading1 = [float(i) * gradient for i in xaxis]
	shading2 = [float(i) / gradient for i in xaxis]
	ax.fill_between(xaxis, [0,0], shading1, alpha=.5, linewidth=0, color=helpers.csolid_group[0])
	ax.fill_between(xaxis, [np.max(shading2), np.max(shading2)], shading2, alpha=.5, linewidth=0, color=helpers.csolid_group[1])
	ax.set_aspect('equal', adjustable='box')
	ax.set_xlabel('LS median voxel probability density')
	ax.set_ylabel('LV median voxel probability density')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	f.colorbar(pc, cax=cax, orientation='vertical')
	
	index = helpers.get_corr_index(ax1_data, ax2_data, gradient)
	ax.text(float(ax.get_xlim()[1]) * 0.9, float(ax.get_ylim()[1]) * 0.75, f'{index:.2g}')
	
	btf.save(f'dmap_corr_scatter_binned_{pmap_params[2]}_fl={pmap_params[1]}', as_type='pdf')

def probability_map(channel, fluorescence, area_num=None, binsize=200, axis=2, sigma=None, subregions=None, subregion_depth=None, projcol='k', padding=0, vmax=None, log=False, log_min=0.0001):
	atlas_res = 10
	assert binsize % atlas_res == 0, f'Binsize must be a multiple of atlas resolution ({atlas_res}um) to display correctly.'
	assert axis in [0, 1, 2], 'Must provide a valid axis number 0-2.'
	if area_num is None:
		area_num = 997
	regions = None # only define regions when subregion_depth and subregions = None
	if subregion_depth is not None:
		regions = bt.children_from(area_num, depth=subregion_depth)[1]
	elif subregions is not None:
		regions = subregions

	groups = helpers.get_bt_groups()
	f, axs = plt.subplots(1, len(groups), figsize=(12,6))

	parent_projection, min_bounds, max_bounds = helpers.get_projection(area_num, padding=padding, axis=axis)
	projections = []
	if regions is not None:
		for child in regions:
			child_projection, (cx_min, cy_min, cz_min), _ = helpers.get_projection(child, padding=padding, axis=axis)
			cx_offset, cy_offset, cz_offset = cx_min - min_bounds[0], cy_min - min_bounds[1], cz_min - min_bounds[2]
			if axis == 2:
				child_projection = np.pad(child_projection, ((cy_offset,0),(cx_offset,0)))
			elif axis == 1:
				child_projection = np.pad(child_projection, ((cz_offset,0),(cx_offset,0)))
			else:
				child_projection = np.pad(child_projection, ((cy_offset,0),(cz_offset,0)))
			projections.append(child_projection)

	def plot_binned_average(ax, channel, area_num, axis, binsize, sigma, group, cmap):
		dmap = helpers.get_density_map(channel, area_num, axis, atlas_res, binsize, sigma, group, min_bounds, max_bounds, fluorescence)
		
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		if log:
			im = ax.imshow(dmap, cmap=cmap, norm=clrs.LogNorm(vmin=log_min, vmax=vmax))
		else:
			im = ax.imshow(dmap, cmap=cmap, vmin=0, vmax=vmax)
			
		plt.colorbar(im, cax)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_title(f'{group} in Fl={fluorescence} datasets')

	for i, g in enumerate(groups):
		ax = axs[i] if isinstance(axs, np.ndarray) else axs
		for child in projections:
			ax.contour(child, colors=projcol, alpha=0.05)
		ax.contour(parent_projection, colors=projcol, alpha=0.1)
		ax.set_aspect('equal')
		if len(groups) > 1: # sample cmaps_group in 2-group mode
			i += 1
		plot_binned_average(ax, channel, area_num, axis, binsize, sigma, g, cmap=helpers.cmaps_group[i])
	btf.save(f'densitymap_{groups}_area={area_num}_subregions={subregions}_axis={axis}_log={log}', as_type='pdf')

def heatmap_spatial_segregation(channel, fluorescence, title, areas, orientation, sigma, gradient=0.1, vmax=None, position=None, cmap='Reds', legend=True, region_labels=True, areas_to_combine=None):
	# orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
	correlations = helpers.get_corr_indexes(channel, fluorescence, areas, gradient, sigma)
	if areas_to_combine is not None:
		area_labels = bt.get_area_info(areas)[0]
		_, _, correlations = helpers.replace_areas_with_combined_area(areas_to_combine, area_labels, do_not_merge=True, 
																		correlations=correlations, corr_channel=channel, corr_fl=fluorescence, corr_gradient=gradient, corr_sigma=sigma)

	regions = dict(zip(areas, correlations))
	cbar_label = f'% signal in magenta/blue (gradient={gradient})'
	f = bgh.Heatmap(regions, position=position, orientation=orientation, title=f'Within-region spatial segregation for Fl={fluorescence}', thickness=1000, atlas_name=bt.atlas.atlas_name, format='2D', vmin=0, vmax=vmax, cmap=cm.get_cmap(cmap), annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)
	plt.figure(f)
	btf.save(f'heatmap_spatialseg_{title}_ch={channel}_Fl={fluorescence}_o={orientation}', as_type='pdf')
	plt.close()

def project_cell_coords(channel, dataset, sum_rather_than_max=False, areas=None, vmin=None, vmax=None, axis=0, padding=None, cmap='Greys', logmax=False, ax=None):
	if ax is None:
		f, ax = plt.subplots(figsize=(10,6))
		f.set_facecolor('white')

	stack = dataset.get_marked_atlas_stack(channel)
	helpers.plot_dataset_projection(ax, stack, sum_rather_than_max, axis, areas, padding, vmin, vmax, cmap, logmax)
	projection_type = 'sum' if sum_rather_than_max else 'max'
	type_name = 'CellCoords'
	btf.save(f'project{type_name}_{dataset.name}_{projection_type}_side={axis}_min={vmin}', dpi=1000, as_type='pdf')

def project_raw_data(channel, dataset, sum_rather_than_max=False, areas=None, vmin=None, vmax=None, axis=0, padding=None, cmap='Greys', logmax=False, ax=None):
	if ax is None:
		f, ax = plt.subplots(figsize=(10,6))
		f.set_facecolor('white')

	stack = btf.open_atlas_registered_stack(dataset, channel)
	helpers.plot_dataset_projection(ax, stack, sum_rather_than_max, axis, areas, padding, vmin, vmax, cmap, logmax)
	projection_type = 'sum' if sum_rather_than_max else 'max'
	type_name = 'RawData'
	btf.save(f'project{type_name}_{dataset.name}_{projection_type}_side={axis}_min={vmin}', dpi=1000, as_type='pdf')

def area_selectivity_scatter(channel, area_func, value_norm='total', custom_lim=None, fluorescence=False, log=False, areas_to_combine=None):
	area_labels, dataset_cells, _, areas_title, axis_title = helpers.get_matrix_data(channel, area_func=area_func, postprocess_for_scatter=False, fluorescence=fluorescence, value_norm=value_norm)
	if areas_to_combine is not None:
		area_labels, dataset_cells, _ = helpers.replace_areas_with_combined_area(areas_to_combine, area_labels, dataset_cells=dataset_cells)
	_, num_g1 = helpers.fetch_groups(fluorescence) # get sum across each group
	collapsed_g1 = np.mean(dataset_cells[0:num_g1,:], axis=0)
	collapsed_g2 = np.mean(dataset_cells[num_g1:,:], axis=0)
	dataset_cells_mean = np.concatenate([[collapsed_g1, collapsed_g2]], axis=1)

	ste1 = bt.ste(dataset_cells[0:num_g1,:], axis=0)
	ste2 = bt.ste(dataset_cells[num_g1:,:], axis=0)

	f, ax = plt.subplots(figsize=(6,6))
	x, y = dataset_cells_mean[0], dataset_cells_mean[1]

	ax.errorbar(x, y, xerr=ste1, yerr=ste2, fmt='o', color='grey', elinewidth=0.2, ms=1.5)
	ax.set_xlabel(f'LS  / mean {axis_title}')
	ax.set_ylabel(f'LV  / mean {axis_title}')
	if log:
		ax.set_yscale('log')
		ax.set_xscale('log')
	ax.set_aspect('equal', adjustable='box') # apply to other scatter plots!
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	
	if not custom_lim:
		max_xy = np.max(dataset_cells_mean) * 10 if log else np.max(dataset_cells_mean) * 1.1
	else:
		max_xy = custom_lim
	min_xy = np.min(dataset_cells_mean) * 0.1 if log else np.min(dataset_cells_mean) * 0.9
	ax.set_ylim(min_xy, max_xy)
	
	ax.set_xlim(min_xy, max_xy)
	ax.axline((0, 0), (max_xy, max_xy), c='orange', linestyle=(0, (5, 10))) # add y=x line
	
	r, p = stats.pearsonr(dataset_cells_mean[0], dataset_cells_mean[1])
	ax.annotate(f'r = {r:.2f}, p = {p:.2g}', xy=(0.05, 0.95), xycoords='axes fraction')
	
	markers = ['s', 'D', '^', '>', '<', 'd', 'x', 'X', '+', '.', 'p', '*', 'h', '1', '2', '3', '4']
	markers_used_g1 = []
	markers_used_g2 = []
	def select_marker(used_markers):
		for marker in markers:
			if marker not in used_markers:
				used_markers.append(marker)
				return marker
		print('Couldn\'t find unused marker')
		return 'o'
	
	for i, (ix, iy) in enumerate(zip(x, y)): # then actually plot
		if ix > iy: # if point is below the line
			if ix - ste1[i] > iy and iy + ste2[i] < ix: # if bounds of error bars are below the line
				marker = select_marker(markers_used_g1)
				ax.scatter([ix], [iy], s=30, c=helpers.csolid_group[0], marker=marker, label=area_labels[i])
		else: # if point is above the line
			if ix + ste1[i] < iy and iy - ste2[i] > ix: # if bounds of error bars are above the line
				marker = select_marker(markers_used_g2)
				ax.scatter([ix], [iy], s=30, c=helpers.csolid_group[1], marker=marker, label=area_labels[i])
				
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	btf.save(f'areaSelectivityScatter_{areas_title}_log={log}', as_type='pdf')
	
# TODO: make the function work properly with the normalisation of _cells_in_areas_in_datasets
def region_comparison_scatter(fluorescence, config=None, areas=None, labels=False, do_test=False, exclude_dataset_idx_from_line_fit=[], swap_axes=False):
	datasets, _ = helpers.fetch_groups(fluorescence=fluorescence)
	dataset_names = [i.name for i in datasets]
	
	if config == 'prePost': # do not provide areas for this config
		x_axis = np.array([i.presynaptics() for i in datasets])
		y_axis = np.array([i.postsynaptics() for i in datasets])
		x_label = f'{bt.presyn_ch} - ({bt.postsyn_region} + {bt.presyn_regions_exclude})'
		y_label = f'{bt.postsyn_region}'
		
		if do_test:
			presynaptics_to_fit = [ele for idx, ele in enumerate(x_axis) if idx not in exclude_dataset_idx_from_line_fit]
			postsynaptics_to_fit = [ele for idx, ele in enumerate(y_axis) if idx not in exclude_dataset_idx_from_line_fit]
			presynaptics_to_fit = np.pad(presynaptics_to_fit, [(0,100)]) # pad with (0,0) values to force fit through origin
			postsynaptics_to_fit = np.pad(postsynaptics_to_fit, [(0,100)])
			z = np.polyfit(presynaptics_to_fit, postsynaptics_to_fit, 1)
			p = np.poly1d(z)
			#line_X = np.pad(x_axis, [(1,0)])
			#ax.plot(line_X, p(line_X), 'k')
			#for idx, cells in enumerate(x_axis):
			#	datasets[idx].true_postsynaptics = p(cells) # set true presynaptics value
			print(f'{x_axis[0]/p(x_axis[0])} inputs per {bt.postsyn_region} neuron.')
			return p
	elif config == 'prePostAntero': # special case for anterogradely-labelled datasets. re-add the postsynaptics back because skimmed is set to True
		print('Warning: When running this function, it is assumed that the cells of the postsynaptic region are not included in the dataset.')
		x_axis = np.array([i.presynaptics() + i.postsynaptics() for i in datasets]) # add postsynaptics because presynaptics() subtracts them by default
		y_axis = np.array([i.postsynaptics() for i in datasets])
		x_axis = list(map(lambda x: x * (bt.resolution_total / 10**9), x_axis)) # do the normalisation like in _cells_in_areas_in_datasets
		x_label = f'{bt.presyn_ch} - {bt.presyn_regions_exclude}'
		y_label = f'{bt.postsyn_region} (=starters for each dataset)'
	elif config == 'areaStarterNorm': # provide one area for this config
		assert areas is not None, 'Provide one area to compare to the starter normaliser value.'
		x_axis = [d.starter_normaliser for d in datasets]
		y_axis = [bt.get_area_info(areas[0], d)[2][0] for d in datasets]
		x_label = f'Starter normaliser value'
		y_label = f'{areas[0]} labelling'
	else: # provide two areas for no config
		assert len(areas) == 2, 'Provide two areas to compare.'
		x_axis = [bt.get_area_info(areas[0], d)[2][0] for d in datasets]
		y_axis = [bt.get_area_info(areas[1], d)[2][0] for d in datasets]
		x_label = f'{areas[0]} labelling'
		y_label = f'{areas[1]} labelling'
	
	f, ax = plt.subplots(figsize=(6,6))
	f.set_facecolor('white')
	if swap_axes:
		x_axis, y_axis = y_axis, x_axis
		x_label, y_label = y_label, x_label
	sns.regplot(x=x_axis, y=y_axis, ci=95, robust=True, line_kws=dict(color='gray'), scatter_kws=dict(color=helpers.colours_from_labels(dataset_names)), ax=ax)
	if labels:
		for i, name in enumerate(dataset_names):
			ax.annotate(name, (x_axis[i], y_axis[i]))
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.set_xlim(0, None)
	btf.save(f'regionComparison_c={config}_a={areas}_F={fluorescence}', as_type='pdf')
	return (x_axis, y_axis)

def area_total_signal_bar(channel, area_func, value_norm='total', fluorescence=False, areas_to_combine=None):
	area_labels, dataset_cells, _, areas_title, _ = helpers.get_matrix_data(channel, area_func=area_func, postprocess_for_scatter=False, sort_matrix=False, fluorescence=fluorescence, value_norm=value_norm)
	if areas_to_combine is not None:
		area_labels, dataset_cells, _ = helpers.replace_areas_with_combined_area(areas_to_combine, area_labels, dataset_cells=dataset_cells)
	_, num_g1 = helpers.fetch_groups(fluorescence)
	ste1 = bt.ste(dataset_cells[0:num_g1,:], axis=0)
	ste2 = bt.ste(dataset_cells[num_g1:,:], axis=0)

	collapsed_g1 = np.mean(dataset_cells[0:num_g1,:], axis=0)
	collapsed_g2 = np.mean(dataset_cells[num_g1:,:], axis=0)
	
	sort_order = helpers.get_sorting_from_SI(dataset_cells, fluorescence)
	area_labels = [area_labels[i] for i in sort_order]
	collapsed_g1 = collapsed_g1[sort_order[::1]]
	collapsed_g2 = collapsed_g2[sort_order[::1]]
	ste1 = ste1[sort_order[::1]]
	ste2 = ste2[sort_order[::1]]
	
	f, ax = plt.subplots(figsize=(6,6))
	# add offset to error bars
	trans1 = Affine2D().translate(0.0, -0.1) + ax.transData
	trans2 = Affine2D().translate(0.0, +0.1) + ax.transData
	
	groups = helpers.get_bt_groups()
	ax.barh(area_labels, collapsed_g1, xerr=ste1, error_kw={'transform':trans1}, label=groups[0], color=helpers.csolid_group[0])
	ax.barh(area_labels, collapsed_g2, xerr=ste2, error_kw={'transform':trans2}, label=groups[1], color=helpers.csolid_group[1], left=collapsed_g1)
	ax.set_ylabel(f'Area')
	ax.set_xlabel(f'norm={value_norm}')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	btf.save(f'areaTotalSignalBar_{areas_title}', as_type='pdf')

def density_map_corr_bar(channel, fluorescence, area_func, sigma, gradient=0.1, value_norm='total', areas_to_combine=None):
	area_labels, areas_title = area_func
	area_labels = bt.get_area_info(area_labels)[0]
	correlations = helpers.get_corr_indexes(channel, fluorescence, area_labels, gradient, sigma)
	_, dataset_cells, _, _, _ = helpers.get_matrix_data(channel, area_func=area_func, postprocess_for_scatter=False, sort_matrix=False, fluorescence=fluorescence, value_norm=value_norm)

	if areas_to_combine is not None:
		area_labels, dataset_cells, correlations = helpers.replace_areas_with_combined_area(areas_to_combine, area_labels, dataset_cells=dataset_cells,
																							correlations=correlations, corr_channel=channel, corr_fl=fluorescence, corr_gradient=gradient, corr_sigma=sigma)

	sort_order = helpers.get_sorting_from_SI(dataset_cells, fluorescence)
	area_labels = [area_labels[i] for i in sort_order]
	correlations = [correlations[i] * 100 for i in sort_order]
	
	f, ax = plt.subplots(figsize=(6,6))
	ax.barh(area_labels, correlations, color='k')
	ax.set_ylabel(f'Area')
	ax.set_xlabel(f'% signal in magenta/blue')
	ax.set_xlim(0,100)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.axvline(x=0, c='k')
	btf.save(f'density_map_corr_bar_{areas_title}', as_type='pdf')

def area_selectivity_with_errors(channel, area_func, value_norm='total', fluorescence=False, areas_to_combine=None):
	area_labels, dataset_cells, _, areas_title, axis_title = helpers.get_matrix_data(channel, area_func=area_func, postprocess_for_scatter=False, sort_matrix=False, fluorescence=fluorescence, value_norm=value_norm)
	if areas_to_combine is not None:
		area_labels, dataset_cells, _ = helpers.replace_areas_with_combined_area(areas_to_combine, area_labels, dataset_cells=dataset_cells)
	_, num_g1 = helpers.fetch_groups(fluorescence) # get sum across each group

	group1_cells_by_area, group2_cells_by_area = dataset_cells[0:num_g1,:], dataset_cells[num_g1:,:]
	avgs, stes, colours = helpers.calculate_SI_with_errors(group1_cells_by_area, group2_cells_by_area)
	
	sort_order = helpers.get_sorting_from_SI(dataset_cells, fluorescence)
	area_labels = [area_labels[i] for i in sort_order]
	avgs = [avgs[i] for i in sort_order]
	stes = [stes[i] for i in sort_order]
	colours = [colours[i] for i in sort_order]
	
	f, ax = plt.subplots(figsize=(6,6))
	ax.barh(area_labels, avgs, color=colours, xerr=stes) #, ecolor=error_colours)
	ax.set_ylabel(f'Area')
	ax.set_xlabel(f'% selectivity  ({axis_title})')
	ax.set_xlim(-1,1)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.axvline(x=0, c='k')
	btf.save(f'areaSelectivity_errors_{areas_title}', as_type='pdf')

def probability_map_overlap(channel, fluorescence, area_num=None, areas_mask_out=None, binsize=200, axis=2, sigma=None, subregions=None, subregion_depth=None, projcol='k', padding=0, lower_lim=0, saturation_multiplier=1, colour_set="magenta-cyan"):
	atlas_res = 10
	assert binsize % atlas_res == 0, f'Binsize must be a multiple of atlas resolution ({atlas_res}um) to display correctly.'
	assert axis in [0, 1, 2], 'Must provide a valid axis number 0-2.'
	if area_num is None:
		area_num = 997
	regions = None # only define regions when subregion_depth and subregions = None
	if subregion_depth is not None:
		regions = bt.children_from(area_num, depth=subregion_depth)[1]
	elif subregions is not None:
		regions = subregions

	f, ax = plt.subplots(figsize=(6,6))
	if padding is None:
		plt.axis('off')
	mins, _ = helpers.plot_area_projection(ax, area_num, padding=padding, axis=axis, projcol=projcol, alpha=0.1)
	if regions is not None:
		for child in regions:
			child_projection, (cx_min, cy_min, cz_min), _ = helpers.get_projection(child, padding=padding, axis=axis)
			cx_offset, cy_offset, cz_offset = cx_min - mins[0], cy_min - mins[1], cz_min - mins[2]
			if axis == 2:
				child_projection = np.pad(child_projection, ((cy_offset,0),(cx_offset,0)))
			elif axis == 1:
				child_projection = np.pad(child_projection, ((cz_offset,0),(cx_offset,0)))
			else:
				child_projection = np.pad(child_projection, ((cy_offset,0),(cz_offset,0)))
			#child_projection = child_projection.T if axis == 0 else child_projection # side-on orientation does not need axis swapping
			ax.contour(child_projection, colors=projcol, alpha=0.05)
	
	group1_data, group2_data = helpers.probability_map_data(channel, fluorescence, area_num=area_num, binsize=binsize, sigma=sigma, padding=padding, three_dimensions=False, axis=axis, exclude_subregions=areas_mask_out)
	group1_name = helpers.get_bt_groups()[0]
	group2_name = helpers.get_bt_groups()[1]
	
	im = helpers.map_for_data(group1_data, group2_data, lower_lim=lower_lim, saturation_multiplier=saturation_multiplier, colour_set=colour_set)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	
	im = ax.imshow(im, cmap=helpers.cmap_groups_mixed_for_density_maps)
	cbar = plt.colorbar(im, cax)

	cbar.ax.get_yaxis().set_ticks([])
	for j, lab in enumerate([f'${group1_name}=0, {group2_name}=1$', f'${group1_name}=1, {group2_name}=1$', f'${group1_name}=1, {group2_name}=0$']):
		cbar.ax.text(2, j / 2, lab, ha='left', va='center')
	ax.set_xticklabels([])
	ax.set_yticklabels([])

	area_name = bt.get_area_info(area_num)[0][0]
	ax.set_title(f'{area_name} {group1_name}-{group2_name} mix from Fl={fluorescence} datasets')
	btf.save(f'dmap_HSV_{area_name}_ax={axis}_F={fluorescence}_subrgns={subregions}_sat={saturation_multiplier}', as_type='pdf')

def region_signal_matrix(channel, area_func, value_norm='total', postprocess_for_scatter=False, vmax=None, figsize=(3,6), fluorescence=False, log_plot=True, sorting=True, ax=None,
						 areas_to_combine=None):
	if ax is None:
		_, ax = plt.subplots(figsize=figsize)
	area_labels, dataset_cells, datasets, areas_title, axis_title = helpers.get_matrix_data(channel, area_func=area_func, postprocess_for_scatter=postprocess_for_scatter, sort_matrix=False, fluorescence=fluorescence, value_norm=value_norm)
	x_labels = [i.name for i in datasets]
	if postprocess_for_scatter:
		x_labels = helpers.get_bt_groups()

	if areas_to_combine is not None:
		area_labels, dataset_cells, _ = helpers.replace_areas_with_combined_area(areas_to_combine, area_labels, dataset_cells=dataset_cells)
	
	if sorting:
		sort_order = helpers.get_sorting_from_SI(dataset_cells, fluorescence)
		dataset_cells = dataset_cells[:, sort_order[::-1]]
		area_labels = [area_labels[i] for i in reversed(sort_order)]
	
	_, num_g1 = helpers.fetch_groups(fluorescence=fluorescence)
	g1_mask = np.full(dataset_cells.T.shape, False) # get shape of the matrix
	g1_mask[:, num_g1:] = True # array where LS datasets on left are False
	
	if vmax == None:
		print('Warning: If vbounds is set to None, halves of the matrix may not have the same vmax.')
		vmax = dataset_cells.max()
	
	if log_plot:
		norm = clrs.LogNorm(vmin=1, vmax=vmax) # seems like both halves render with same vmin vmax without specifying
		sns.heatmap(dataset_cells.T, annot=False, mask=~g1_mask, cmap=helpers.cmaps_group[2], xticklabels=x_labels, yticklabels=area_labels, square=True, ax=ax, norm=norm)
		sns.heatmap(dataset_cells.T, annot=False, mask=g1_mask, cmap=helpers.cmaps_group[1], xticklabels=x_labels, yticklabels=area_labels, square=True, ax=ax, norm=norm)# , cbar_kws=dict(ticks=[])
	else:
		sns.heatmap(dataset_cells.T, annot=False, mask=~g1_mask, cmap=helpers.cmaps_group[2], xticklabels=x_labels, yticklabels=area_labels, square=True, ax=ax, vmin=0, vmax=vmax)
		sns.heatmap(dataset_cells.T, annot=False, mask=g1_mask, cmap=helpers.cmaps_group[1], xticklabels=x_labels, yticklabels=area_labels, square=True, ax=ax, vmin=0, vmax=vmax)# , cbar_kws=dict(ticks=[])
	
	if postprocess_for_scatter:
		colours = helpers.colours_from_labels([datasets[0].name, datasets[-1].name])
	else:
		colours = helpers.colours_from_labels(x_labels)
	[t.set_color(colours[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
	ax.set_title(areas_title)
	ax.set_ylabel(f'{axis_title}')
	btf.save(f'regionSignalMatrix_{areas_title}_log={log_plot}', as_type='pdf')
	return area_labels, dataset_cells

def generate_group_comparison_heatmaps(channel, fluorescence, areas, title, norm, dmap_sigma=0.5, orientation='sagittal', pair_plot_vmax=None, spatialseg_vmax=None, position=None, legend=True, region_labels=True, areas_to_combine=None):
	datasets, num_g1 = helpers.fetch_groups(fluorescence=fluorescence)
	dataset_cells, cbar_label = bt._cells_in_areas_in_datasets(areas, datasets, 'r', normalisation=norm)
	
	area_labels = bt.get_area_info(areas)[0]
	if areas_to_combine is not None:
		_, dataset_cells, _ = helpers.replace_areas_with_combined_area(areas_to_combine, area_labels, np.array(dataset_cells), do_not_merge=True)
	area_labels = bt.get_area_acronyms(areas)
	
	cells_g1 = np.array(dataset_cells)[0:num_g1,:]
	cells_g2 = np.array(dataset_cells)[num_g1:,:]

	group1_name = helpers.get_bt_groups()[0]
	group2_name = helpers.get_bt_groups()[1]

	def make_paired_plots(areas, cells_g1, cells_g2, title, pair_plot_vmax):
		mean_global = np.mean(np.array(dataset_cells), axis=0)
		mean_g1, mean_g2 = np.mean(cells_g1, axis=0), np.mean(cells_g2, axis=0)
		mean_global /= np.sum(mean_global)
		mean_g1 /= np.sum(mean_g1)
		mean_g2 /= np.sum(mean_g2)

		if pair_plot_vmax is None:
			pair_plot_vmax = np.max(np.concatenate([[mean_g1, mean_g2]], axis=1)) # get max value in the two arrays to set the same vmax for both plots
		
		global_regions = dict(zip(areas, mean_global))
		g1_regions = dict(zip(areas, mean_g1))
		g2_regions = dict(zip(areas, mean_g2))
		f1 = bgh.Heatmap(global_regions, position=position, orientation=orientation, title=f'{title}: Global', atlas_name=bt.atlas_name, format='2D', vmin=0, vmax=None, cmap=helpers.cmap_midpoint_of_both_groups, annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)
		plt.figure(f1)
		btf.save(f'heatmap_means_{title}_ch={channel}_o={orientation}_group=global', as_type='pdf')
		plt.close()
		f2 = bgh.Heatmap(g1_regions, position=position, orientation=orientation, title=f'{title}: {group1_name}', atlas_name=bt.atlas_name, format='2D', vmin=0, vmax=pair_plot_vmax, cmap=helpers.cmaps_group[1], annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)
		plt.figure(f2)
		btf.save(f'heatmap_means_{title}_ch={channel}_o={orientation}_group={group1_name}', as_type='pdf')
		plt.close()
		f3 = bgh.Heatmap(g2_regions, position=position, orientation=orientation, title=f'{title}: {group2_name}', atlas_name=bt.atlas_name, format='2D', vmin=0, vmax=pair_plot_vmax, cmap=helpers.cmaps_group[2], annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)
		plt.figure(f3)
		btf.save(f'heatmap_means_{title}_ch={channel}_o={orientation}_group={group2_name}', as_type='pdf')
		plt.close()
		
	def make_SI_plot(areas, cells_g1, cells_g2, title):
		avgs, _ = helpers.calculate_SI_from_means(cells_g1, cells_g2)
		bounds = np.abs(avgs).max()
		regions = dict(zip(areas, avgs))
		cbar_SI_label = f'{group2_name} - {group1_name} ({cbar_label})'
		f = bgh.Heatmap(regions, position=position, orientation=orientation, title=f'{title}: SI', atlas_name=bt.atlas_name, format='2D', vmin=-bounds, vmax=bounds, cmap=helpers.cmap_group1_to_group2, annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_SI_label)
		plt.figure(f)
		btf.save(f'heatmap_SI_{title}_ch={channel}_o={orientation}', as_type='pdf')
		plt.close()

	make_paired_plots(area_labels, cells_g1, cells_g2, title, pair_plot_vmax)
	make_SI_plot(area_labels, cells_g1, cells_g2, title)
	heatmap_spatial_segregation('r', fluorescence, title, area_labels, orientation, sigma=dmap_sigma, gradient=0.1, vmax=spatialseg_vmax, position=position, cmap='Greys', legend=False, areas_to_combine=areas_to_combine)



















# MARK: NOT YET INTEGRATED

def generate_mega_overview_figure(channel, title):
	f = plt.figure(figsize=(24, 35))
	gs = f.add_gridspec(60, 30)
	f.suptitle(title, y=0.92, size='xx-large', weight='bold')
	f.set_facecolor('white')
	ax1 = f.add_subplot(gs[0:9, 5:20])
	ax_totals = f.add_subplot(gs[0:4, 24:27])
	ax_io = f.add_subplot(gs[5:9, 24:27])
	ax2, ax3, ax4 = f.add_subplot(gs[12:22, 0:9]), f.add_subplot(gs[12:22, 10:20]), f.add_subplot(gs[12:22, 21:30])
	ax5, ax6, ax7 = f.add_subplot(gs[30:40, 0:9]), f.add_subplot(gs[30:40, 10:20]), f.add_subplot(gs[30:40, 21:30])
	ax8, ax9, ax10 = f.add_subplot(gs[50:60, 0:9]), f.add_subplot(gs[50:60, 10:20]), f.add_subplot(gs[50:60, 21:30])
	axes = [ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]

	# total cells plot
	cells = [i.num_cells(ch1=True) for i in bt.datasets]
	names = [i.group for i in bt.datasets]
	titles = ['Injection site', 'Total cells in channel 1']
	df = pd.DataFrame(zip(names, cells), columns=titles)
	if bt.grouped:
		sns.barplot(x=titles[0], y=titles[1], order=['LV','LS'], ax=ax_totals, data=df, ci=None)
		sns.stripplot(x=titles[0], y=titles[1], order=['LV','LS'], dodge=True, edgecolor='w', linewidth=0.5, ax=ax_totals, data=df)
	else:
		sns.barplot(x=titles[0], y=titles[1], ax=ax_totals, data=df, ci=None)

	# IO cells plot
	io_cells = [bt.get_area_info(['IO'], i, channel)[-1] for i in bt.datasets]
	io_cells = io_cells[0] if len(io_cells) == 1 else chain.from_iterable(io_cells)
	io_titles = ['Injection site', 'Cells in inferior olive']
	io_df = pd.DataFrame(zip(names, io_cells), columns=io_titles)
	if bt.grouped:
		sns.barplot(x=io_titles[0], y=io_titles[1], order=['LV','LS'], ax=ax_io, data=io_df, ci=None)
		sns.stripplot(x=io_titles[0], y=io_titles[1], order=['LV','LS'], dodge=True, edgecolor='w', linewidth=0.5, ax=ax_io, data=io_df)
	else:
		sns.barplot(x=io_titles[0], y=io_titles[1], ax=ax_io, data=io_df, ci=None)

	# summary and zoom plots for each area
	summary_plot(channel, log=False, norm=None, horizontal=True, ax=ax1)
	summary_areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN']
	summary_areas, _, _ = bt.get_area_info(summary_areas)
	for idx, ax in enumerate(axes[:-1]):
		generate_zoom_plot(summary_areas[idx], threshold=0.1, ax=ax)
	generate_zoom_plot(summary_areas[-1], depth=1, threshold=0, ax=ax10)


def generate_starter_cell_bar(ax=None, true_only=False, log=False):
	if ax is None:
		f, ax = plt.subplots(figsize=(8,5))
		f.set_facecolor('white')
	datasets = [i for i in bt.datasets if not i.fluorescence]
	dataset_names = [i.name for i in datasets]
	if true_only:
		ax.set(ylabel=f'Starter cells in {bt.postsyn_region} (ch={bt.postsyn_ch})')
		starter_cells = [bt.get_area_info(bt.postsyn_region, dataset=d, channels=bt.postsyn_ch)[2][0] for d in datasets]
	else:
		ax.set(ylabel=f'Starter cells in {bt.postsyn_region} (corrected)')
		starter_cells = [i.postsynaptics() for i in datasets] # green cells in starter region
	sns.barplot(x=dataset_names, y=starter_cells, ax=ax)
	if log:
		ax.set_yscale('log')
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

def generate_projection_plot(area, include_surrounding=False, padding=10, ch1=None, colours=['r','g'], s=2, contour=True, legend=True):
	group1, group2 = helpers.get_bt_groups()
	f, axs = plt.subplots(2, 2, figsize=(12,9), sharex=False)
	f.set_facecolor('white')
	plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
	for dataset in bt.datasets:
		xax = 0 if dataset.group == group1 else 1
		x, y, z = helpers.project_with_cells(axs[0,xax], dataset, area, padding, ch1, s, contour, colours=colours, all_cells=include_surrounding)
		f.suptitle(f'Cell distribution in {area} where x={x}, y={y}, z={z} across '+'_'.join([i.name for i in bt.datasets]))
		helpers.project_with_cells(axs[1,xax], dataset, area, padding, ch1, s, contour, axis=1, colours=colours, all_cells=include_surrounding)
	axs[0,0].set_title(f'Cells inside {group1} datasets')
	axs[0,1].set_title(f'Cells inside {group2} datasets')
	axs[0,0].set_ylabel('Y axis distance from dorsal end of region / px')
	axs[1,0].set_ylabel('Z axis distance from rostral end of region / px')
	axs[1,0].set_xlabel('X axis distance from right end of region / px')
	axs[1,1].set_xlabel('X axis distance from right end of region / px')
	for ax in list(chain.from_iterable(axs)):
			ax.invert_yaxis()
			ax.grid()
	f.tight_layout()
	if legend:
		helpers.display_legend_subset(axs[0,0], (0,1,))
		helpers.display_legend_subset(axs[0,1], (0,1,))
	
def _generate_starter_validation_plot(padding=10, ch1=None, s=2, contour=True):
	area = bt.postsyn_region
	if area is None:
		print('Starter region unknown. Define it with bt.postsyn_region = \'IO\'')
		return
	for dataset in bt.datasets:
		f, axs = plt.subplots(2, 2, figsize=(12,9), sharex=True)
		f.set_facecolor('white')
		plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
		x, y, z = helpers.project_with_cells(axs[0,0], dataset, area, padding, ch1, s, contour)
		f.suptitle(f'Cell distribution in {dataset.name} {area} where x={x}, y={y}, z={z}')
		helpers.project_with_cells(axs[0,1], dataset, area, padding, ch1, s, contour, all_cells=True)
		helpers.project_with_cells(axs[1,0], dataset, area, padding, ch1, s, contour, axis=1)
		helpers.project_with_cells(axs[1,1], dataset, area, padding, ch1, s, contour, axis=1, all_cells=True)
		axs[0,0].set_title(f'Cells inside registered area')
		axs[0,1].set_title(f'All cells')
		axs[0,0].set_ylabel('Y axis distance from dorsal end of region / px')
		axs[1,0].set_ylabel('Z axis distance from rostral end of region / px')
		axs[1,0].set_xlabel('X axis distance from right end of region / px')
		axs[1,1].set_xlabel('X axis distance from right end of region / px')
		for ax in list(chain.from_iterable(axs)):
			ax.invert_yaxis()
			ax.grid()
		f.tight_layout()
		btf.save(f'injection_{dataset.name}_{area}', as_type='png')
	print('View results in braintracer/TRIO.')

def generate_3D_shape(areas, colours):
	assert len(areas) == len(colours), 'Each area should have a corresponding colour.'
	area_nums = bt.get_area_info(areas)[1]
	def _subsample_atlas_pixels(x, y, z): # reduce pixel density 20x
		x = [val for i, val in enumerate(x) if i % 20 == 0]
		y = [val for i, val in enumerate(y) if i % 20 == 0]
		z = [val for i, val in enumerate(z) if i % 20 == 0]
		return x, y, z
	data = []
	for idx, area_num in enumerate(area_nums):
		z_vals, y_vals, x_vals = np.nonzero(bt.atlas.annotation == area_num)
		x_vals, y_vals, z_vals = _subsample_atlas_pixels(x_vals, y_vals, z_vals)
		trace = go.Scatter3d(x = y_vals, y = x_vals, z = z_vals, mode='markers',
		marker={'size': 1, 'opacity': 0.8, 'color':colours[idx]})
		data.append(trace)
	plotly.offline.init_notebook_mode()
	layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
	plot_figure = go.Figure(data=data, layout=layout)
	plotly.offline.iplot(plot_figure)

def generate_heatmap_difference(channel, areas, orientation, position=None, normalisation='total', cmap='bwr', legend=True, region_labels=True, limit=None):
	# orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
	group_names = [i.group for i in bt.datasets]
	values, cbar_label = bt._cells_in_areas_in_datasets(areas, bt.datasets, channel, normalisation=normalisation)
	groups, cells = helpers.compress_into_groups(group_names, values)
	cells = np.array(cells)
	differences = cells[0] - cells[1]
	bounds = np.abs(differences).max()
	regions = dict(zip(areas, differences))
	cbar_label = 'LS - LV inputs / postsynaptic cell'
	if limit is not None:
		bounds = np.abs(limit)
	bgh.Heatmap(regions, position=position, orientation=orientation, thickness=1000, atlas_name=bt.atlas.atlas_name, format='2D', vmin=-bounds, vmax=bounds, cmap=cm.get_cmap(cmap), annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)

def generate_heatmap_ratios(channel, areas, orientation, position=None, normalisation='total', cmap='bwr', legend=True, region_labels=True, limit=None, add=False):
	# orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
	group_names = [i.group for i in bt.datasets]
	values, cbar_label = bt._cells_in_areas_in_datasets(areas, bt.datasets, channel, normalisation=normalisation)
	groups, cells = helpers.compress_into_groups(group_names, values)
	cells = np.array(cells)
	print(cells[cells==0])
	ratios = cells[0] / cells[1]
	if add:
		ratios = (cells[0] / cells[1]) + 1
	differences = np.log10(ratios, where=ratios > 0) # calculate log ratio rather than absolute difference
	bounds = np.abs(differences).max()
	regions = dict(zip(areas, differences))
	cbar_label = 'LS - LV inputs / postsynaptic cell'
	if limit is not None:
		bounds = np.abs(limit)
	bgh.Heatmap(regions, position=position, orientation=orientation, thickness=1000, atlas_name=bt.atlas.atlas_name, format='2D', vmin=-bounds, vmax=bounds, cmap=cm.get_cmap(cmap), annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)
	
def generate_slice_heatmap(channel, position, normalisation='total', depth=3, region_labels=True):
	group_names = [i.group for i in bt.datasets]
	_, areas = bt.children_from('root', depth=depth)
	areas = bt.area_indexes.loc[areas, 'acronym'].tolist()

	values, cbar_label = bt._cells_in_areas_in_datasets(areas, bt.datasets, channel, normalisation=normalisation)
	groups, cells = helpers.compress_into_groups(group_names, values)

	highest_value = np.max(cells) # remove regions in the bottom 1% from plot
	g1_regions = dict(zip(areas, cells[0]))
	g2_regions = dict(zip(areas, cells[1]))
	bgh.Heatmap(g1_regions, position=position, orientation='frontal', title=groups[0], thickness=1000, atlas_name=bt.atlas.atlas_name, format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap('hot'), annotate_regions=region_labels).show(show_legend=True, cbar_label=cbar_label)
	bgh.Heatmap(g2_regions, position=position, orientation='frontal', title=groups[1], thickness=1000, atlas_name=bt.atlas.atlas_name, format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap('hot'), annotate_regions=region_labels).show(show_legend=True, cbar_label=cbar_label)

def bin_3D_matrix(channel, area_num=None, binsize=500, aspect='equal', zscore=False, sigma=None, vbounds=None, threshold=1, override_order=None, order_method=None, blind_order=False, cmap='Reds', covmat=False, figsize=(8,8)):
	x_bins, y_bins, z_bins = helpers.get_bins(0, binsize), helpers.get_bins(1, binsize), helpers.get_bins(2, binsize)
	
	groups = helpers.get_bt_groups()
	datasets1 = [i for i in bt.datasets if i.group == groups[0]]
	datasets2 = [i for i in bt.datasets if i.group == groups[1]]
	datasets = datasets1 + datasets2
	y_labels = [i.name for i in datasets]

	voxels = []
	num_nonzero_bins = []
	for d in datasets:
		if area_num is None:
			points = np.array(d.cell_coords(channel)).T
			old_num_points = points.shape
			points = points[points.min(axis=1)>=0,:] # remove coordinates with negative values so the next step works
			neg_num_points = points.shape
			points_IO = np.array(bt._vectorised_get_cells_in([83,528], d, ch1=ch1)).T
			dims = np.maximum(points_IO.max(0),points.max(0))+1 # this and following line are to filter out IO points from points
			points = points[~np.in1d(np.ravel_multi_index(points.T,dims),np.ravel_multi_index(points_IO.T,dims))]
			if bt.debug:
				print(f'Num neg points removed: {old_num_points[0] - neg_num_points[0]}, num IO+CBX points removed: {neg_num_points[0] - points.shape[0]}, num acc values: {points_IO.shape[0]}')
		else:
			parent, children = bt.children_from(area_num, depth=0)
			areas = [parent] + children
			points = np.array(bt._vectorised_get_cells_in(areas, d, channel)).T
		hist, _ = np.histogramdd(points, bins=(x_bins, y_bins, z_bins), range=((0,1140),(0,800),(0,1320)), density=False)
		num_nonzero_bins.append(np.count_nonzero(hist)) # just debug stuff
		last_hist_shape = hist.shape			
		hist = helpers.apply_sigma_smoothing_to_3D_array(hist, sigma)
		cell_voxels = hist.flatten() # generates list of cell numbers within defined voxels
		voxels.append(cell_voxels)
	all_voxels = np.array(voxels)
	print(f'Bins of last dataset: {last_hist_shape}, average number of bins containing cells: {np.mean(num_nonzero_bins)}')

	summed = np.sum(all_voxels, axis=0) # delete columns where mean is below threshold
	averaged = summed / all_voxels.shape[0]
	idxs_to_remove = np.where(averaged <= threshold)[0]
	print(f'{len(idxs_to_remove)} voxels removed by threshold={threshold}, unless showing correlation matrix.')
	voxels = np.delete(all_voxels, idxs_to_remove, axis=1)

	if zscore:
		voxels = stats.zscore(voxels, axis=0)
		all_voxels = stats.zscore(all_voxels, axis=0)

	if not covmat:
		f, ax = plt.subplots(figsize=figsize)
		f.set_facecolor('white')
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='1%', pad=0.02)
		if vbounds is None:
			im = ax.matshow(voxels, aspect=aspect, cmap=cmap)
		else:
			im = ax.matshow(voxels, aspect=aspect, cmap=cmap, vmin=vbounds[0], vmax=vbounds[1])
		f.colorbar(im, cax=cax, orientation='vertical')
		ax.set_title(f'Matrix of {binsize} um voxels')
	else:
		f, ax = plt.subplots(figsize=figsize)
		f.set_facecolor('white')
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0.05)

		cov = np.corrcoef(all_voxels, rowvar=1)
		if np.isnan(np.sum(cov)):
			cov = np.nan_to_num(cov, copy=True, nan=0.0)
			print('Warning: All correlations probably NaN. Specified region likely contains no cells.')
		if override_order is None:
			if order_method is not None:
				if not blind_order:
					d1b, d2b = len(datasets1), len(datasets2)
					print(f'Group 1: {d1b}, Group 2: {d2b}')
					mat1 = cov[:d1b,:d1b]
					mat2 = cov[d1b:,d1b:] # split the matrix up into two sub-matrices
					mat1, res_order1, _ = helpers.compute_serial_matrix(mat1, order_method)
					mat2, res_order2, _ = helpers.compute_serial_matrix(mat2, order_method)
					res_order2 = list(np.array(res_order2) + d1b) # offset dataset indexes
					res_order = res_order1 + res_order2 # create final order array
				else:
					_, res_order, _ = helpers.compute_serial_matrix(cov, order_method)
				print(f'Sorted order: {res_order}')
				sorted_mat = cov[res_order,:] # sort both axes of the matrix by the computed order
				cov = sorted_mat[:,res_order]
				y_labels = list(np.array(y_labels)[res_order])
		else:
			sorted_mat = cov[override_order,:] # sort both axes of the matrix by the computed order
			cov = sorted_mat[:,override_order]
			y_labels = list(np.array(y_labels)[override_order])
		if vbounds is None:
			im = ax.matshow(cov, cmap=cmap)
		else:
			im = ax.matshow(cov, cmap=cmap, vmin=vbounds[0], vmax=vbounds[1])
		f.colorbar(im, cax=cax, orientation='vertical')
		ax.set_title(f'Correlation matrix of {binsize} um voxels')
		ax.set_xticks(range(len(y_labels)))
		ax.set_xticklabels(y_labels, rotation=90)
	ax.set_yticks(range(len(y_labels)))
	ax.set_yticklabels(y_labels)
	colours = helpers.colours_from_labels(y_labels)
	if covmat:
		[t.set_color(colours[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
	[t.set_color(colours[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]

def zoom_plot(channel, parent, depth=0, threshold=None, normalisation=None, log=False, horizontal=True, ax=None):
	title = f'Zoom into {parent} at depth {depth}'
	datasets = [i for i in bt.datasets]
	parent, children = bt.children_from(parent, depth)
	area_labels = bt.get_area_info(children)[0]
	dataset_cells, axis_title = bt._cells_in_areas_in_datasets(children, datasets, channel, normalisation, log)

	def remove_regions_below_threshold(cells, labels): # exclude brain areas where the average of all datasets is less than threshold
		cells_array = np.array(cells)
		print(cells_array.shape)
		summed = np.sum(cells_array, axis=0)
		averaged = summed / cells_array.shape[0]
		print(averaged.shape)
		idxs_to_remove = np.where(averaged < threshold)[0] # determine indexes at which to remove the data and labels
		print(idxs_to_remove)

		for idx, cells_dset in enumerate(cells): # remove the corresponding data points and labels
			cells[idx] = [v for i, v in enumerate(cells_dset) if i not in idxs_to_remove]
		if bt.debug:
			names_removed = [v for i, v in enumerate(labels) if i in idxs_to_remove]
			string = ', '.join(names_removed)
			print(f'Areas excluded: {string}')
		labels = [v for i, v in enumerate(labels) if i not in idxs_to_remove]
		return cells, labels

	if threshold is not None:
		dataset_cells, area_labels = remove_regions_below_threshold(dataset_cells, area_labels)
	helpers.draw_plot(ax, datasets, area_labels, dataset_cells, axis_title, fig_title=title, horizontal=horizontal, l_space=0.3)

def generate_zoom_plot(channel, parent_name, depth=0, threshold=0, prop_all=True, ax=None):
	'''
	prop_all: True; cell counts as fraction of total cells in signal channel. False; cell counts as fraction in parent area
	'''
	datasets = bt.datasets
	new_counters = [i.ch1_cells_by_area for i in datasets]
	original_counters = [i.raw_ch1_cells_by_area for i in datasets]

	parent, children = bt.children_from(parent_name, depth)
	list_cells, axis_title = bt._cells_in_areas_in_datasets(children, datasets, channel, normalisation='presynaptics')
	### rewrite needed for next 3 paras

	list_cells = [] # 2D array of number of cells in each child area for each dataset
	for counter in new_counters:
		try:
			area_labels, _, cells = bt.get_area_info(children, counter) # TODO: change to no counter
		except IndexError:
			print('Cannot zoom into an area with no children.')
			return
		list_cells.append(cells)

	parent_totals = []
	for idx, cells in enumerate(list_cells): # do conversion to % area cells before/after sorting to sort by proportion/absolute cells
		_, p_cells = bt._get_extra_cells([parent], original_counters[idx])
		total_cells = sum(cells) + p_cells[0]
		parent_totals.append(total_cells)
		if not prop_all:
			list_cells[idx] = list(map(lambda x: (x / total_cells)*100, cells))
		else:
			list_cells[idx] = list(map(lambda x: (x / datasets[idx].presynaptics())*100, cells))
	_, axis_title = bt._cells_in_areas_in_datasets(children, datasets, channel, normalisation='presynaptics')

	cells_sort_by = [sum(x) for x in zip(*list_cells)] # sum each area for each dataset
	cells_sort_by, area_labels, *list_cells = zip(*sorted(zip(cells_sort_by, area_labels, *list_cells), reverse=True))
	list_cells = [list(i) for i in list_cells]

	for idx, counter in enumerate(original_counters): # add any extra cells that were assigned to the parent area
		p_name, p_cells = bt._get_extra_cells([parent], counter)
		if not prop_all:
			p_cells = list(map(lambda x: (x / parent_totals[idx])*100, p_cells))
		else:
			p_cells = list(map(lambda x: (x / bt.datasets[idx].presynaptics())*100, p_cells))
		list_cells[idx] = list_cells[idx] + p_cells
	area_labels = area_labels + tuple(['Rest of ' + p_name[0]])

	list_cells_2d = np.array(list_cells) # exclude brain areas where the average of all datasets is less than threshold
	thresh = np.repeat(threshold, len(list_cells_2d[0]))
	summed = np.sum(list_cells_2d, axis=0)
	averaged = summed / len(list_cells_2d)
	idxs_to_remove = np.where(averaged < thresh)[0]
	for idx, cells in enumerate(list_cells):
		list_cells[idx] = [v for i, v in enumerate(cells) if i not in idxs_to_remove]
	if bt.debug:
		names_removed = [v for i, v in enumerate(area_labels) if i in idxs_to_remove]
		string = ', '.join(names_removed)
		print(f'Areas excluded: {names_removed}')
	area_labels = [v for i, v in enumerate(area_labels) if i not in idxs_to_remove]
	
	prop_title = 'presynaptic' if prop_all else p_name[0]
	axis_title = f'% {prop_title} cells'
	helpers.draw_plot(ax, datasets, area_labels, list_cells, axis_title, fig_title=f'{parent_name}', horizontal=False, b_space=0.3)

def generate_heatmap(channel, dataset, orientation='sagittal', vmax=None, position=None, normalisation='total', cmap='Reds', legend=True, region_labels=True):
	areas, areas_title = bta.summary_regions()
	values, cbar_label = bt._cells_in_areas_in_datasets(areas, [dataset], channel, normalisation=normalisation)
	regions = dict(zip(areas, np.array(values[0]).T))
	f = bgh.Heatmap(regions, position=position, orientation=orientation, title=f'{areas_title}: {dataset.name}', thickness=1000, atlas_name=bt.atlas.atlas_name,format='2D', vmin=0, vmax=vmax, cmap=cm.get_cmap(cmap), annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)

def generate_heatmap_comparison(channel, fluorescence, areas, orientation, vmax=None, position=None, normalisation='total', legend=True, region_labels=True):
	# orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
	datasets, num_g1 = helpers.fetch_groups(fluorescence)
	dataset_cells, cbar_label = bt._cells_in_areas_in_datasets(areas, datasets, channel, normalisation=normalisation)
	dataset_cells = np.array(dataset_cells)
	mean_g1 = np.mean(dataset_cells[0:num_g1,:], axis=0)
	mean_g2 = np.mean(dataset_cells[num_g1:,:], axis=0)
	if vmax is None:
		vmax = np.max(np.concatenate([[mean_g1, mean_g2]], axis=1)) # get max value in the two arrays to set the same vmax for both plots
	g1_regions = dict(zip(areas, mean_g1))
	g2_regions = dict(zip(areas, mean_g2))
	f1 = bgh.Heatmap(g1_regions, position=position, orientation=orientation, title=helpers.get_bt_groups()[0], thickness=1000, atlas_name=bt.atlas.atlas_name, format='2D', vmin=0, vmax=vmax, cmap=helpers.cmaps_group[1], annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)
	plt.figure(f1)
	areas_savename = ''.join([string[0] for string in areas])
	btf.save(f'heatmap_means_areas={areas_savename}_ch={channel}_Fl={fluorescence}_o={orientation}_group={helpers.get_bt_groups()[0]}', as_type='pdf')
	plt.close()
	f2 = bgh.Heatmap(g2_regions, position=position, orientation=orientation, title=helpers.get_bt_groups()[1], thickness=1000, atlas_name=bt.atlas.atlas_name, format='2D', vmin=0, vmax=vmax, cmap=helpers.cmaps_group[2], annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)
	plt.figure(f2)
	btf.save(f'heatmap_means_areas={areas_savename}_ch={channel}_Fl={fluorescence}_o={orientation}_group={helpers.get_bt_groups()[1]}', as_type='pdf')
	plt.close()

def heatmap_SI(channel, fluorescence, areas, orientation, vlim=None, position=None, normalisation='total', cmap='bwr', legend=True, region_labels=True):
	# orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
	datasets, num_g1 = helpers.fetch_groups(fluorescence)
	dataset_cells, axis_title = bt._cells_in_areas_in_datasets(areas, datasets, channel, normalisation=normalisation)
	dataset_cells = np.array(dataset_cells)
	LS, LV = dataset_cells[0:num_g1,:], dataset_cells[num_g1:,:]
	avgs, _ = helpers.calculate_SI_from_means(LV, LS)
	bounds = np.abs(avgs).max()
	if vlim is not None:
		bounds = np.abs(vlim)
	regions = dict(zip(areas, avgs))
	cbar_label = f'LS - LV ({axis_title})'
	f = bgh.Heatmap(regions, position=position, orientation=orientation, title=f'SI for Fl={fluorescence}', thickness=1000, atlas_name=bt.atlas.atlas_name, format='2D', vmin=-bounds, vmax=bounds, cmap=cm.get_cmap(cmap), annotate_regions=region_labels).show(show_legend=legend, cbar_label=cbar_label)
	plt.figure(f)
	areas_savename = ''.join([string[0] for string in areas])
	btf.save(f'heatmap_SI_areas={areas_savename}_ch={channel}_Fl={fluorescence}_o={orientation}', as_type='pdf')
	plt.close()

def matrix_plot(channel, fluorescence, area_func, value_norm='custom_pedestal', cmap='Reds', vbounds=(None, None)):
	area_idxs, areas_title = area_func
	area_labels = np.array(bt.get_area_info(area_idxs)[0])
	datasets = [d for d in bt.datasets if d.fluorescence == fluorescence]
	dataset_cells, _ = bt._cells_in_areas_in_datasets(area_labels, datasets, channel, normalisation=value_norm)

	filter_cells = np.mean(np.array(dataset_cells), axis=0)
	new_idxs = np.array(area_idxs)[filter_cells > 0.1] # 30 cells

	area_labels = np.array(bt.get_area_info(new_idxs)[0])
	dataset_cells, axis_title = bt._cells_in_areas_in_datasets(area_labels, datasets, channel, normalisation=value_norm)

	f, ax = plt.subplots()
	f.set_facecolor('white')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)

	if vbounds is None:
		im = ax.matshow(dataset_cells, aspect='equal', cmap=cmap)
	else:
		im = ax.matshow(dataset_cells, aspect='equal', cmap=cmap, vmin=vbounds[0], vmax=vbounds[1])

	y_labels = [d.name for d in datasets]
	ax.set_yticks(range(len(y_labels)))
	ax.set_yticklabels(y_labels)
	ax.set_xticks(range(len(area_labels)))
	ax.set_xlabel(areas_title)
	ax.set_xticklabels(area_labels, rotation=135, ha='left')
	ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
	colours = helpers.colours_from_labels(y_labels)
	[t.set_color(colours[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
	f.colorbar(im, cax=cax, orientation='vertical')
	ax.set_title(axis_title)

def generate_matrix_plot(channel, depth=None, areas=None, threshold=10, sort=True, ignore=None, override_order=None, vbounds=None, normalisation='presynaptics', covmat=False, rowvar=1, zscore=True, div=False, order_method=None, cmap='bwr', figsize=(35,6), aspect='equal'):
	if areas is None and depth is not None:
		area_idxs = bt.children_from('root', depth=depth)[1]
	if areas is None and depth is None:
		area_idxs = bt.children_from('root', depth=0)[1] # get all children and remove the ignored regions
		for i in ignore:
			try:
				area_idxs.remove(i)
			except (ValueError, TypeError):
				print(f'Warning: Could not remove area index {i}')
	if areas is not None and depth is None:
		area_idxs = areas
	
	# to order by group
	groups = helpers.get_bt_groups()
	datasets1 = [i for i in bt.datasets if i.group == groups[0]]
	datasets2 = [i for i in bt.datasets if i.group == groups[1]]
	datasets = datasets1 + datasets2
	y_labels = [i.name for i in datasets]

	if override_order is not None:
		datasets = list(np.array(datasets)[override_order]) # sort both axes of the matrix by the computed order
		y_labels = list(np.array(y_labels)[override_order])

	area_labels = np.array(bt.get_area_info(area_idxs)[0])
	dataset_cells, axis_title = bt._cells_in_areas_in_datasets(area_labels, datasets, channel, normalisation=normalisation)
	dataset_cells = np.array(dataset_cells)

	if areas is None:
		summed = np.sum(dataset_cells, axis=0) # delete columns where mean is below threshold
		averaged = summed / dataset_cells.shape[0]
		idxs_to_remove = np.where(averaged <= threshold)[0]
		dataset_cells = np.delete(dataset_cells, idxs_to_remove, axis=1)
		area_labels = np.delete(area_labels, idxs_to_remove)
		area_idxs = np.delete(area_idxs, idxs_to_remove) # also remove idxs so sort can work
	if bt.debug:
		print(f'All areas: {list(area_idxs)}')

	if sort is True:
		#col_sorter = np.sum(dataset_cells, axis=0).argsort()[::-1] # sort columns by sum
		#dataset_cells, area_labels = dataset_cells[:,col_sorter], area_labels[col_sorter]
		new_labels = np.array([])
		new_matrix = []
		sorted_already = np.array([])
		for i in area_idxs:
			if i not in sorted_already:
				children = bt.children_from(i, depth=0)[1]
				present_children = [i for i in area_idxs if i in children and i not in sorted_already]
				if bt.debug:
					print(f'Area {i} contains: {present_children}')
				paired_arr = np.append(i, present_children)
				new_labels = np.append(new_labels, paired_arr)
				sorted_already = np.append(sorted_already, paired_arr)
				for index in paired_arr:
					i = np.where(area_idxs == index)[0]
					new_matrix.append(list(dataset_cells[:,i].flatten()))
		area_labels = bt.get_area_info(new_labels.astype(int))[0]
		dataset_cells = np.array(new_matrix).T

	if div:
		dataset_cells = dataset_cells / 100
	if zscore:
		dataset_cells = stats.zscore(dataset_cells, axis=0)
		
	f, ax = plt.subplots(figsize=figsize)
	f.set_facecolor('white')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	if covmat:
		cov = np.corrcoef(dataset_cells, rowvar=rowvar)
		if order_method is not None:
			cov, res_order, res_linkage = helpers.compute_serial_matrix(cov,order_method)
		if vbounds is None:
			im = ax.matshow(cov, aspect=aspect, cmap=cmap)
		else:
			im = ax.matshow(cov, aspect=aspect, cmap=cmap, vmin=vbounds[0], vmax=vbounds[1])
		if rowvar == 0:
			ax.set_yticks(range(len(area_labels)))
			ax.set_yticklabels(area_labels)
			ax.set_xticks(range(len(area_labels)))
			ax.set_xticklabels(area_labels, rotation=90)
		elif rowvar == 1:
			ax.set_yticks(range(len(y_labels)))
			ax.set_yticklabels(y_labels)
			ax.set_xticks(range(len(y_labels)))
			ax.set_xticklabels(y_labels, rotation=90)
			colours = helpers.colours_from_labels(y_labels)
			[t.set_color(colours[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
			[t.set_color(colours[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
	else:
		if vbounds is None:
			im = ax.matshow(dataset_cells, aspect=aspect, cmap=cmap)
		else:
			im = ax.matshow(dataset_cells, aspect=aspect, cmap=cmap, vmin=vbounds[0], vmax=vbounds[1])
		ax.set_yticks(range(len(y_labels)))
		ax.set_yticklabels(y_labels)
		ax.set_xticks(range(len(area_labels)))
		ax.set_xticklabels(area_labels, rotation=135, ha='left')
		ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
		colours = helpers.colours_from_labels(y_labels)
		[t.set_color(colours[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
	f.colorbar(im, cax=cax, orientation='vertical')
	ax.set_title(axis_title)

# plot starter cell counts, perform correction, then show the updated starter cell counts
def correct_retro_starter_cell_count():
	f, axs = plt.subplots(1,3, figsize=(16,5))
	generate_starter_cell_bar(ax=axs[0])
	axs[0].set_ylim(0, 1000)
	region_comparison_scatter(fluorescence=False, config='prePost', labels=True, set_postsyn_values_to_line=True, exclude_dataset_idx_from_line_fit=[])
	generate_starter_cell_scatter(use_manual_count=True, ax=axs[1]) # this function only executes for datasets with fluorescence = False
	generate_starter_cell_bar(ax=axs[2])
	axs[2].set_ylim(0, 1000)