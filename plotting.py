import braintracer.file_management as btf
import braintracer.analysis as bt
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import bgheatmaps as bgh
import seaborn as sns
import pandas as pd
import numpy as np
import plotly
from collections import Counter
from itertools import chain
from matplotlib import cm
from vedo import embedWindow # for displaying bg-heatmaps
embedWindow(None)

def __draw_plot(ax, datasets, areas, values, axis_title, fig_title, horizontal=False, l_space=None, b_space=None):
	if ax is None:
		f, ax = plt.subplots(figsize=(8,6))
		f.subplots_adjust(left=l_space, bottom=b_space)
		f.set_facecolor('white')
	if bt.grouped:
		groups = [i.group for i in datasets]
		_plot_grouped_points(ax, values, groups, areas, axis_title, is_horizontal=horizontal)
	else:
		dataset_names = [i.name for i in datasets]
		a_names, d_names, values = _prep_for_sns(areas, dataset_names, values)
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

def generate_summary_plot(ax=None):
	summary_areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN']
	area_labels, _, _ = bt.get_area_info(summary_areas)
	if bt.fluorescence:
		datasets = [i for i in bt.datasets if i.fluorescence]
		dataset_cells = _fluorescence_by_area_across_fl_datasets(summary_areas)
		axis_title = 'Average normalised fluorescence'
	else:
		datasets = [i for i in bt.datasets if not i.fluorescence]
		dataset_cells, axis_title = _cells_in_areas_in_datasets(summary_areas, datasets, normalisation='ch1')
	
	assert len(datasets) != 0, f'No datasets exist of type fluorescence={bt.fluorescence}'
	if bt.debug:
		print(dataset_cells)
		percentages = [f'{sum(dataset):.1f}% ' for dataset in dataset_cells]
		print(', '.join(percentages)+'cells are within brain boundaries and in non-tract and non-ventricular areas')
	__draw_plot(ax, datasets, area_labels, dataset_cells, axis_title, fig_title='Whole brain', horizontal=True, l_space=0.2)

def generate_custom_plot(area_names, title, normalisation='presynaptics', ax=None):
	area_labels, _, _ = bt.get_area_info(area_names)
	if bt.fluorescence:
		datasets = [i for i in bt.datasets if i.fluorescence]
		dataset_cells = _fluorescence_by_area_across_fl_datasets(area_names)
		axis_title = 'Average normalised fluorescence'
	else:
		datasets = [i for i in bt.datasets if not i.fluorescence]
		dataset_cells, axis_title = _cells_in_areas_in_datasets(area_names, datasets, normalisation=normalisation)
	assert len(datasets) != 0, f'No datasets exist of type fluorescence={bt.fluorescence}'
	__draw_plot(ax, datasets, area_labels, dataset_cells, axis_title, fig_title=title, horizontal=True, l_space=0.35)

def generate_whole_fluorescence_plot(dataset=None, values=None):
	indexes = list(bt.area_indexes.index)
	f, ax = plt.subplots(figsize=(10,100))
	title = 'Propagated Fluorescence Plot'
	generate_custom_plot(indexes, title, ax)

def generate_zoom_plot(parent_name, depth=2, threshold=1, prop_all=True, ax=None):
	'''
	prop_all: True; cell counts as fraction of total cells in signal channel. False; cell counts as fraction in parent area
	'''
	datasets = bt.datasets
	new_counters = [i.ch1_cells_by_area for i in datasets]
	original_counters = [i.raw_ch1_cells_by_area for i in datasets]

	parent, children = bt.children_from(parent_name, depth)
	list_cells, axis_title = _cells_in_areas_in_datasets(children, datasets, normalisation='presynaptics')
	### rewrite needed for next 3 paras

	list_cells = [] # 2D array of number of cells in each child area for each dataset
	for counter in new_counters:
		try:
			area_labels, _, cells = bt.get_area_info(children, counter)
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
	_, axis_title = _cells_in_areas_in_datasets(children, datasets, normalisation='presynaptics')

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
	__draw_plot(ax, datasets, area_labels, list_cells, axis_title, fig_title=f'{parent_name}', horizontal=False, b_space=0.3)

def generate_heatmap_comparison(areas, orientation='sagittal', position=None, normalisation='ch1'):
	# orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
	if bt.fluorescence:
		return
	group_names = [i.group for i in bt.datasets]
	values, axis_title = _cells_in_areas_in_datasets(areas, bt.datasets, normalisation=normalisation)
	groups, cells = _compress_into_groups(group_names, values)
	highest_value = np.max(cells)
	g1_regions = dict(zip(areas, cells[0]))
	g2_regions = dict(zip(areas, cells[1]))
	bgh.heatmap(g1_regions, position=position, orientation=orientation, title=groups[0], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap('viridis')).show()
	bgh.heatmap(g2_regions, position=position, orientation=orientation, title=groups[1], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap('viridis')).show()

def generate_heatmap(dataset, orientation='sagittal', normalisation='ch1'):
	summary_areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN','IO']
	if dataset.fluorescence:
		return
	else:
		values, axis_title = _cells_in_areas_in_datasets(summary_areas, [dataset], normalisation=normalisation)
		values = np.array(values)[0]/100
	regions = dict(zip(summary_areas, values.T))
	bgh.heatmap(regions, position=None, orientation=orientation, title=dataset.name, thickness=1000, atlas_name='allen_mouse_10um',format='2D',).show()

def generate_mega_overview_figure(title):
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
	io_cells = [bt.get_area_info(['IO'], i.ch1_cells_by_area)[-1] for i in bt.datasets]
	io_cells = io_cells[0] if len(io_cells) == 1 else chain.from_iterable(io_cells)
	io_titles = ['Injection site', 'Cells in inferior olive']
	io_df = pd.DataFrame(zip(names, io_cells), columns=io_titles)
	if bt.grouped:
		sns.barplot(x=io_titles[0], y=io_titles[1], order=['LV','LS'], ax=ax_io, data=io_df, ci=None)
		sns.stripplot(x=io_titles[0], y=io_titles[1], order=['LV','LS'], dodge=True, edgecolor='w', linewidth=0.5, ax=ax_io, data=io_df)
	else:
		sns.barplot(x=io_titles[0], y=io_titles[1], ax=ax_io, data=io_df, ci=None)

	# summary and zoom plots for each area
	generate_summary_plot(ax1)
	summary_areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN']
	summary_areas, _, _ = bt.get_area_info(summary_areas)
	for idx, ax in enumerate(axes[:-1]):
		generate_zoom_plot(summary_areas[idx], threshold=0.1, ax=ax)
	generate_zoom_plot(summary_areas[-1], depth=1, threshold=0, ax=ax10)

def generate_projection_plot(area, include_surrounding=False, padding=10, ch1=None, s=2, contour=True):
	group1, group2 = __get_bt_groups()
	f, axs = plt.subplots(2, 2, figsize=(12,9), sharex=True)
	f.set_facecolor('white')
	plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
	for dataset in bt.datasets:
		xax = 0 if dataset.group == group1 else 1
		x, y, z = bt._project(axs[0,xax], dataset, area, padding, ch1, s, contour, all_cells=include_surrounding)
		f.suptitle(f'Cell distribution in {area} where x={x}, y={y}, z={z} across '+'_'.join([i.name for i in bt.datasets]))
		bt._project(axs[1,xax], dataset, area, padding, ch1, s, contour, axis=1, all_cells=include_surrounding)
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
	_display_legend_subset(axs[0,0], (0,1,))
	_display_legend_subset(axs[0,1], (0,1,))
	
def _generate_starter_validation_plot(padding=10, ch1=None, s=2, contour=True):
	area = bt.starter_region
	if area is None:
		print('Starter region unknown. Define it with bt.starter_region = \'IO\'')
		return
	for dataset in bt.datasets:
		f, axs = plt.subplots(2, 2, figsize=(12,9), sharex=True)
		f.set_facecolor('white')
		plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
		x, y, z = bt._project(axs[0,0], dataset, area, padding, ch1, s, contour)
		f.suptitle(f'Cell distribution in {dataset.name} {area} where x={x}, y={y}, z={z}')
		bt._project(axs[0,1], dataset, area, padding, ch1, s, contour, all_cells=True)
		bt._project(axs[1,0], dataset, area, padding, ch1, s, contour, axis=1)
		bt._project(axs[1,1], dataset, area, padding, ch1, s, contour, axis=1, all_cells=True)
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
		z_vals, y_vals, x_vals = np.nonzero(bt.atlas == area_num)
		x_vals, y_vals, z_vals = _subsample_atlas_pixels(x_vals, y_vals, z_vals)
		trace = go.Scatter3d(x = y_vals, y = x_vals, z = z_vals, mode='markers',
		marker={'size': 1, 'opacity': 0.8, 'color':colours[idx]})
		data.append(trace)
	plotly.offline.init_notebook_mode()
	layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
	plot_figure = go.Figure(data=data, layout=layout)
	plotly.offline.iplot(plot_figure)

def generate_starter_cell_plot(ch1, ax=None):
	if ax is None:
		f, ax = plt.subplots(figsize=(8,5))
		f.set_facecolor('white')
	dataset_names = [i.name for i in bt.datasets]
	starter_cells = [i.num_cells_in(bt.starter_region, ch1=ch1) for i in bt.datasets] # green cells in starter region
	sns.barplot(x=dataset_names, y=starter_cells, ax=ax)
	ax.set_yscale('log')
	ax.set(ylabel=f'Number of starter cells in {bt.starter_region}')

def generate_starter_cell_scatter(ch1, ax=None):
	if ax is None:
		f, ax = plt.subplots(figsize=(8,5))
		f.set_facecolor('white')
	dataset_names = [i.name for i in bt.datasets]
	starter_cells = [i.num_cells_in(bt.starter_region, ch1=ch1) for i in bt.datasets] # green cells in starter region
	presynaptics = np.array([i.presynaptics() for i in bt.datasets])
	assert len(starter_cells) == len(presynaptics), 'Starter cells and total cells must be fetchable for all datasets.'
	ax.scatter(starter_cells, presynaptics)
	for i, name in enumerate(dataset_names):
		text = ax.annotate(name, (starter_cells[i], presynaptics[i]), xytext=(8,3), textcoords='offset points')
	ax.set_xlabel(f'Postsynaptic starter cells (ch1={ch1})')
	ax.set_ylabel('Presynaptic cells (red cells excl. IO + CB)')
	ax.grid()

def _cells_in_areas_in_datasets(areas, datasets, normalisation='presynaptics'):
	cells_list = []
	for dataset in datasets:
		_, _, cells = bt.get_area_info(areas, dataset.ch1_cells_by_area)
		if normalisation == 'ch1': # normalise to total cells in ch1
			axis_title = '% channel 1 cells'
			cells = list(map(lambda x: (x / dataset.num_cells(ch1=True))*100, cells))
		elif normalisation == 'presynaptics': # by default, normalise to the number of presynaptics
			axis_title = '% presynaptic cells'
			cells = list(map(lambda x: (x / dataset.presynaptics())*100, cells))
		elif normalisation == 'postsynaptics': # normalise to inputs per postsynaptic cell
			axis_title = 'Inputs / postsynaptic cell'
			cells = list(map(lambda x: x / dataset.num_cells_in(bt.starter_region, ch1=True), cells))
		cells_list.append(cells)
	return cells_list, axis_title

def _fluorescence_by_area_across_fl_datasets(areas):
	flrs_list = [dataset.get_average_fluorescence(areas) for dataset in bt.datasets if dataset.fluorescence]
	#datasets = [dataset for dataset in bt.datasets if dataset.fluorescence]
	return flrs_list

def _prep_for_sns(area_names, dataset_names, dataset_cells):
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

def _compress_into_groups(group_names, dataset_cells):
	groups = list(dict.fromkeys(group_names)) # get unique values
	group_counter = Counter(group_names)
	group1 = []
	group2 = []
	for idx, group in enumerate(group_names):
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

def _group_points(names, uncompressed_cells, groups):
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

def _plot_grouped_points(ax, dataset_cells, group_names, area_names, axis_title, is_horizontal, parent_name='all'):
	pre_compressed_dataset_cells = dataset_cells
	dataset_names, dataset_cells = _compress_into_groups(group_names, dataset_cells)
	names, datasets, cells = _prep_for_sns(area_names, dataset_names, dataset_cells)
	titles = ['Area', 'Dataset', axis_title]
	df = pd.DataFrame(zip(names, datasets, cells), columns=titles)
	area_name, dataset_name, dataset_cell = _group_points(names, pre_compressed_dataset_cells, group_names)
	points_df = pd.DataFrame(zip(area_name, dataset_name, dataset_cell), columns=titles)
	if is_horizontal:
		sns.barplot(x=titles[2], y=titles[0], hue=titles[1], hue_order=dataset_names, palette=[(0.902,0,0.494),(0.078,0.439,0.721)], data=df, ax=ax)
		sns.stripplot(x=titles[2], y=titles[0], hue=titles[1], hue_order=dataset_names, palette=[(0.902,0,0.494),(0.078,0.439,0.721)], dodge=True, edgecolor='w', linewidth=0.5, data=points_df, ax=ax)
	else:
		sns.barplot(x=titles[0], y=titles[2], hue=titles[1], hue_order=dataset_names, palette=[(0.902,0,0.494),(0.078,0.439,0.721)], data=df, ax=ax)
		sns.stripplot(x=titles[0], y=titles[2], hue=titles[1], hue_order=dataset_names, palette=[(0.902,0,0.494),(0.078,0.439,0.721)], dodge=True, edgecolor='w', linewidth=0.5, data=points_df, ax=ax)
	_display_legend_subset(ax, (2,3))

def _display_legend_subset(ax, idx_tup):
	handles, labels = ax.get_legend_handles_labels()
	ax.legend([handle for i,handle in enumerate(handles) if i in idx_tup],
				[label for i,label in enumerate(labels) if i in idx_tup])

def __get_bt_groups():
	group_names = set([dataset.group for dataset in bt.datasets])
	if len(group_names) != 2:
		print('Comparison plots can only be generated for two dataset groups.')
		return
	left_group = bt.datasets[0].group
	group_names.remove(left_group)
	right_group = list(group_names)[0]
	return left_group, right_group