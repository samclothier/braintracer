import braintracer.analysis as bt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from collections import Counter
import plotly
import plotly.graph_objs as go

def generate_summary_plot(ax, dataset_cells, grouped):
	dataset_names = [i.name for i in bt.datasets]
	group_names = [i.group for i in bt.datasets]
	area_names=bt.summary_names
	if grouped:
		_plot_grouped_points(ax, dataset_cells, group_names, area_names, is_horizontal=True)
	else:
		names, datasets, cells = _prep_for_sns(area_names, dataset_names, dataset_cells)
		column_titles = ['Area', 'Dataset', 'Proportion of cells / %']
		df = pd.DataFrame(zip(names, datasets, cells), columns=column_titles)
		sns.barplot(x=column_titles[2], y=column_titles[0], hue=column_titles[1], data=df, ax=ax)
	ax.grid(axis='x')
	ax.set_title(f'Whole brain')

def generate_custom_plot(ax, dataset_cells, area_names, title, grouped):
	dataset_names = [i.name for i in bt.datasets]
	group_names = [i.group for i in bt.datasets]
	if grouped:
		_plot_grouped_points(ax, dataset_cells, group_names, area_names, is_horizontal=True)
	else:
		names, datasets, cells = _prep_for_sns(area_names, dataset_names, dataset_cells)
		column_titles = ['Area', 'Dataset', 'Proportion of cells / %']
		df = pd.DataFrame(zip(names, datasets, cells), columns=column_titles)
		sns.barplot(x=column_titles[2], y=column_titles[0], hue=column_titles[1], data=df, ax=ax)
	ax.grid(axis='x')
	ax.set_title(f'{title}')

def generate_zoom_plot(ax, parent_name, grouped, depth=2, threshold=1, prop_all=True):
	'''
	prop_all: True; cell counts as fraction of total cells in signal channel. False; cell counts as fraction in parent area
	'''
	dataset_names = [i.name for i in bt.datasets]
	group_names = [i.group for i in bt.datasets]
	new_counters = [i.ch1_cells_by_area for i in bt.datasets]
	original_counters = [i.raw_ch1_cells_by_area for i in bt.datasets]

	parent, children = bt.children_from(parent_name, depth)

	list_cells = [] # 2D array of number of cells in each child area for each dataset
	for counter in new_counters:
		try:
			names, _, cells = bt.get_area_info(children, counter)
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
			list_cells[idx] = list(map(lambda x: (x / bt.datasets[idx].num_cells(ch1=True))*100, cells))

	cells_sort_by = [sum(x) for x in zip(*list_cells)] # sum each area for each dataset
	cells_sort_by, names, *list_cells = zip(*sorted(zip(cells_sort_by, names, *list_cells), reverse=True))
	list_cells = [list(i) for i in list_cells]

	for idx, counter in enumerate(original_counters): # add any extra cells that were assigned to the parent area
		p_name, p_cells = bt._get_extra_cells([parent], counter)
		if not prop_all:
			p_cells = list(map(lambda x: (x / parent_totals[idx])*100, p_cells))
		else:
			p_cells = list(map(lambda x: (x / bt.datasets[idx].num_cells(ch1=True))*100, p_cells))
		list_cells[idx] = list_cells[idx] + p_cells
	names = names + tuple(['Rest of ' + p_name[0]])

	list_cells_2d = np.array(list_cells) # exclude brain areas where the average of all datasets is less than threshold
	thresh = np.repeat(threshold, len(list_cells_2d[0]))
	summed = np.sum(list_cells_2d, axis=0)
	averaged = summed / len(list_cells_2d)
	idxs_to_remove = np.where(averaged < thresh)[0]
	for idx, cells in enumerate(list_cells):
		list_cells[idx] = [v for i, v in enumerate(cells) if i not in idxs_to_remove]
	if bt.debug:
		names_removed = [v for i, v in enumerate(names) if i in idxs_to_remove]
		string = ', '.join(names_removed)
		print(f'Areas excluded: {names_removed}')
	names = [v for i, v in enumerate(names) if i not in idxs_to_remove]
	
	prop_title = 'all' if prop_all else p_name[0]
	if grouped:
		_plot_grouped_points(ax, list_cells, group_names, names, parent_name=prop_title)
	else:
		names, list_datasets, list_cells = _prep_for_sns(names, dataset_names, list_cells)
		column_titles = ['Area', 'Dataset', f'Proportion of {prop_title} cells / %']
		df = pd.DataFrame(zip(names, list_datasets, list_cells), columns=column_titles)
		if not df.empty:
			sns.barplot(x=column_titles[0], y=column_titles[2], hue=column_titles[1], data=df, ax=ax)
	ax.set(xlabel=None)
	ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
	ax.grid(axis='y')
	ax.set_title(f'{parent_name}')

def generate_projection_plot(area, s=2, contour=True):
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
	f.set_facecolor('lightgrey')
	f.suptitle('Cell distribution in '+area+' across '+'_'.join([i.name for i in bt.datasets]))
	for dataset in bt.datasets:
		ax = ax1 if dataset.group == bt.datasets[0].group else ax2
		bt.project_dataset(ax, dataset, area, s, contour)
	ax1.invert_xaxis()
	ax1.invert_yaxis()
	ax2.invert_xaxis()
	ax2.invert_yaxis()
	_display_legend_subset(ax1, (0,))
	_display_legend_subset(ax2, (0,))

def make_areas_3D(areas, colours):
	assert len(areas) == len(colours), 'Each area should have a corresponding colour.'
	atlas = np.array(bt.atlas)
	area_nums = bt.get_area_info(areas)[1]
	def _subsample_atlas_pixels(x, y, z): # reduce pixel density 20x
		x = [val for i, val in enumerate(x) if i % 20 == 0]
		y = [val for i, val in enumerate(y) if i % 20 == 0]
		z = [val for i, val in enumerate(z) if i % 20 == 0]
		return x, y, z
	data = []
	for idx, area_num in enumerate(area_nums):
		z_vals, y_vals, x_vals = np.nonzero(atlas == area_num)
		x_vals, y_vals, z_vals = _subsample_atlas_pixels(x_vals, y_vals, z_vals)
		trace = go.Scatter3d(x = y_vals, y = x_vals, z = z_vals, mode='markers',
		marker={'size': 1, 'opacity': 0.8, 'color':colours[idx]})
		data.append(trace)
	plotly.offline.init_notebook_mode()
	layout = go.Layout(margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
	plot_figure = go.Figure(data=data, layout=layout)
	plotly.offline.iplot(plot_figure)

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
	groups = list(set(group_names))
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

def _plot_grouped_points(ax, dataset_cells, group_names, area_names, is_horizontal=False, parent_name='all'):
	pre_compressed_dataset_cells = dataset_cells
	dataset_names, dataset_cells = _compress_into_groups(group_names, dataset_cells)
	names, datasets, cells = _prep_for_sns(area_names, dataset_names, dataset_cells)
	titles = ['Area', 'Dataset', f'Proportion of {parent_name} cells / %']
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