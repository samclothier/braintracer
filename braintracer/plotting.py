import plotly
import braintracer.file_management as btf
import braintracer.analysis as bt
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import plotly.graph_objs as go
import bgheatmaps as bgh
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from matplotlib import colors
from itertools import chain
from matplotlib import cm
from scipy import signal
from scipy import stats
from vedo import embedWindow # for displaying bg-heatmaps
embedWindow(None)
from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage

g1_clr, g2_clr = (0.902,0,0.494), (0.078,0.439,0.721)

def __draw_plot(ax, datasets, areas, values, axis_title, fig_title, horizontal=False, l_space=None, b_space=None):
	if ax is None:
		f, ax = plt.subplots(figsize=(8,6))
		f.subplots_adjust(left=l_space, bottom=b_space)
		f.set_facecolor('white')
	if bt.grouped:
		groups = [i.group for i in datasets]
		df = _plot_grouped_points(ax, values, groups, areas, axis_title, is_horizontal=horizontal)
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
	return df

def generate_summary_plot(ax=None):
	summary_areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN']
	area_labels, _, _ = bt.get_area_info(summary_areas)
	if bt.fluorescence:
		datasets = [i for i in bt.datasets if i.fluorescence]
		dataset_cells, axis_title = _fluorescence_by_area_across_fl_datasets(summary_areas, datasets, normalisation='ch1')
	else:
		datasets = [i for i in bt.datasets if not i.fluorescence]
		dataset_cells, axis_title = _cells_in_areas_in_datasets(summary_areas, datasets, normalisation='ch1')
		# Implement IO count subtraction from MY here
		#IO_cells = bt.get_area_info('IO', dataset.ch1_cells_by_area)[2][0]
		#cells[5] = cells[5] - IO_cells
	
	assert len(datasets) != 0, f'No datasets exist of type fluorescence={bt.fluorescence}'
	if bt.debug:
		print(dataset_cells)
		percentages = [f'{sum(dataset):.1f}% ' for dataset in dataset_cells]
		print(', '.join(percentages)+'cells are within brain boundaries and in non-tract and non-ventricular areas')
	__draw_plot(ax, datasets, area_labels, dataset_cells, axis_title, fig_title='Whole brain', horizontal=True, l_space=0.2)

def generate_custom_plot(area_names, title, normalisation='presynaptics', flr_log=False, horizontal=True, ax=None):
	area_labels, _, _ = bt.get_area_info(area_names)
	if bt.fluorescence:
		datasets = [i for i in bt.datasets if i.fluorescence]
		dataset_cells, axis_title = _fluorescence_by_area_across_fl_datasets(area_names, datasets, normalisation=normalisation, log=flr_log)
	else:
		datasets = [i for i in bt.datasets if not i.fluorescence]
		dataset_cells, axis_title = _cells_in_areas_in_datasets(area_names, datasets, normalisation=normalisation)
	assert len(datasets) != 0, f'No datasets exist of type fluorescence={bt.fluorescence}'
	__draw_plot(ax, datasets, area_labels, dataset_cells, axis_title, fig_title=title, horizontal=horizontal, l_space=0.35)

def generate_whole_fluorescence_plot(dataset=None, values=None):
	indexes = list(bt.area_indexes.index)
	f, ax = plt.subplots(figsize=(10,100))
	title = 'Propagated Fluorescence Plot'
	generate_custom_plot(indexes, title, flr_log=True, ax=ax)

def generate_matrix_plot(depth=None, areas=None, threshold=10, sort=True, ignore=None, override_order=None, vbounds=None, normalisation='presynaptics', covmat=False, rowvar=1, zscore=True, div=False, order_method=None, cmap='bwr', figsize=(35,6), aspect='equal'):
	if areas is None and depth is not None:
		area_idxs = bt.children_from('root', depth=depth)[1]
	if areas is None and depth is None:
		area_idxs = bt.children_from('root', depth=0)[1] # get all children and remove the ignored regions
		for i in ignore:
			try:
				area_idxs.remove(i)
			except (ValueError, TypeError):
				print(f'Warning: Could not remove area index')
	if areas is not None and depth is None:
		area_idxs = areas
	
	group_names = [i.group for i in bt.datasets] # to order by group
	groups = __get_bt_groups()
	datasets1 = [i for i in bt.datasets if i.group == groups[0]]
	datasets2 = [i for i in bt.datasets if i.group == groups[1]]
	datasets = datasets1 + datasets2
	y_labels = [i.name for i in datasets]

	if override_order is not None:
		datasets = list(np.array(datasets)[override_order]) # sort both axes of the matrix by the computed order
		y_labels = list(np.array(y_labels)[override_order])

	area_labels = np.array(bt.get_area_info(area_idxs)[0])
	if bt.fluorescence:
		dataset_cells, axis_title = _fluorescence_by_area_across_fl_datasets(area_labels, datasets, normalisation=normalisation)
	else:
		dataset_cells, axis_title = _cells_in_areas_in_datasets(area_labels, datasets, normalisation=normalisation)
	dataset_cells = np.array(dataset_cells)

	if areas is None:
		summed = np.sum(dataset_cells, axis=0) # delete columns where mean is below threshold
		averaged = summed / dataset_cells.shape[0]
		idxs_to_remove = np.where(averaged <= threshold)[0]
		dataset_cells = np.delete(dataset_cells, idxs_to_remove, axis=1)
		area_labels = np.delete(area_labels, idxs_to_remove)
		area_idxs = np.delete(area_idxs, idxs_to_remove) # also remove idxs so sort can work

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
				print(i, present_children)
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

	assert len(datasets) != 0, f'No datasets exist of type fluorescence={bt.fluorescence}'

	f, ax = plt.subplots(figsize=figsize)
	f.set_facecolor('white')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	if covmat:
		cov = np.corrcoef(dataset_cells, rowvar=rowvar)
		if order_method is not None:
			cov, res_order, res_linkage = compute_serial_matrix(cov,order_method)
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
			colours = _colours_from_labels(y_labels)
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
		colours = _colours_from_labels(y_labels)
		[t.set_color(colours[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
	f.colorbar(im, cax=cax, orientation='vertical')
	ax.set_title(axis_title)

def bin_3D_show(area_num=None, binsize=200, axis=2, sigma=None, subregion_depth=None, projcol='k', padding=10, vlim=None, ch1=True):
	atlas_res = 10
	assert binsize % atlas_res == 0, f'Binsize must be a multiple of atlas resolution ({atlas_res}um) to display correctly.'
	x_bins, y_bins, z_bins = get_bins(0, binsize), get_bins(1, binsize), get_bins(2, binsize)
	axes = [x for x in [0, 1, 2] if x != axis]
	assert axis in [0, 1, 2], 'Must provide a valid axis number 0-2.'

	f, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
	parent_projection, (px_min, py_min, pz_min), (px_max, py_max, pz_max) = bt.get_projection(area_num, padding=padding, axis=2-axis)
	parent_projection = parent_projection.T if axis == 0 else parent_projection # side-on orientation does not need axis swapping
	if subregion_depth is not None:
		for child in bt.children_from(area_num, depth=subregion_depth)[1]:
			child_projection, (cx_min, cy_min, cz_min), _ = bt.get_projection(child, padding=padding, axis=2-axis)
			cx_offset, cy_offset, cz_offset = cx_min - px_min, cy_min - py_min, cz_min - pz_min
			if axis == 2:
				child_projection = np.pad(child_projection, ((cy_offset,0),(cx_offset,0)))
			elif axis == 1:
				child_projection = np.pad(child_projection, ((cz_offset,0),(cx_offset,0)))
			else:
				child_projection = np.pad(child_projection, ((cz_offset,0),(cy_offset,0)))
			child_projection = child_projection.T if axis == 0 else child_projection # side-on orientation does not need axis swapping
			ax1.contour(child_projection, colors=projcol, alpha=0.05)
			ax2.contour(child_projection, colors=projcol, alpha=0.05)
	ax1.contour(parent_projection, colors=projcol, alpha=0.1)
	ax2.contour(parent_projection, colors=projcol, alpha=0.1)
	ax1.set_aspect('equal')
	ax2.set_aspect('equal')

	def plot_binned_average(ax, axis, group, cmap):
		hist_list = []
		for d in [d for d in bt.datasets if d.group == group]:
			if area_num is None:
				points = np.array(d.ch1_cells).T
			else:
				parent, children = bt.children_from(area_num, depth=0)
				areas = [parent] + children
				points = np.array(bt._get_cells_in(areas, d, ch1=ch1)).T
			hist, binedges = np.histogramdd(points, bins=(x_bins, y_bins, z_bins), range=((0,1140),(0,800),(0,1320)), normed=False)
			
			if hist.sum() != 0:
				hist = hist / hist.sum() # turn into probability density distribution

			if sigma is not None: # 3D smooth # sigma = width of kernel
				x, y, z = np.arange(-3,4,1), np.arange(-3,4,1), np.arange(-3,4,1) # coordinate arrays -- make sure they include (0,0)!
				xx, yy, zz = np.meshgrid(x,y,z)
				kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
				hist = signal.convolve(hist, kernel, mode='same')

			hist = np.sum(hist, axis=axis) # take the maximum projection of the distribution

			scale = int(binsize / atlas_res) ## make ready for plotting
			hist = hist.repeat(scale, axis=0).repeat(scale, axis=1) # multiply up to the atlas resolution
			at_shp = bt.atlas.shape
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
		av_im = np.median(all_hists, axis=0) # get the mean cell distribution
		av_im = av_im if axis == 0 else av_im.T # side-on orientation does not need axis swapping
		cax = ax.imshow(av_im, cmap=cmap, vmin=0, vmax=vlim)
		plt.colorbar(cax)
		ax.set_title(f'{group} cell detection probability')

	orig_cmap = plt.cm.RdBu
	clrs_blue = orig_cmap(np.linspace(0.5, 1.0, 10))
	clrs_reds = orig_cmap(np.linspace(0.0, 0.5, 10)[::-1])
	cmap_blue = clrs.LinearSegmentedColormap.from_list("mycmap", clrs_blue)
	cmap_reds = clrs.LinearSegmentedColormap.from_list("mycmap", clrs_reds)
	plot_binned_average(ax1, axis, __get_bt_groups()[0], cmap=cmap_reds)
	plot_binned_average(ax2, axis, __get_bt_groups()[1], cmap=cmap_blue)

def bin_3D_matrix(area_num=None, binsize=500, aspect='equal', zscore=False, sigma=None, vbounds=None, threshold=1, override_order=None, order_method=None, blind_order=False, ch1=True, cmap='Reds', covmat=False, figsize=(8,8)):
	x_bins, y_bins, z_bins = get_bins(0, binsize), get_bins(1, binsize), get_bins(2, binsize)

	group_names = [i.group for i in bt.datasets] # to order by group
	groups = __get_bt_groups()
	datasets1 = [i for i in bt.datasets if i.group == groups[0]]
	datasets2 = [i for i in bt.datasets if i.group == groups[1]]
	datasets = datasets1 + datasets2
	y_labels = [i.name for i in datasets]

	voxels = []
	num_nonzero_bins = []
	for i, d in enumerate(datasets):
		if area_num is None:
			points = np.array(d.ch1_cells).T
			old_num_points = points.shape
			points = points[points.min(axis=1)>=0,:] # remove coordinates with negative values so the next step works
			neg_num_points = points.shape
			points_IO = np.array(bt._get_cells_in([83,528], d, ch1=ch1)).T
			dims = np.maximum(points_IO.max(0),points.max(0))+1 # this and following line are to filter out IO points from points
			points = points[~np.in1d(np.ravel_multi_index(points.T,dims),np.ravel_multi_index(points_IO.T,dims))]
			if bt.debug:
				print(f'Num neg points removed: {old_num_points[0] - neg_num_points[0]}, num IO+CBX points removed: {neg_num_points[0] - points.shape[0]}, num acc values: {points_IO.shape[0]}')
		else:
			points = np.array(bt._get_cells_in(area_num, d, ch1=ch1)).T
		hist, binedges = np.histogramdd(points, bins=(x_bins, y_bins, z_bins), range=((0,1140),(0,800),(0,1320)), normed=False)
		num_nonzero_bins.append(np.count_nonzero(hist)) # just debug stuff
		last_hist_shape = hist.shape			
		if sigma is not None: # 3D smooth # sigma = width of kernel
			x, y, z = np.arange(-3,4,1), np.arange(-3,4,1), np.arange(-3,4,1) # coordinate arrays -- make sure they include (0,0)!
			xx, yy, zz = np.meshgrid(x,y,z)
			kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
			hist = signal.convolve(hist, kernel, mode='same')
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
		if override_order is None:
			if order_method is not None:
				if not blind_order:
					d1b, d2b = len(datasets1), len(datasets2)
					print(f'Group 1: {d1b}, Group 2: {d2b}')
					mat1 = cov[:d1b,:d1b]
					mat2 = cov[d1b:,d1b:] # split the matrix up into two sub-matrices
					mat1, res_order1, res_linkage1 = compute_serial_matrix(mat1, order_method)
					mat2, res_order2, res_linkage2 = compute_serial_matrix(mat2, order_method)
					res_order2 = list(np.array(res_order2) + d1b) # offset dataset indexes
					res_order = res_order1 + res_order2 # create final order array
				else:
					mat, res_order, res_linkage = compute_serial_matrix(cov, order_method)
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
	colours = _colours_from_labels(y_labels)
	if covmat:
		[t.set_color(colours[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
	[t.set_color(colours[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]

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

def generate_heatmap_comparison(areas, orientation, position=None, normalisation='ch1', cmap='Reds', legend=True):
	# orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
	group_names = [i.group for i in bt.datasets]
	if bt.fluorescence:
		values, cbar_label = _fluorescence_by_area_across_fl_datasets(areas, bt.datasets, normalisation=normalisation)
	else:
		values, cbar_label = _cells_in_areas_in_datasets(areas, bt.datasets, normalisation=normalisation)
	groups, cells = _compress_into_groups(group_names, values)
	highest_value = np.max(cells)
	g1_regions = dict(zip(areas, cells[0]))
	g2_regions = dict(zip(areas, cells[1]))
	bgh.heatmap(g1_regions, position=position, orientation=orientation, title=groups[0], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)
	bgh.heatmap(g2_regions, position=position, orientation=orientation, title=groups[1], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)

def generate_heatmap_difference(areas, orientation, position=None, normalisation='ch1', cmap='bwr', legend=True, limit=None):
	# orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
	group_names = [i.group for i in bt.datasets]
	if bt.fluorescence:
		values, cbar_label = _fluorescence_by_area_across_fl_datasets(areas, bt.datasets, normalisation=normalisation)
	else:
		values, cbar_label = _cells_in_areas_in_datasets(areas, bt.datasets, normalisation=normalisation)
	groups, cells = _compress_into_groups(group_names, values)
	cells = np.array(cells)
	differences = cells[0] - cells[1]
	bounds = np.abs(differences).max()
	regions = dict(zip(areas, differences))
	cbar_label = 'LS - LV inputs / postsynaptic cell'
	if limit is not None:
		bounds = np.abs(limit)
	bgh.heatmap(regions, position=position, orientation=orientation, thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=-bounds, vmax=bounds, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)

def generate_heatmap_ratios(areas, orientation, position=None, normalisation='ch1', cmap='bwr', legend=True, limit=None):
	# orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
	group_names = [i.group for i in bt.datasets]
	if bt.fluorescence:
		values, cbar_label = _fluorescence_by_area_across_fl_datasets(areas, bt.datasets, normalisation=normalisation)
	else:
		values, cbar_label = _cells_in_areas_in_datasets(areas, bt.datasets, normalisation=normalisation)
	groups, cells = _compress_into_groups(group_names, values)
	cells = np.array(cells)
	print(cells[cells==0])
	ratios = cells[0] / cells[1]
	differences = np.log10(ratios, where=ratios > 0) # calculate log ratio rather than absolute difference
	bounds = np.abs(differences).max()
	regions = dict(zip(areas, differences))
	cbar_label = 'LS - LV inputs / postsynaptic cell'
	if limit is not None:
		bounds = np.abs(limit)
	bgh.heatmap(regions, position=position, orientation=orientation, thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=-bounds, vmax=bounds, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)

def generate_heatmap(dataset, orientation='sagittal', vmax=None, position=None, normalisation='ch1', cmap='Reds', legend=True):
	summary_areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN','IO']
	if dataset.fluorescence:
		values, cbar_label = _fluorescence_by_area_across_fl_datasets(summary_areas, [dataset], normalisation=normalisation)
	else:
		values, cbar_label = _cells_in_areas_in_datasets(summary_areas, [dataset], normalisation=normalisation)
	regions = dict(zip(summary_areas, np.array(values[0]).T))
	bgh.heatmap(regions, position=position, orientation=orientation, title=dataset.name, thickness=1000, atlas_name='allen_mouse_10um',format='2D', vmin=0, vmax=vmax, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)

def generate_slice_heatmap(position, normalisation='ch1', depth=3):
	group_names = [i.group for i in bt.datasets]
	'''
	slice_num = int((position / 14000) * 1320)
	atlas_slice = bt.atlas[slice_num,:,:]
	areas = np.unique(atlas_slice)
	areas = list(np.delete(areas, np.where(areas == 0)).astype(int))
	areas = bt.area_indexes.loc[areas, 'acronym'].tolist()
	areas.remove('root','fiber tracts')
	#print(areas)
	'''
	_, areas = bt.children_from('root', depth=depth)
	areas = bt.area_indexes.loc[areas, 'acronym'].tolist()

	values, cbar_label = _cells_in_areas_in_datasets(areas, bt.datasets, normalisation=normalisation)
	groups, cells = _compress_into_groups(group_names, values)
	print(areas)
	highest_value = np.max(cells) # remove regions in the bottom 1% from plot
	g1_regions = dict(zip(areas, cells[0]))
	g2_regions = dict(zip(areas, cells[1]))
	bgh.heatmap(g1_regions, position=position, orientation='frontal', title=groups[0], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap('hot')).show(show_legend=True, cbar_label=cbar_label)
	bgh.heatmap(g2_regions, position=position, orientation='frontal', title=groups[1], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap('hot')).show(show_legend=True, cbar_label=cbar_label)

def generate_brain_overview(dataset, vmin=5, top_down=False, cmap='gray', ax=None):
	if ax is None:
		f, ax = plt.subplots(figsize=(10,6))
		f.set_facecolor('white')
	stack = np.array(btf.open_file(f'reg_{dataset.name}_{dataset.sig[0]}.tiff'))[0]
	'''
	nz = np.nonzero(stack)
	z_min, y_min, x_min = nz[0].min(), nz[1].min(), nz[2].min()
	z_max, y_max, x_max = nz[0].max(), nz[1].max(), nz[2].max()
	if bt.debug:
		print('x:'+str(x_min)+' '+str(x_max)+' y:'+str(y_min)+' '+str(y_max)+' z:'+str(z_min)+' '+str(z_max))
	#cropped_stack = stack[z_min : z_max, y_min : y_max, x_min : x_max]
	'''
	axis = 1 if top_down else 2
	projection = np.log(stack.max(axis=axis))
	projection = projection.astype(int).T
	im = ax.imshow(projection, vmin=vmin, cmap=cmap)
	f.colorbar(im, label='log max px value along axis')
	plt.axis('off')

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

def generate_projection_plot(area, include_surrounding=False, padding=10, ch1=None, colours=['r','g'], s=2, contour=True, legend=True):
	group1, group2 = __get_bt_groups()
	f, axs = plt.subplots(2, 2, figsize=(12,9), sharex=False)
	f.set_facecolor('white')
	plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
	for dataset in bt.datasets:
		xax = 0 if dataset.group == group1 else 1
		x, y, z = bt._project(axs[0,xax], dataset, area, padding, ch1, s, contour, colours=colours, all_cells=include_surrounding)
		f.suptitle(f'Cell distribution in {area} where x={x}, y={y}, z={z} across '+'_'.join([i.name for i in bt.datasets]))
		bt._project(axs[1,xax], dataset, area, padding, ch1, s, contour, axis=1, colours=colours, all_cells=include_surrounding)
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

def generate_starter_cell_plot(ax=None, true_only=False, log=False):
	if ax is None:
		f, ax = plt.subplots(figsize=(8,5))
		f.set_facecolor('white')
	dataset_names = [i.name for i in bt.datasets]
	if true_only:
		ax.set(ylabel=f'Starter cells in {bt.starter_region} (ch1={bt.starter_ch1})')
		starter_cells = [i.num_cells_in(bt.starter_region, ch1=bt.starter_ch1) for i in bt.datasets]
	else:
		ax.set(ylabel=f'Starter cells in {bt.starter_region} (corrected)')
		starter_cells = [i.postsynaptics() for i in bt.datasets] # green cells in starter region
	sns.barplot(x=dataset_names, y=starter_cells, ax=ax)
	if log:
		ax.set_yscale('log')
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

def generate_starter_cell_scatter(exclude_from_fit=[], use_manual_count=False, show_extras=True, ax=None):
	# exclude_from_fit: list of ints referring to indexes of datasets in bt.datasets
	if not use_manual_count:
		for dataset in bt.datasets:
			dataset.true_postsynaptics = None # ensure this is reset for each new calculation
	if ax is None:
		f, ax = plt.subplots(figsize=(6,6))
		f.set_facecolor('white')
	dataset_names = [i.name for i in bt.datasets]
	if not bt.fluorescence:
		postsynaptics = [i.postsynaptics() for i in bt.datasets] # green cells in starter region
		presynaptics = np.array([i.presynaptics() for i in bt.datasets])
	else:
		postsynaptics = [i.postsynaptics() for i in bt.datasets] # CB brightness
		presynaptics = np.array([i.presynaptics() for i in bt.datasets]) # non CB brightness
	assert len(postsynaptics) == len(presynaptics), 'Starter cells and total cells must be fetchable for all datasets.'

	ax.scatter(presynaptics, postsynaptics, c=_colours_from_labels(dataset_names))

	postsynaptics_to_fit = [ele for idx, ele in enumerate(postsynaptics) if idx not in exclude_from_fit]
	presynaptics_to_fit = [ele for idx, ele in enumerate(presynaptics) if idx not in exclude_from_fit]
	postsynaptics_to_fit = np.pad(postsynaptics_to_fit, [(0,100)])
	presynaptics_to_fit = np.pad(presynaptics_to_fit, [(0,100)]) # pad with (0,0) values to force fit through origin
	z = np.polyfit(presynaptics_to_fit, postsynaptics_to_fit, 1)
	p = np.poly1d(z)
	line_X = np.pad(presynaptics, [(1,0)])
	ax.plot(line_X, p(line_X), 'k')
	for idx, cells in enumerate(presynaptics):
		bt.datasets[idx].true_postsynaptics = p(cells) # set true presynaptics value
	if show_extras:
		ax.scatter(presynaptics, p(presynaptics), c='gray')
	print(f'{presynaptics[0]/p(presynaptics[0])} inputs per {bt.starter_region} neuron.')

	for i, name in enumerate(dataset_names):
		text = ax.annotate(name, (presynaptics[i], postsynaptics[i]), xytext=(8,3), textcoords='offset points')
	ax.set_ylabel(f'Postsynaptic starter cells (ch1={bt.starter_ch1})')
	ax.set_xlabel('Presynaptic inputs (red cells excluding IO + CBX)')
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
			try:
				cells = list(map(lambda x: x / dataset.postsynaptics(), cells))
			except ZeroDivisionError:
				print('Dividing by zero cells. Ensure postsynaptic correction has been applied.')
		else:
			axis_title = '# cells'
		cells_list.append(cells)
	return cells_list, axis_title

def _fluorescence_by_area_across_fl_datasets(areas, datasets, normalisation='presynaptics', log=False):
	normalised_area_fluorescence_fl_datasets = []
	for dataset in datasets:
		area_values = dataset.get_tot_fluorescence(areas)
		def do_norm_to_none(area_values):
			axis_title = '# pixels'
			if log:
				area_values = list(map(lambda x: np.log(x), area_values))
				axis_title = 'log(# pixels)'
			return axis_title, area_values
		if normalisation == None:
			axis_title, area_values = do_norm_to_none(area_values)
		elif normalisation == 'ch1':
			axis_title = '% pixels'
			area_values = list(map(lambda x: (x / dataset.get_tot_fluorescence(['root'])[0]) * 100, area_values))
		else:
			print(f'Normalisation set to {normalisation}, defaulting to None')
			axis_title, area_values = do_norm_to_none(area_values)
		normalised_area_fluorescence_fl_datasets.append(area_values)
	return normalised_area_fluorescence_fl_datasets, axis_title

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
	groups = __get_bt_groups()
	group_counter = Counter(group_names)
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
		sns.barplot(x=titles[2], y=titles[0], hue=titles[1], hue_order=dataset_names, palette=[g1_clr, g2_clr], data=df, ax=ax)
		sns.stripplot(x=titles[2], y=titles[0], hue=titles[1], hue_order=dataset_names, palette=[g1_clr, g2_clr], dodge=True, edgecolor='w', linewidth=0.5, data=points_df, ax=ax)
	else:
		sns.barplot(x=titles[0], y=titles[2], hue=titles[1], hue_order=dataset_names, palette=[g1_clr, g2_clr], data=df, ax=ax)
		sns.stripplot(x=titles[0], y=titles[2], hue=titles[1], hue_order=dataset_names, palette=[g1_clr, g2_clr], dodge=True, edgecolor='w', linewidth=0.5, data=points_df, ax=ax)
	_display_legend_subset(ax, (2,3))
	return df

def _display_legend_subset(ax, idx_tup):
	handles, labels = ax.get_legend_handles_labels()
	ax.legend([handle for i,handle in enumerate(handles) if i in idx_tup],
				[label for i,label in enumerate(labels) if i in idx_tup])

def __get_bt_groups():
	groups = list(dict.fromkeys([dataset.group for dataset in bt.datasets])) # get unique values
	assert len(groups) == 2, 'Comparison plots can only be generated for two dataset groups.'
	return groups

def _colours_from_labels(names):
	datasets = [next((d for d in bt.datasets if d.name==i), None) for i in names] # get datasets by ordered labels
	groups = [i.group for i in datasets] # get their groups and return list of row/column label colours
	return list(map(lambda x: g1_clr if x == __get_bt_groups()[0] else g2_clr, groups)) # list of colours

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
		num_slices = len(bt.atlas)
	elif dim == 1: # y 800
		num_slices = bt.atlas[0].shape[0]
	elif dim == 0: # x 1140
		num_slices = bt.atlas[0].shape[1]
	bin_size = int(size / atlas_res)
	return len([i for i in range(0, num_slices + bin_size, bin_size)]) # return num bins