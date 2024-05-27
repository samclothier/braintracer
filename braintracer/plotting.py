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
bta = importlib.import_module(bt_path+'.area_lists')
bt = importlib.import_module(bt_path+'.analysis')

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
from collections import Counter
from fastcluster import linkage
from itertools import chain
from matplotlib import cm
from scipy import signal
from scipy import stats

# colour helper
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

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
from myterial import white
bgh.heatmaps.settings.ROOT_COLOR = white

colour_LS = '#ED008C'
colour_LV = '#1E74BD'
colblk = [0/255, 0/255, 0/255, 1]
cmap_LS = clrs.LinearSegmentedColormap.from_list('LS', ['#FFFFFF', colour_LS])
cmap_LV = clrs.LinearSegmentedColormap.from_list('LV', ['#FFFFFF', colour_LV])
cmap_LV_to_LS = clrs.LinearSegmentedColormap.from_list('LV to LS', [colour_LV, '#FFFFFF', colour_LS])
LV_LS_mix_colour = (RGB.from_str(colour_LV) + RGB.from_str(colour_LS)) / 2
cmap_LV_LS_mix = clrs.LinearSegmentedColormap.from_list('Global', ['#FFFFFF', str(LV_LS_mix_colour)])
cmap_LV_LS_pmap_cbar = clrs.LinearSegmentedColormap.from_list('LV to LS', [get_angle_for_cmap(0, 1), get_angle_for_cmap(1, 1), get_angle_for_cmap(1, 0)])
csolid_group = [colour_LS, colour_LV]
cmaps_group = [cmap_LS, cmap_LV]



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

def custom_plot(channel, area_names, title='Custom plot', normalisation=None, log=False, horizontal=True, ax=None):
    area_labels, _, _ = bt.get_area_info(area_names)
    datasets = [i for i in bt.datasets]
    dataset_cells, axis_title = bt._cells_in_areas_in_datasets(area_names, datasets, channel, normalisation, log)
    __draw_plot(ax, datasets, area_labels, dataset_cells, axis_title, fig_title=title, horizontal=horizontal, l_space=0.3)
    if bt.debug:
        print(dataset_cells)
        percentages = [f'{sum(dataset):.1f}% ' for dataset in dataset_cells]
        print(', '.join(percentages)+'cells are within brain boundaries and in non-tract and non-ventricular areas')

def summary_plot(channel, log=False, horizontal=True, ax=None):
    summary_areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN']
    custom_plot(channel, summary_areas, title='Whole brain', normalisation='ch1', log=log, horizontal=horizontal, ax=ax)

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
    colours = _colours_from_labels(y_labels)
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
    groups = __get_bt_groups()
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

"""
pmap_params = (channel, fluorescence, area_num, pmap_binsize, axis, sigma)
"""
def pmap_corr_scatter_binned(pmap_params, lower_lim=0.05, binsize=0.005, gradient=0.1, kde=True, vbounds=(None, None)):
    f, ax = plt.subplots(1,1, figsize=(5,5))
    
    ax1_data, ax2_data = probability_map_data(pmap_params[0], pmap_params[1], area_num=pmap_params[2], binsize=pmap_params[3], axis=pmap_params[4], sigma=pmap_params[5], padding=0, no_alignment_to_region_bounds=True)
    ax1_data, ax2_data, max_xy = remove_corner_points(ax1_data, ax2_data, lower_lim)
    
    def my_round(x, prec=2, base=0.05):
        return (base * (np.array(x) / base).round())
    maxlen = max(ax1_data.max(), ax2_data.max())
    maxlen = my_round(maxlen, base=binsize)
    
    if kde:
        nbins = (maxlen / binsize) + 1
        xi, yi = np.mgrid[0:maxlen:nbins*1j, 0:maxlen:nbins*1j] # basically an np.linspace()
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        k = stats.gaussian_kde([ax1_data,ax2_data])
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        xi, yi = xi+binsize/2, yi+binsize/2
        pc = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Greys', vmin=vbounds[0], vmax=vbounds[1])
    else:
        # make a 2d hist
        ax1_bins = np.arange(0, maxlen + binsize, binsize)
        ax2_bins = np.arange(0, maxlen + binsize, binsize)
        #norm = clrs.LogNorm(vmin=vbounds[0], vmax=vbounds[1])
        hist, _, _, pc = ax.hist2d(ax1_data, ax2_data, bins=(ax1_bins, ax2_bins), cmap='Greys', vmin=vbounds[0], vmax=vbounds[1]);
    
    ax.set_xlim(0, max_xy * 1.05) # min was previously: -binsize * 2
    ax.set_ylim(0, max_xy * 1.05)
    ax.axline((0, 0), (max_xy, max_xy), linestyle=(0, (5, 10)), c='orange') # add y=x line
    xaxis = [0, ax.get_xlim()[1]]
    shading1 = [float(i) * gradient for i in xaxis]
    shading2 = [float(i) / gradient for i in xaxis]
    ax.fill_between(xaxis, [0,0], shading1, alpha=.5, linewidth=0, color=csolid_group[0])
    ax.fill_between(xaxis, [np.max(shading2), np.max(shading2)], shading2, alpha=.5, linewidth=0, color=csolid_group[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('LS signal in voxel x')
    ax.set_ylabel('LV signal in voxel x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(pc, cax=cax, orientation='vertical')
    
    index = get_corr_index(ax1_data, ax2_data, gradient)
    ax.text(float(ax.get_xlim()[1]) * 0.9, float(ax.get_ylim()[1]) * 0.75, f'{index:.2g}')
    
    btf.save(f'dmap_corr_scatter_binned_{pmap_params[2]}_fl={pmap_params[1]}', as_type='pdf')

def probability_map(channel, fluorescence, area_num=None, binsize=200, axis=2, sigma=None, subregions=None, subregion_depth=None, projcol='k', padding=10, vmax=None):
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

    groups = __get_bt_groups()
    f, axs = plt.subplots(1, len(groups), figsize=(12,6))

    parent_projection, min_bounds, max_bounds = _get_projection(area_num, padding=padding, axis=axis)
    projections = []
    if regions is not None:
        for child in regions:
            child_projection, (cx_min, cy_min, cz_min), _ = _get_projection(child, padding=padding, axis=axis)
            cx_offset, cy_offset, cz_offset = cx_min - min_bounds[0], cy_min - min_bounds[1], cz_min - min_bounds[2]
            if axis == 2:
                child_projection = np.pad(child_projection, ((cy_offset,0),(cx_offset,0)))
            elif axis == 1:
                child_projection = np.pad(child_projection, ((cz_offset,0),(cx_offset,0)))
            else:
                child_projection = np.pad(child_projection, ((cy_offset,0),(cz_offset,0)))
            projections.append(child_projection)

    def plot_binned_average(ax, channel, area_num, axis, binsize, sigma, group, cmap):
        dmap = get_density_map(channel, area_num, axis, atlas_res, binsize, sigma, group, min_bounds, max_bounds, fluorescence)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
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
        plot_binned_average(ax, channel, area_num, axis, binsize, sigma, g, cmap=cmaps_group[i])

def bin_3D_matrix(channel, area_num=None, binsize=500, aspect='equal', zscore=False, sigma=None, vbounds=None, threshold=1, override_order=None, order_method=None, blind_order=False, cmap='Reds', covmat=False, figsize=(8,8)):
    x_bins, y_bins, z_bins = get_bins(0, binsize), get_bins(1, binsize), get_bins(2, binsize)
    
    groups = __get_bt_groups()
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
            points_IO = np.array(bt._get_cells_in([83,528], d, ch1=ch1)).T
            dims = np.maximum(points_IO.max(0),points.max(0))+1 # this and following line are to filter out IO points from points
            points = points[~np.in1d(np.ravel_multi_index(points.T,dims),np.ravel_multi_index(points_IO.T,dims))]
            if bt.debug:
                print(f'Num neg points removed: {old_num_points[0] - neg_num_points[0]}, num IO+CBX points removed: {neg_num_points[0] - points.shape[0]}, num acc values: {points_IO.shape[0]}')
        else:
            parent, children = bt.children_from(area_num, depth=0)
            areas = [parent] + children
            points = np.array(bt._get_cells_in(areas, d, channel)).T
        hist, _ = np.histogramdd(points, bins=(x_bins, y_bins, z_bins), range=((0,1140),(0,800),(0,1320)), normed=False)
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
                    mat1, res_order1, _ = compute_serial_matrix(mat1, order_method)
                    mat2, res_order2, _ = compute_serial_matrix(mat2, order_method)
                    res_order2 = list(np.array(res_order2) + d1b) # offset dataset indexes
                    res_order = res_order1 + res_order2 # create final order array
                else:
                    _, res_order, _ = compute_serial_matrix(cov, order_method)
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
    __draw_plot(ax, datasets, area_labels, dataset_cells, axis_title, fig_title=title, horizontal=horizontal, l_space=0.3)

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
    __draw_plot(ax, datasets, area_labels, list_cells, axis_title, fig_title=f'{parent_name}', horizontal=False, b_space=0.3)

def generate_heatmap(channel, dataset, orientation='sagittal', vmax=None, position=None, normalisation='total', cmap='Reds', legend=True):
    areas, areas_title = bta.summary_regions()
    values, cbar_label = bt._cells_in_areas_in_datasets(areas, [dataset], channel, normalisation=normalisation)
    regions = dict(zip(areas, np.array(values[0]).T))
    f = bgh.heatmap(regions, position=position, orientation=orientation, title=f'{areas_title}: {dataset.name}', thickness=1000, atlas_name='allen_mouse_10um',format='2D', vmin=0, vmax=vmax, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)

def generate_heatmap_comparison(channel, fluorescence, areas, orientation, vmax=None, position=None, normalisation='total', legend=True):
    # orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
    datasets, num_g1 = fetch_groups(fluorescence)
    dataset_cells, cbar_label = bt._cells_in_areas_in_datasets(areas, datasets, channel, normalisation=normalisation)
    dataset_cells = np.array(dataset_cells)
    mean_g1 = np.mean(dataset_cells[0:num_g1,:], axis=0)
    mean_g2 = np.mean(dataset_cells[num_g1:,:], axis=0)
    if vmax is None:
        vmax = np.max(np.concatenate([[mean_g1, mean_g2]], axis=1)) # get max value in the two arrays to set the same vmax for both plots
    g1_regions = dict(zip(areas, mean_g1))
    g2_regions = dict(zip(areas, mean_g2))
    f1 = bgh.heatmap(g1_regions, position=position, orientation=orientation, title=__get_bt_groups()[0], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=vmax, cmap=cmap_LS).show(show_legend=legend, cbar_label=cbar_label)
    plt.figure(f1)
    areas_savename = ''.join([string[0] for string in areas])
    btf.save(f'heatmap_means_areas={areas_savename}_ch={channel}_Fl={fluorescence}_o={orientation}_group={__get_bt_groups()[0]}', as_type='pdf')
    f2 = bgh.heatmap(g2_regions, position=position, orientation=orientation, title=__get_bt_groups()[1], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=vmax, cmap=cmap_LV).show(show_legend=legend, cbar_label=cbar_label)
    plt.figure(f2)
    btf.save(f'heatmap_means_areas={areas_savename}_ch={channel}_Fl={fluorescence}_o={orientation}_group={__get_bt_groups()[1]}', as_type='pdf')

def heatmap_SI(channel, fluorescence, areas, orientation, vlim=None, position=None, normalisation='total', cmap='bwr', legend=True):
    # orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
    datasets, num_g1 = fetch_groups(fluorescence)
    dataset_cells, axis_title = bt._cells_in_areas_in_datasets(areas, datasets, channel, normalisation=normalisation)
    dataset_cells = np.array(dataset_cells)
    LS, LV = dataset_cells[0:num_g1,:], dataset_cells[num_g1:,:]
    avgs, _ = calculate_SI_from_means(LV, LS)
    bounds = np.abs(avgs).max()
    if vlim is not None:
        bounds = np.abs(vlim)
    regions = dict(zip(areas, avgs))
    cbar_label = f'LS - LV ({axis_title})'
    f = bgh.heatmap(regions, position=position, orientation=orientation, title=f'SI for Fl={fluorescence}', thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=-bounds, vmax=bounds, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)
    plt.figure(f)
    areas_savename = ''.join([string[0] for string in areas])
    btf.save(f'heatmap_SI_areas={areas_savename}_ch={channel}_Fl={fluorescence}_o={orientation}', as_type='pdf')

def heatmap_spatial_segregation(channel, fluorescence, areas, orientation, gradient=0.1, vmax=None, position=None, cmap='Reds', legend=True):
    # orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
    correlations = get_corr_indexes(channel, areas, gradient, fluorescence=fluorescence)
    regions = dict(zip(areas, correlations))
    cbar_label = f'% signal in magenta/blue (gradient={gradient})'
    f = bgh.heatmap(regions, position=position, orientation=orientation, title=f'Within-region spatial segregation for Fl={fluorescence}', thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=vmax, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)
    plt.figure(f)
    areas_savename = ''.join([string[0] for string in areas])
    btf.save(f'heatmap_spatialseg_areas={areas_savename}_ch={channel}_Fl={fluorescence}_o={orientation}', as_type='pdf')

def generate_heatmap_difference(channel, areas, orientation, position=None, normalisation='total', cmap='bwr', legend=True, limit=None):
    # orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
    group_names = [i.group for i in bt.datasets]
    values, cbar_label = bt._cells_in_areas_in_datasets(areas, bt.datasets, channel, normalisation=normalisation)
    groups, cells = _compress_into_groups(group_names, values)
    cells = np.array(cells)
    differences = cells[0] - cells[1]
    bounds = np.abs(differences).max()
    regions = dict(zip(areas, differences))
    cbar_label = 'LS - LV inputs / postsynaptic cell'
    if limit is not None:
        bounds = np.abs(limit)
    bgh.heatmap(regions, position=position, orientation=orientation, thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=-bounds, vmax=bounds, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)

def generate_heatmap_ratios(channel, areas, orientation, position=None, normalisation='total', cmap='bwr', legend=True, limit=None, add=False):
    # orientation: 'frontal', 'sagittal', 'horizontal' or a tuple (x,y,z)
    group_names = [i.group for i in bt.datasets]
    values, cbar_label = bt._cells_in_areas_in_datasets(areas, bt.datasets, channel, normalisation=normalisation)
    groups, cells = _compress_into_groups(group_names, values)
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
    bgh.heatmap(regions, position=position, orientation=orientation, thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=-bounds, vmax=bounds, cmap=cm.get_cmap(cmap)).show(show_legend=legend, cbar_label=cbar_label)
    
def generate_slice_heatmap(channel, position, normalisation='total', depth=3):
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

    values, cbar_label = bt._cells_in_areas_in_datasets(areas, bt.datasets, channel, normalisation=normalisation)
    groups, cells = _compress_into_groups(group_names, values)
    print(areas)
    highest_value = np.max(cells) # remove regions in the bottom 1% from plot
    g1_regions = dict(zip(areas, cells[0]))
    g2_regions = dict(zip(areas, cells[1]))
    bgh.heatmap(g1_regions, position=position, orientation='frontal', title=groups[0], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap('hot')).show(show_legend=True, cbar_label=cbar_label)
    bgh.heatmap(g2_regions, position=position, orientation='frontal', title=groups[1], thickness=1000, atlas_name='allen_mouse_10um', format='2D', vmin=0, vmax=highest_value, cmap=cm.get_cmap('hot')).show(show_legend=True, cbar_label=cbar_label)

def generate_brain_overview(dataset, vmin=None, vmax=None, axis=0, padding=0, cmap='gray', logmax=True, ax=None):
    if ax is None:
        f, ax = plt.subplots(figsize=(10,6))
        f.set_facecolor('white')
    stack = btf.open_registered_stack(dataset)

    plot_projection(ax, 997, padding=padding, axis=axis)
    
    if not logmax:
        data_projection = np.sum(stack, axis=2-axis)
        title = f'#px along axis={axis}'
    else:
        data_projection = np.log(stack.max(axis=2-axis))
        title = f'log max px value along axis={axis}'
    data_projection = data_projection.astype(int)
    if axis == 0:
        data_projection = data_projection.T
    im = ax.imshow(data_projection, vmin=vmin, vmax=vmax, cmap=cmap)
    f.colorbar(im, label=title)
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
    io_cells = [bt.get_area_info(['IO'], i, i.channels[0])[-1] for i in bt.datasets]
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
        x, y, z = _project_with_cells(axs[0,xax], dataset, area, padding, ch1, s, contour, colours=colours, all_cells=include_surrounding)
        f.suptitle(f'Cell distribution in {area} where x={x}, y={y}, z={z} across '+'_'.join([i.name for i in bt.datasets]))
        _project_with_cells(axs[1,xax], dataset, area, padding, ch1, s, contour, axis=1, colours=colours, all_cells=include_surrounding)
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
    area = bt.postsyn_region
    if area is None:
        print('Starter region unknown. Define it with bt.postsyn_region = \'IO\'')
        return
    for dataset in bt.datasets:
        f, axs = plt.subplots(2, 2, figsize=(12,9), sharex=True)
        f.set_facecolor('white')
        plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)
        x, y, z = _project_with_cells(axs[0,0], dataset, area, padding, ch1, s, contour)
        f.suptitle(f'Cell distribution in {dataset.name} {area} where x={x}, y={y}, z={z}')
        _project_with_cells(axs[0,1], dataset, area, padding, ch1, s, contour, all_cells=True)
        _project_with_cells(axs[1,0], dataset, area, padding, ch1, s, contour, axis=1)
        _project_with_cells(axs[1,1], dataset, area, padding, ch1, s, contour, axis=1, all_cells=True)
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

def area_selectivity_scatter(area_func, value_norm='total', custom_lim=None, fluorescence=False, log=False, add_labels=None):
    cm = plt.get_cmap('nipy_spectral')
    area_labels, dataset_cells, std1, std2, datasets, areas_title, axis_title = get_matrix_data(area_func=area_func, postprocess_for_scatter=True, fluorescence=fluorescence, value_norm=value_norm)
    print(area_labels)
    f, ax = plt.subplots(figsize=(6,6))
    x, y = dataset_cells[0], dataset_cells[1]

    ax.errorbar(x, y, xerr=std1, yerr=std2, fmt='o', color='grey', elinewidth=0.2, ms=1.5)
    ax.set_xlabel(f'LS  / mean {axis_title}')
    ax.set_ylabel(f'LV  / mean {axis_title}')
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    ax.set_aspect('equal', adjustable='box') # apply to other scatter plots!
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if not custom_lim:
        max_xy = np.max(dataset_cells) * 10 if log else np.max(dataset_cells) * 1.1
    else:
        max_xy = custom_lim
    min_xy = np.min(dataset_cells) * 0.1 if log else np.min(dataset_cells) * 0.9
    ax.set_ylim(min_xy, max_xy)
    
    ax.set_xlim(min_xy, max_xy)
    ax.axline((0, 0), (max_xy, max_xy), linestyle=(0, (5, 10))) # add y=x line
    
    r, p = stats.pearsonr(dataset_cells[0], dataset_cells[1])
    ax.annotate(f'r = {r:.2f}, p = {p:.2g}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    points_to_plot = 0
    for i, (ix, iy) in enumerate(zip(x, y)): # first cycle through to count number of points to be plotted
        if ix > iy: # if point is below the line
            if ix - std1[i] > iy and iy + std2[i] < ix: # if bounds of error bars are below the line
                points_to_plot += 1
        else: # if point is above the line
            if ix + std1[i] < iy and iy - std2[i] > ix: # if bounds of error bars are above the line
                points_to_plot += 1
                
    ax.set_prop_cycle(color=[cm(1.*i/points_to_plot) for i in range(points_to_plot)]) # set the colours for the extra points
    
    for i, (ix, iy) in enumerate(zip(x, y)): # then actually plot
        if ix > iy: # if point is below the line
            if ix - std1[i] > iy and iy + std2[i] < ix: # if bounds of error bars are below the line
                ax.scatter([ix], [iy], s=30, label=area_labels[i])
        else: # if point is above the line
            if ix + std1[i] < iy and iy - std2[i] > ix: # if bounds of error bars are above the line
                ax.scatter([ix], [iy], s=30, label=area_labels[i])
                
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    btf.save(f'areaSelectivityScatter_{areas_title}_log={log}', as_type='pdf')

def generate_starter_cell_bar(ax=None, true_only=False, log=False):
    if ax is None:
        f, ax = plt.subplots(figsize=(8,5))
        f.set_facecolor('white')
    datasets = [i for i in bt.datasets if not i.fluorescence]
    dataset_names = [i.name for i in datasets]
    if true_only:
        ax.set(ylabel=f'Starter cells in {bt.postsyn_region} (ch={bt.postsyn_ch})')
        starter_cells = [i.num_cells_in(bt.postsyn_region, ch=bt.postsyn_ch) for i in datasets]
    else:
        ax.set(ylabel=f'Starter cells in {bt.postsyn_region} (corrected)')
        starter_cells = [i.postsynaptics() for i in datasets] # green cells in starter region
    sns.barplot(x=dataset_names, y=starter_cells, ax=ax)
    if log:
        ax.set_yscale('log')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
def region_comparison_scatter(fluorescence, config=None, areas=None, labels=False, set_postsyn_values_to_line=False, exclude_dataset_idx_from_line_fit=[]):
    datasets, _ = fetch_groups(fluorescence=fluorescence)
    dataset_names = [i.name for i in datasets]
    
    if config == 'prePost': # do not provide areas for this config
        x_axis = np.array([i.presynaptics() for i in datasets])
        y_axis = np.array([i.postsynaptics() for i in datasets])
        x_label = f'{bt.presyn_ch} - ({bt.postsyn_region} + {bt.presyn_regions_exclude})'
        y_label = f'{bt.postsyn_region}'
        
        if set_postsyn_values_to_line:
            presynaptics_to_fit = [ele for idx, ele in enumerate(x_axis) if idx not in exclude_dataset_idx_from_line_fit]
            postsynaptics_to_fit = [ele for idx, ele in enumerate(y_axis) if idx not in exclude_dataset_idx_from_line_fit]
            presynaptics_to_fit = np.pad(presynaptics_to_fit, [(0,100)]) # pad with (0,0) values to force fit through origin
            postsynaptics_to_fit = np.pad(postsynaptics_to_fit, [(0,100)])
            z = np.polyfit(presynaptics_to_fit, postsynaptics_to_fit, 1)
            p = np.poly1d(z)
            line_X = np.pad(x_axis, [(1,0)])
            ax.plot(line_X, p(line_X), 'k')
            for idx, cells in enumerate(x_axis):
                datasets[idx].true_postsynaptics = p(cells) # set true presynaptics value
            print(f'{x_axis[0]/p(x_axis[0])} inputs per {bt.postsyn_region} neuron.')
    elif config == 'prePostAntero': # special case for anterogradely-labelled datasets. re-add the postsynaptics back because skimmed is set to True
        print('Warning: When running this function, it is assumed that the cells of the postsynaptic region are not included in the dataset.')
        x_axis = np.array([i.presynaptics() + i.postsynaptics() for i in datasets]) 
        y_axis = np.array([i.postsynaptics() for i in datasets])
        x_label = f'{bt.presyn_ch} - {bt.presyn_regions_exclude}'
        y_label = f'{bt.postsyn_region}'
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
    sns.regplot(x=x_axis, y=y_axis, ci=95, robust=True, line_kws=dict(color='gray'), scatter_kws=dict(color=_colours_from_labels(dataset_names)), ax=ax)
    if labels:
        for i, name in enumerate(dataset_names):
            ax.annotate(name, (x_axis[i], y_axis[i]))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, None)
    btf.save(f'regionComparison_c={config}_a={areas}_F={fluorescence}', as_type='pdf')

def area_total_signal_bar(area_func, value_norm='total', fluorescence=False):
    area_labels, dataset_cells, std1, std2, datasets, areas_title, axis_title = get_matrix_data(area_func=area_func, postprocess_for_scatter=False, sort_matrix=False, fluorescence=fluorescence, value_norm=value_norm)
    _, num_g1 = fetch_groups(fluorescence) # get sum across each group
    collapsed_g1 = np.mean(dataset_cells[0:num_g1,:], axis=0)
    collapsed_g2 = np.mean(dataset_cells[num_g1:,:], axis=0)
    
    sort_order = get_sorting_from_SI(area_func, value_norm, fluorescence)
    area_labels = [area_labels[i] for i in sort_order]
    collapsed_g1 = collapsed_g1[sort_order[::1]]
    collapsed_g2 = collapsed_g2[sort_order[::1]]
    std1 = std1[sort_order[::1]]
    std2 = std2[sort_order[::1]]
    
    f, ax = plt.subplots(figsize=(6,6))
    # add offset to error bars
    trans1 = Affine2D().translate(0.0, -0.1) + ax.transData
    trans2 = Affine2D().translate(0.0, +0.1) + ax.transData
    
    groups = __get_bt_groups()
    ax.barh(area_labels, collapsed_g1, xerr=std1, error_kw={'transform':trans1}, label=groups[0], color=csolid_group[0])
    ax.barh(area_labels, collapsed_g2, xerr=std2, error_kw={'transform':trans2}, label=groups[1], color=csolid_group[1], left=collapsed_g1)
    ax.set_ylabel(f'Area')
    ax.set_xlabel(f'norm={value_norm}')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    btf.save(f'areaTotalSignalBar_{areas_title}', as_type='pdf')

def density_map_corr_bar(channel, fluorescence, area_func, gradient=0.1, value_norm='total'):
    area_labels, areas_title = area_func
    area_labels = bt.get_area_info(area_labels)[0]
    correlations = get_corr_indexes(channel, area_labels, gradient, fluorescence=fluorescence)
    
    sort_order = get_sorting_from_SI(area_func, value_norm, fluorescence)
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

def area_selectivity_with_errors(area_func, value_norm='total', fluorescence=False):
    area_labels, dataset_cells, std1, std2, datasets, areas_title, axis_title = get_matrix_data(area_func=area_func, postprocess_for_scatter=False, sort_matrix=False, fluorescence=fluorescence, value_norm=value_norm)
    _, num_g1 = fetch_groups(fluorescence) # get sum across each group
    LS, LV = dataset_cells[0:num_g1,:], dataset_cells[num_g1:,:]
    
    avgs, stds, colours = calculate_SI_with_errors(LS, LV)
    
    sort_order = get_sorting_from_SI(area_func, value_norm, fluorescence)
    area_labels = [area_labels[i] for i in sort_order]
    avgs = [avgs[i] for i in sort_order]
    stds = [stds[i] for i in sort_order]
    colours = [colours[i] for i in sort_order]
    
    f, ax = plt.subplots(figsize=(6,6))
    ax.barh(area_labels, avgs, color=colours, xerr=stds) #, ecolor=error_colours)
    ax.set_ylabel(f'Area')
    ax.set_xlabel(f'% selectivity  ({axis_title})')
    ax.set_xlim(-1,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=0, c='k')
    btf.save(f'areaSelectivity_errors_{areas_title}', as_type='pdf')

def _project_with_cells(ax, dataset, area, padding, s, channels=None, axis=0, all_cells=False):
    '''
    Plot a coronal or horizontal projection of a brain region with cells superimposed.
    '''
    _, (x_min, y_min, z_min), (x_max, y_max, z_max) = plot_projection(ax, area, padding, axis=2)
    def show_cells(ch, colour):
        if all_cells:
            region = (x_min, x_max), (y_min, y_max), (z_min, z_max)
        else:
            parent, children = bt.children_from(area, depth=0)
            areas = [parent] + children
            region = areas
        X_r, Y_r, Z_r = bt._get_cells_in(region, dataset, ch)
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

def plot_projection(ax, area, padding, axis=0, projcol='k', alpha=0.1):
    projection, (x_min, y_min, z_min), (x_max, y_max, z_max) = _get_projection(area, padding=padding, axis=axis)
    if bt.debug:
        ax.imshow(projection)
    else:
        ax.contour(projection, colors=projcol, alpha=alpha)
        ax.set_aspect('equal')
    # add padding
    print(ax.get_xlim(), ax.get_ylim())
    if padding != None:
        ax.set_xlim(ax.get_xlim()[0] - padding, ax.get_xlim()[1] + padding)
        ax.set_ylim(ax.get_ylim()[0] - padding, ax.get_ylim()[1] + padding)
    if axis != 1:
        ax.invert_yaxis()

    return (x_min, y_min, z_min), (x_max, y_max, z_max)

def probability_map_overlap(channel, fluorescence, area_num=None, binsize=200, axis=2, sigma=None, subregions=None, subregion_depth=None, projcol='k', padding=10, lower_lim=0.0001, saturation_multiplier=1):
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
    mins, _ = plot_projection(ax, area_num, padding=padding, axis=axis, projcol=projcol, alpha=0.1)
    if regions is not None:
        for child in regions:
            child_projection, (cx_min, cy_min, cz_min), _ = _get_projection(child, padding=padding, axis=axis)
            cx_offset, cy_offset, cz_offset = cx_min - mins[0], cy_min - mins[1], cz_min - mins[2]
            if axis == 2:
                child_projection = np.pad(child_projection, ((cy_offset,0),(cx_offset,0)))
            elif axis == 1:
                child_projection = np.pad(child_projection, ((cz_offset,0),(cx_offset,0)))
            else:
                child_projection = np.pad(child_projection, ((cy_offset,0),(cz_offset,0)))
            #child_projection = child_projection.T if axis == 0 else child_projection # side-on orientation does not need axis swapping
            ax.contour(child_projection, colors=projcol, alpha=0.05)
    
    group1_data, group2_data = probability_map_data(channel, fluorescence, area_num, binsize, axis, sigma, padding)
    
    im = __map_for_data(group1_data, group2_data, lower_lim=lower_lim, saturation_multiplier=saturation_multiplier)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    im = ax.imshow(im, cmap=cmap_LV_LS_pmap_cbar)
    cbar = plt.colorbar(im, cax)

    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(['$LS=0, LV=1$', '$LS=1, LV=1$', '$LS=1, LV=0$']):
        cbar.ax.text(2, j / 2, lab, ha='left', va='center')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    area_name = bt.get_area_info(area_num)[0][0]
    ax.set_title(f'{area_name} LS+LV mix from Fl={fluorescence} datasets')
    btf.save(f'dmap_HSV_{area_name}_ax={axis}_F={fluorescence}_subrgns={subregions}_sat={saturation_multiplier}', as_type='pdf')






## HELPERS

def __map_for_data(ax1_data, ax2_data, lower_lim=0.0001, saturation_multiplier=10):
    ax1_data[ax1_data < lower_lim] = 0
    ax2_data[ax2_data < lower_lim] = 0
    gradabs = np.add(ax1_data, ax2_data) / 2
    
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
        
        ### normalise angle to range cyan - magenta
        # cyan_angle = np.pi * 1
        # magenta_angle = np.pi * 1.6666666
        # angle_perc_circle = angle / (2 * np.pi)
        # angle = angle % (magenta_angle - cyan_angle)
        # while angle < cyan_angle:
            # angle += (magenta_angle - cyan_angle)
        # angle_norm = cyan_angle + ((angle_perc_circle * (magenta_angle - cyan_angle)))

        angle = angle * 1.5 + np.pi
        rgb_space = clrs.hsv_to_rgb((angle / 2 / np.pi, 
                                    absolute / max_abs, 
                                    1)) #absolute / max_abs))
        return rgb_space

    grad = np.array(list(map(grad_to_rgb, gradang.flatten(), gradabs.flatten())))
    grad = grad.reshape(tuple(list(gradabs.shape) + [3]))
    
    return grad


def _get_projection(area, padding=None, axis=0):
    area = 83 if area in ['Rostral-medial IO', 'Rostral-lateral IO', 'Caudal-medial IO', 'Caudal-lateral IO', 'Medial IO', 'Lateral IO', 'Rostral IO', 'Caudal IO'] else area
    parent, children = bt.children_from(area, depth=0)
    areas = [parent] + children
    
    atlas_ar = np.isin(bt.atlas, areas)

    nz = np.nonzero(atlas_ar)
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

def _plot_grouped_points(ax, dataset_cells, group_names, area_names, axis_title, is_horizontal):
    pre_compressed_dataset_cells = dataset_cells
    dataset_names, dataset_cells = _compress_into_groups(group_names, dataset_cells)
    names, datasets, cells = _prep_for_sns(area_names, dataset_names, dataset_cells)
    titles = ['Area', 'Dataset', axis_title]
    df = pd.DataFrame(zip(names, datasets, cells), columns=titles)
    area_name, dataset_name, dataset_cell = _group_points(names, pre_compressed_dataset_cells, group_names)
    points_df = pd.DataFrame(zip(area_name, dataset_name, dataset_cell), columns=titles)
    if is_horizontal:
        sns.barplot(x=titles[2], y=titles[0], hue=titles[1], hue_order=dataset_names, palette=csolid_group, data=df, ax=ax)
        sns.stripplot(x=titles[2], y=titles[0], hue=titles[1], hue_order=dataset_names, palette=csolid_group, dodge=True, edgecolor='w', linewidth=0.5, data=points_df, ax=ax)
    else:
        sns.barplot(x=titles[0], y=titles[2], hue=titles[1], hue_order=dataset_names, palette=csolid_group, data=df, ax=ax)
        sns.stripplot(x=titles[0], y=titles[2], hue=titles[1], hue_order=dataset_names, palette=csolid_group, dodge=True, edgecolor='w', linewidth=0.5, data=points_df, ax=ax)
    _display_legend_subset(ax, (2,3))
    return df

def _display_legend_subset(ax, idx_tup):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handle for i,handle in enumerate(handles) if i in idx_tup],
                [label for i,label in enumerate(labels) if i in idx_tup])

def __get_bt_groups():
    groups = list(dict.fromkeys([dataset.group for dataset in bt.datasets])) # get unique values
    if len(groups) != 2:
        print('Warning: Comparison plots should only be generated for two dataset groups.')
    return groups

def _colours_from_labels(names):
    datasets = [next((d for d in bt.datasets if d.name==i), None) for i in names] # get datasets by ordered labels
    groups = [i.group for i in datasets] # get their groups and return list of row/column label colours
    return list(map(lambda x: csolid_group[0] if x == __get_bt_groups()[0] else csolid_group[1], groups)) # list of colours

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






## Matrix plots

def fetch_groups(fluorescence):
    groups = __get_bt_groups()
    dataset_selection = [i for i in bt.datasets if i.fluorescence == fluorescence]
    datasets1 = [i for i in dataset_selection if i.group == groups[0]]
    datasets2 = [i for i in dataset_selection if i.group == groups[1]]
    datasets = datasets1 + datasets2
    num_datasets_in_group1 = len(datasets1)
    return datasets, num_datasets_in_group1

def is_IO_pixel_rostral(x, y, z):
    return z < 1225
def is_IO_pixel_medial(x, y, z):
    return x < 604

def get_matrix_data_for_io(datasets, split, norm):
    cells_list = []
    norm_options = ['none', 'IO', 'CBX', 'total']

    for d in datasets:
        points = bt._get_cells_in([83], d, 'r')
        cells = []
        if split == 'ml':
            split_mask = [is_IO_pixel_medial(x, y, z) for x, y, z in zip(*points)]
            cells.append(split_mask.count(True))
            cells.append(split_mask.count(False))
        elif split == 'rc':
            split_mask = [is_IO_pixel_rostral(x, y, z) for x, y, z in zip(*points)]
            cells.append(split_mask.count(True))
            cells.append(split_mask.count(False))
        elif split == 'both':
            split_mask = [is_IO_pixel_rostral(x, y, z) == True and is_IO_pixel_medial(x, y, z) == True for x, y, z in zip(*points)]
            cells.append(split_mask.count(True))
            split_mask = [is_IO_pixel_rostral(x, y, z) == True and is_IO_pixel_medial(x, y, z) == False for x, y, z in zip(*points)]
            cells.append(split_mask.count(True))
            split_mask = [is_IO_pixel_rostral(x, y, z) == False and is_IO_pixel_medial(x, y, z) == True for x, y, z in zip(*points)]
            cells.append(split_mask.count(True))
            split_mask = [is_IO_pixel_rostral(x, y, z) == False and is_IO_pixel_medial(x, y, z) == False for x, y, z in zip(*points)]
            cells.append(split_mask.count(True))

        # normalise
        if norm == norm_options[1]:
            total_labelling = sum(cells)
            cells = [count / total_labelling for count in cells]
        elif norm == norm_options[2]:
            cells = [(count - (d.starter_pedestal_norm * count)) / d.starter_normaliser for count in cells]
        elif norm == norm_options[3]:
            brain_labelling = d.num_cells(channel='r')
            cells = [(count / brain_labelling) * 100 for count in cells]
        else:
            print('Normalisation was not performed.')

        cells_list.append(cells)
        axis_title = 'Corrected cell count in IO'
    return cells_list, axis_title


def get_matrix_data(area_func, postprocess_for_scatter=False, fluorescence=False, value_norm=None, sort_matrix=True):
    datasets, num_g1 = fetch_groups(fluorescence)
    if postprocess_for_scatter == False:
        print('Warning: This function does not sort, even if postprocess_for_scatter=False')
    
    area_idxs, areas_title = area_func
    if areas_title == 'CF Inputs (anterograde)':
        area_labels = area_idxs
        if 'Rostral-medial IO' in area_labels:
            dataset_cells, axis_title = get_matrix_data_for_io(datasets, 'both', value_norm)
        elif 'Rostral IO' in area_labels:
            dataset_cells, axis_title = get_matrix_data_for_io(datasets, 'rc', value_norm)
        elif 'Medial IO' in area_labels:
            dataset_cells, axis_title = get_matrix_data_for_io(datasets, 'ml', value_norm)
        else:
            print('Error getting IO split information.')
    else:
        area_labels = bt.get_area_info(area_idxs)[0]
        dataset_cells, axis_title = bt._cells_in_areas_in_datasets(area_labels, datasets, 'r', normalisation=value_norm)
    
    dataset_cells = np.array(dataset_cells)
    std_g1 = np.std(dataset_cells[0:num_g1,:], axis=0)
    std_g2 = np.std(dataset_cells[num_g1:,:], axis=0)

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
    
    return area_labels, dataset_cells, std_g1, std_g2, datasets, areas_title, axis_title

def region_signal_matrix(area_func, value_norm='total', postprocess_for_scatter=False, vmax=None, figsize=(3,6), fluorescence=False, log_plot=True, sorting=True, ax=None,
                         areas_to_combine=None):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    area_labels, dataset_cells, _, _, datasets, areas_title, axis_title = get_matrix_data(area_func=area_func, postprocess_for_scatter=postprocess_for_scatter, sort_matrix=False, fluorescence=fluorescence, value_norm=value_norm)
    x_labels = [i.name for i in datasets]
    if postprocess_for_scatter:
        x_labels = __get_bt_groups()


    def replace_areas_with_combined_area(area_labels, dataset_cells, areas_to_combine):
        for new_area_name, old_area_codes in areas_to_combine.items():
            old_area_names = bt.get_area_info(old_area_codes)[0]
            print(old_area_names)
            indexes_to_remove = [area_labels.index(name) for name in old_area_names]

            combined_sum = np.array([np.sum(dataset_cells[:, indexes_to_remove], axis=1)])

            dataset_cells = np.delete(dataset_cells, indexes_to_remove, axis=1)
            area_labels = [ label for i, label in enumerate(area_labels) if i not in indexes_to_remove ]

            dataset_cells = np.concatenate((dataset_cells, combined_sum.T), axis=1)
            area_labels.append(new_area_name)
        return area_labels, dataset_cells

    if areas_to_combine is not None:
        area_labels, dataset_cells = replace_areas_with_combined_area(area_labels, dataset_cells, areas_to_combine)
    
    if sorting:
        sort_order = get_sorting_from_SI(dataset_cells, fluorescence)
        dataset_cells = dataset_cells[:, sort_order[::-1]]
        area_labels = [area_labels[i] for i in reversed(sort_order)]
    
    g1_mask = np.full(dataset_cells.T.shape, False)
    g1_mask[:, int(g1_mask.shape[1]/2):] = True # array where LS datasets on left are False
    
    if vmax == None:
        print('Warning: If vbounds is set to None, halves of the matrix may not have the same vmax.')
        vmax = dataset_cells.max()
    
    if log_plot:
        norm = clrs.LogNorm(vmin=1, vmax=vmax) # seems like both halves render with same vmin vmax without specifying
        sns.heatmap(dataset_cells.T, annot=False, mask=~g1_mask, cmap=cmaps_group[1], xticklabels=x_labels, yticklabels=area_labels, square=True, ax=ax, norm=norm)
        sns.heatmap(dataset_cells.T, annot=False, mask=g1_mask, cmap=cmaps_group[0], xticklabels=x_labels, yticklabels=area_labels, square=True, ax=ax, norm=norm)# , cbar_kws=dict(ticks=[])
    else:
        sns.heatmap(dataset_cells.T, annot=False, mask=~g1_mask, cmap=cmaps_group[1], xticklabels=x_labels, yticklabels=area_labels, square=True, ax=ax, vmin=0, vmax=vmax)
        sns.heatmap(dataset_cells.T, annot=False, mask=g1_mask, cmap=cmaps_group[0], xticklabels=x_labels, yticklabels=area_labels, square=True, ax=ax, vmin=0, vmax=vmax)# , cbar_kws=dict(ticks=[])
    
    if postprocess_for_scatter:
        colours = _colours_from_labels([datasets[0].name, datasets[-1].name])
    else:
        colours = _colours_from_labels(x_labels)
    [t.set_color(colours[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
    ax.set_title(areas_title)
    ax.set_ylabel(f'{axis_title}')
    btf.save(f'regionSignalMatrix_{areas_title}_log={log_plot}', as_type='pdf')
    return area_labels, dataset_cells



# don't plot the outline, just return the data
def probability_map_data(channel, fluorescence, area_num=None, binsize=200, axis=0, sigma=None, padding=10, no_alignment_to_region_bounds=False):
    atlas_res = 10
    assert binsize % atlas_res == 0, f'Binsize must be a multiple of atlas resolution ({atlas_res}um) to display correctly.'
    assert axis in [0, 1, 2], 'Must provide a valid axis number 0-2.'
    if area_num is None:
        area_num = 997

    _, min_bounds, max_bounds = _get_projection(area_num, padding=padding, axis=axis)
    
    ax1_data = get_density_map(channel, area_num, axis, atlas_res, binsize, sigma, __get_bt_groups()[0], min_bounds, max_bounds, fluorescence, no_alignment_to_region_bounds)
    ax2_data = get_density_map(channel, area_num, axis, atlas_res, binsize, sigma, __get_bt_groups()[1], min_bounds, max_bounds, fluorescence, no_alignment_to_region_bounds)
    return ax1_data, ax2_data


def get_density_map(channel, area, axis, atlas_res, binsize, sigma, group, min_bounds, max_bounds, fluorescence, no_alignment_to_region_bounds=False):
    (px_min, py_min, pz_min), (px_max, py_max, pz_max) = min_bounds, max_bounds

    hist_list = []
    datasets_to_bin = [d for d in bt.datasets if d.group == group and d.fluorescence == fluorescence]
    assert len(datasets_to_bin) > 0, 'Could not find any datasets matching group and fluorescence setting.'
    for d in datasets_to_bin:
        if area is None:
            points = np.array(d.cell_coords[channel]).T
        elif area == 'Rostral IO':
            points = bt._get_cells_in(83, d, channel)
            points = np.array([[x, y, z] for x, y, z in zip(*points) if is_IO_pixel_rostral(x, y, z) == True])
        elif area == 'Caudal IO':
            points = bt._get_cells_in(83, d, channel)
            points = np.array([[x, y, z] for x, y, z in zip(*points) if is_IO_pixel_rostral(x, y, z) == False])
        elif area == 'Medial IO':
            points = bt._get_cells_in(83, d, channel)
            points = np.array([[x, y, z] for x, y, z in zip(*points) if is_IO_pixel_medial(x, y, z) == True])
        elif area == 'Caudal IO':
            points = bt._get_cells_in(83, d, channel)
            points = np.array([[x, y, z] for x, y, z in zip(*points) if is_IO_pixel_medial(x, y, z) == False])
        elif area == 'Rostral-medial IO':
            points = bt._get_cells_in(83, d, channel)
            points = np.array([[x, y, z] for x, y, z in zip(*points) if is_IO_pixel_rostral(x, y, z) == True and is_IO_pixel_medial(x, y, z) == True])
        elif area == 'Rostral-lateral IO':
            points = bt._get_cells_in(83, d, channel)
            points = np.array([[x, y, z] for x, y, z in zip(*points) if is_IO_pixel_rostral(x, y, z) == True and is_IO_pixel_medial(x, y, z) == False])
        elif area == 'Caudal-medial IO':
            points = bt._get_cells_in(83, d, channel)
            points = np.array([[x, y, z] for x, y, z in zip(*points) if is_IO_pixel_rostral(x, y, z) == False and is_IO_pixel_medial(x, y, z) == True])
        elif area == 'Caudal-lateral IO':
            points = bt._get_cells_in(83, d, channel)
            points = np.array([[x, y, z] for x, y, z in zip(*points) if is_IO_pixel_rostral(x, y, z) == False and is_IO_pixel_medial(x, y, z) == False])
        else:
            parent, children = bt.children_from(area, depth=0)
            areas = [parent] + children
            points = np.array(bt._get_cells_in(areas, d, channel)).T

        x_bins, y_bins, z_bins = get_bins(0, binsize), get_bins(1, binsize), get_bins(2, binsize)
        hist, _ = np.histogramdd(points, bins=(x_bins, y_bins, z_bins), range=((0,1140),(0,800),(0,1320)), normed=False)
        
        if hist.sum() != 0:
            hist = hist / hist.sum() # turn into probability density distribution

        if sigma is not None: # 3D smooth # sigma = width of kernel
            x, y, z = np.arange(-3,4,1), np.arange(-3,4,1), np.arange(-3,4,1) # coordinate arrays -- make sure they include (0,0)!
            xx, yy, zz = np.meshgrid(x,y,z)
            kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
            hist = signal.convolve(hist, kernel, mode='same')

        hist = np.sum(hist, axis=axis) # take the maximum projection of the distribution

        if no_alignment_to_region_bounds == False:
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
    
    return av_im





# matrix sorting helpers

def selectivity_index(g1_labelling, g2_labelling):
    return ((g1_labelling - g2_labelling) / (g1_labelling + g2_labelling))

def colours_from_SI_array(SIs):
    def get_colour(x):
        return csolid_group[0] if x <= 0 else csolid_group[1]
    return np.array([get_colour(xi) for xi in SIs])

def calculate_SI_from_means(g1_matrix, g2_matrix):
    LS_labelling = np.mean(g1_matrix, axis=0)
    LV_labelling = np.mean(g2_matrix, axis=0)
    
    avgs = selectivity_index(LV_labelling, LS_labelling)
    colours = colours_from_SI_array(avgs)
    
    return avgs, colours

def calculate_SI_with_errors(g1_matrix, g2_matrix):
    SI_combos_array = []
    for i in range(0, g1_matrix.shape[1]):
        LS_area_vals, LV_area_vals = g1_matrix[:,i], g2_matrix[:,i]
        SI_combinations = []
        for LS_val in LS_area_vals:
            for LV_val in LV_area_vals:
                SI_combinations.append(selectivity_index(LV_val, LS_val))
        SI_combos_array.append(SI_combinations)
        
    avgs = [np.mean(i) for i in SI_combos_array]
    stds = [np.std(i) / np.sqrt(len(i)) for i in SI_combos_array] # convert standard deviation to error
    colours = colours_from_SI_array(avgs)
    
    return avgs, stds, colours

def get_sorting_from_SI(dataset_cells, fluorescence, mean=False):
    _, num_g1 = fetch_groups(fluorescence) # get sum across each group
    LS, LV = dataset_cells[0:num_g1,:], dataset_cells[num_g1:,:]
    
    if mean:
        avgs, _ = calculate_SI_from_means(LS, LV)
    else:
        avgs, _, _ = calculate_SI_with_errors(LS, LV)
    
    # sort primarily by selectivity direction and then by quantity of labelling
    LS_labelling = np.mean(LS, axis=0)
    LV_labelling = np.mean(LV, axis=0)
    order_weighting = [LV_labelling[i] if (x >=0) else -LS_labelling[i] for i, x in enumerate(avgs)]
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

def get_corr_indexes(channel, area_idxs, gradient, fluorescence=False):
    coefs = []
    for area in area_idxs:
        LS_data, LV_data = probability_map_data(channel, fluorescence, area_num=area, binsize=50, axis=2, sigma=1, no_alignment_to_region_bounds=True)
        LS_data, LV_data, _ = remove_corner_points(LS_data, LV_data, 0.05)
        coef = get_corr_index(LS_data, LV_data, gradient)
        coefs.append(coef)
    return coefs

def remove_corner_points(ax1_data, ax2_data, percentile):
    max_xy = min(np.max(ax1_data), np.max(ax2_data))
    idxs_below_thresh1 = np.argwhere(ax1_data.flatten() < max_xy * percentile)
    idxs_below_thresh2 = np.argwhere(ax2_data.flatten() < max_xy * percentile)
    idxs_to_remove = np.intersect1d(idxs_below_thresh1, idxs_below_thresh2)
    ax1_data = np.delete(ax1_data.flatten(), idxs_to_remove)
    ax2_data = np.delete(ax2_data.flatten(), idxs_to_remove)
    return ax1_data, ax2_data, max_xy