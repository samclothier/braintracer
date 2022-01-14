import braintracer.file_management as btf
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import Counter
import numpy as np
from itertools import chain

datasets = []
area_indexes = btf.open_file('structures.csv')
atlas = btf.open_file(f'atlas.tiff')
debug = False

class Dataset:
    def __init__(self, name, group, sig, bg):
        self.name, self.group, self.sig, self.bg = name, group, sig, bg
        self.ch1_cells = btf.open_file(f'cells_{self.sig}_{self.name}.csv')
        self.ch2_cells = btf.open_file(f'cells_{self.bg}_{self.name}.csv')
        self.data = btf.open_file(f'reg_r_{self.name}.tiff') # will be used for fluorescence analysis
        validate_dimensions(self, display=debug)
        datasets.append(self)
        self.raw_ch1_cells_by_area = self.__count_cells(self.ch1_cells)
        self.raw_ch2_cells_by_area = self.__count_cells(self.ch2_cells)
        self.ch1_cells_by_area = self.__propagate_cells_through_inheritance_tree(self.raw_ch1_cells_by_area)
        self.ch2_cells_by_area = self.__propagate_cells_through_inheritance_tree(self.raw_ch2_cells_by_area)
        if debug:
            _, _, IO_cells1 = get_area_info(['IO'], self.ch1_cells_by_area)
            _, _, IO_cells2 = get_area_info(['IO'], self.ch2_cells_by_area)
            print(f'{self.name} ({self.group}): {IO_cells1[0]} ch1 cells in {self.name} inferior olive, out of {self.num_cells(ch1=True)} total ch1 cells.')
            print(f'{self.name} ({self.group}): {IO_cells2[0]} ch2 cells in {self.name} inferior olive, out of {self.num_cells(ch1=False)} total ch2 cells.')

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
        data = np.array(atlas)
        plt.suptitle(f'{self.name} Slice {str(ds_slice_num)}')
        plt.imshow(data[ds_slice_num,:,:], norm=colors.LogNorm())
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
            area_index = _get_area_index(z, x, y)
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
        already_accessed = []
        new_counter = Counter()
        for idx, row in enumerate(area_indexes.itertuples()):
            cur_area_idx = row.Index
            try: # access the value of the counter for this area:
                child_cells = [item for item in original_counter if item[0] == cur_area_idx][0][-1] # list of tuples, func returns a list of filtered tuples, so get first item, which is a tuple (area, flrsnce) so grab flrsnce
            except IndexError:
                pass
            else:
                if not any(x == cur_area_idx for x in already_accessed): # check no indexes are assigned to more than once
                    id_path = row.structure_id_path
                    id_path_list = id_path.split('/')
                    id_path_list = [i for i in id_path_list if i != ''] # remove blank and current indexes
                    for i in id_path_list: # propagate lowest area count through all parent areas
                        area_index = int(i)
                        new_counter.setdefault(area_index, 0)
                        new_counter[area_index] += child_cells
                for i in id_path_list: # add parent and child areas to done list if not already added
                    if not any(x == i for x in already_accessed):
                        already_accessed.append(i)
                already_accessed.append(cur_area_idx)
        return new_counter

    def get_cells_in(self, area, ch1=True):
        '''
        returns the x, y, z coordinates of cells in a specific brain area
        area: can be string or index reference to single area
        ch1: True or False, True returns cells in signal channel, False returns cells in background channel
        NOTE: doesn't seem to work as expected, likely can remove function
        '''
        cells_x, cells_y, cells_z = [], [], []
        if isinstance(area, str):
            _, area_index, _ = get_area_info([area], self.ch1_cells_by_area)
        else:
            area_index = area
        cell_coords = self.ch1_cells if ch1 else self.ch2_cells
        cls_x, cls_y, cls_z = cell_coords[0], cell_coords[1], cell_coords[2]
        for idx, z in enumerate(cls_z):
            x = cls_x[idx]
            y = cls_y[idx]
            cur_area_idx = _get_area_index(z, x, y)
            if cur_area_idx == area_index:
                cells_x.append(x)
                cells_y.append(y)
                cells_z.append(z)
        return cells_x, cells_y, cells_z

    def get_starter_cells_in(self, area, xy_tol_um=10, z_tol_um=10):
        # atlas is 10um, so divide um tolerance by 10
        xy_tol = np.ceil(xy_tol_um / 10)
        z_tol = np.ceil(z_tol_um / 10)
        if debug:
            print(f'Atlas space xy tolerance is {xy_tol} and z tolerance is {z_tol}')
        parent, children = children_from(area, depth=0)
        areas = [parent] + children
        ch1_cells_in_area = _get_cells_in(areas, self, ch1=True)
        ch2_cells_in_area = _get_cells_in(areas, self, ch1=False)
        matching_ch1_idxs = []
        matching_ch2_idxs = []

        for ch2_idx, Z in enumerate(ch2_cells_in_area[2]):
            X = ch2_cells_in_area[0][ch2_idx]
            Y = ch2_cells_in_area[1][ch2_idx]
            
            for ch1_idx, z in enumerate(ch1_cells_in_area[2]):
                x = ch1_cells_in_area[0][ch1_idx]
                y = ch1_cells_in_area[1][ch1_idx]
                if (z-z_tol <= Z <= z+z_tol) & (x-xy_tol <= X <= x+xy_tol) & (y-xy_tol <= Y <= y+xy_tol):
                    matching_ch1_idxs.append(ch2_idx)
                    matching_ch2_idxs.append(ch1_idx)
                    break
        if debug:
            print(f'{len(ch1_cells_in_area[2])} ch1 cells and {len(ch2_cells_in_area[2])} ch2 cells in the {str(area)}.')
            print(f'{len(matching_ch1_idxs)} starter cells found in the {str(area)}.')
        return len(matching_ch2_idxs)

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

# validate image dimensions
def validate_dimensions(dataset, display=False):
    im_sets = [atlas, dataset.data]
    first_images = list(map(lambda dataset: next(x for x in dataset if True), im_sets))
    def check_image_dimensions(images):
        for first_im in images:
            for second_im in images:
                if first_im.shape != second_im.shape:
                    return False
        return True
    def check_dataset_length(datasets):
        for first_dataset in datasets:
            for second_dataset in datasets:
                if len(first_dataset) != len(second_dataset):
                    return False
        return True
    assert check_image_dimensions(first_images), 'Images do not have the same dimensions.'
    assert check_dataset_length(im_sets), 'Datasets are not the same length in z-axis.'
    if display:
        print(f'Resolutions: {list(map(lambda x: x.shape, first_images))}')
        fig, axes = plt.subplots(1, len(im_sets), figsize=(8,5))
        for idx, ax in enumerate(axes):
            dist = int(len(im_sets[idx])*0.7)
            ax.imshow(im_sets[idx][dist], norm=colors.LogNorm())

# downsample points from raw coordinates
def _raw_to_downsampled(raw_dim, downsampled_dim, cell_coords):
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

# get the index referring to the brain area in which a cell is located
def _get_area_index(z, x, y):
    im = atlas[z]
    if x <= im.shape[1] and y <= im.shape[0]:
        area_index = int(im[y,x])
    else:
        print('Warning: Point out of bounds')
    if area_index >= 0:
        return area_index
    else:
        print('Warning: Area index is < 0')
        return 0

def children_from(parent, depth):
    '''
    parent: choose parent area
    depth: number of steps down the inheritance tree to fetch child areas from. 0 returns all children.
    '''
    parent = area_indexes.loc[area_indexes['name']==parent].index[0]
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
    if not isinstance(codes, list):
        codes = [codes]
    if new_counter == None:
        new_counter = datasets[0].ch1_cells_by_area
    if isinstance(codes[0], int):
        names = area_indexes.loc[codes, 'name'].tolist()
        idxes = codes
        cells = list(map(lambda x: new_counter[int(x)], idxes))
    elif isinstance(codes[0], str):
        try:
            names = list(map(lambda x: area_indexes.loc[area_indexes['acronym']==x, 'name'].item(), codes))
            idxes = list(map(lambda x: area_indexes.loc[area_indexes['acronym']==x, 'name'].index[0], codes))
        except ValueError:
            names = list(map(lambda x: area_indexes.loc[area_indexes['name']==x, 'name'].item(), codes))
            idxes = list(map(lambda x: area_indexes.loc[area_indexes['name']==x, 'name'].index[0], codes))
        cells = list(map(lambda x: new_counter[int(x)], idxes))
    else:
        'Unknown area reference format.'
    return names, idxes, cells

# check if there are extra cells assigned to the parent area rather than the children
def _get_extra_cells(codes, original_counter):
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

def _get_cells_in(areas, dataset, ch1=True):
    cells_x, cells_y, cells_z = [], [], []
    cell_coords = dataset.ch1_cells if ch1 else dataset.ch2_cells
    cls_x, cls_y, cls_z = cell_coords[0], cell_coords[1], cell_coords[2]
    for idx, z in enumerate(cls_z):
        x = cls_x[idx]
        y = cls_y[idx]
        cur_area_idx = _get_area_index(z, x, y)
        if cur_area_idx in areas:
            cells_x.append(x)
            cells_y.append(y)
            cells_z.append(z)
    return cells_x, cells_y, cells_z

def _project_dataset(ax, dataset, area, ch1, s, contour):
    parent, children = children_from(area, depth=0)
    areas = [parent] + children
    
    atlas_ar = np.array(atlas)
    atlas_ar = np.isin(atlas_ar, areas)
    projection = atlas_ar.any(axis=0)
    nz = np.nonzero(projection)
    x_offset = nz[0].min()-10
    y_offset = nz[1].min()-10
    arr_trimmed = projection[nz[0].min()-10:nz[0].max()+11,
                             nz[1].min()-10:nz[1].max()+11]
    projected_area = arr_trimmed.astype(int)
    if contour:
        ax.contour(projected_area, colors='k')
        ax.set_aspect('equal')
    else:
        ax.imshow(arr_trimmed)
    
    # colour = 'magenta' if dataset.group == datasets[0].group else (0,0.5,1)
    def show_cells(channel, colour):
        X_r, Y_r, _ = _get_cells_in(areas, dataset, channel)
        X_r = [x-y_offset for x in X_r]
        Y_r = [y-x_offset for y in Y_r]
        ax.scatter(X_r, Y_r, color=colour, s=s, label=dataset.group, zorder=10)
    if ch1 == None:
        show_cells(True, 'r')
        show_cells(False, 'g')
    elif ch1 == True:
        show_cells(True, 'r')
    else:
        show_cells(False, 'g')