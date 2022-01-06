import braintracer.file_management as btf
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import Counter
import numpy as np
from itertools import chain

datasets = []
area_indexes = btf.open_file('structures.csv')

class Dataset:
    def __init__(self, name, group, raw_dim, display_validation=False):
        self.name = name
        self.group = group
        self.raw_dim = raw_dim
        #all_cf, neg_cf, pos_cf = btf.open_file(f'cells_r_{self.name}.xml')
        #if display_validation:
            #print(f'All: {len(all_cf[0])}, Negatives: {len(neg_cf[0])}, Positives: {len(pos_cf[0])}')
        #self.all_cells, self.neg_cells, self.pos_cells = all_cf, neg_cf, pos_cf # only pos_cells are moved into downsampled space
        self.cells = btf.open_file(f'cells_r_{self.name}.csv')
        self.atlas = btf.open_file(f'atlas.tiff')
        self.data = btf.open_file(f'reg_r_{self.name}.tiff')
        self.dwn_dim = self.set_downsampled_dim()
        validate_dimensions(self, display=display_validation)
        datasets.append(self)
        raw_cell_counter = Counter()
        propagated_cell_counter = Counter()
    def set_downsampled_dim(self):
        first_im = self.data[0]
        dim_x, dim_y = first_im.shape[0], first_im.shape[1]
        dim_z = len(self.data)
        return (dim_x, dim_y, dim_z)
    def get_total_cells(self):
        return len(self.cells[0])
    def show_coronal_section(self, raw_slice_num, raw_len):
        frac = raw_slice_num / raw_len
        atlas_len = 1255 # atlas len is 1320 but registration cut-off is about 1250
        ds_slice_num = atlas_len - int(atlas_len*frac)
        data = np.array(self.atlas)
        plt.suptitle(f'{self.name} Slice {str(ds_slice_num)}')
        plt.imshow(data[ds_slice_num,:,:], norm=colors.LogNorm())
        layer_point_idxs = np.array(np.where(np.array(self.cells[2])==ds_slice_num))[0]
        for i in layer_point_idxs:
            plt.scatter(self.cells[0][i], self.cells[1][i], c='r', s=0.5)

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
    im_sets = [dataset.atlas, dataset.data]
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
def raw_to_downsampled(raw_dim, downsampled_dim, cell_coords):
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
def get_area_index_from(dataset, z, x, y):
    im = dataset[z]
    if x <= im.shape[1] and y <= im.shape[0]:
        area_index = int(im[y,x])
    else:
        print('Warning: Point out of bounds')
    if area_index >= 0:
        return area_index
    else:
        print('Warning: Area index is < 0')
        return 0

# return the number of cells in each brain region
def count_cells(atlas, x_vals, y_vals, z_vals):
    counter = Counter()
    for idx, z in enumerate(z_vals):
        x = x_vals[idx]
        y = y_vals[idx]
        area_index = get_area_index_from(atlas, z, x, y)
        counter.setdefault(area_index, 0)
        counter[area_index] = counter[area_index] + 1
    total_cells = sum(counter.values()) # <-- if total cells needs to be verified
    counter = counter.most_common()
    return counter

def get_cells_in(area, dataset):
    cells_x, cells_y, cells_z = [], [], []
    if isinstance(area, str):
        _, area_index, _ = get_area_info([area], dataset.propagated_cell_counter)
    else:
        area_index = area
    cls_x, cls_y, cls_z = dataset.cells[0], dataset.cells[1], dataset.cells[2]
    for idx, z in enumerate(cls_z):
        x = cls_x[idx]
        y = cls_y[idx]
        cur_area_idx = get_area_index_from(dataset.atlas, z, x, y)
        if cur_area_idx == area_index:
            cells_x.append(x)
            cells_y.append(y)
            cells_z.append(z)
    return cells_x, cells_y, cells_z

def children_from(parent, depth):
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

# quantify numbers of cells for parent brain areas
def propagate_cells_through_inheritance_tree(original_counter, area_indexes=area_indexes):
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

# return area full-names, area indexes, and area cell count given short-letter codes or area indexes
def get_area_info(codes, new_counter=None, area_indexes=area_indexes):
    if not isinstance(codes, list):
        codes = [codes]
    if new_counter == None:
        new_counter = datasets[0].propagated_cell_counter
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
def get_extra_cells(codes, original_counter, area_indexes):
    names = area_indexes.loc[codes, 'name'].tolist()
    try:
        cells = list(map(lambda x: [item for item in original_counter if item[0] == x][0][-1], codes))
    except IndexError:
        #print('The zoomed in parent area has no cells outside of children')
        cells = list(map(lambda x: x*0, codes))
    return names, cells


# main analysis script
summary_areas = ['CTX','CNU','TH','HY','MB','MY','P','CBX','CBN']
summary_names, _, _ = get_area_info(summary_areas, Counter(), area_indexes)
nrmdltn_areas = ['VTA','RAmb', 'RM', 'RO', 'RPA', 'RPO', 'CS', 'SNc', 'SNr', 'LC']
nrmdltn_names, _, _ = get_area_info(nrmdltn_areas, Counter(), area_indexes)
def run_analysis(debug=False):
    summary_cells_list = []
    nrmdltn_cells_list = []
    for idx, dataset in enumerate(datasets):
        dataset.raw_cell_counter = count_cells(dataset.atlas, dataset.cells[0], dataset.cells[1], dataset.cells[2])
        dataset.propagated_cell_counter = propagate_cells_through_inheritance_tree(dataset.raw_cell_counter)
        
        summary_names, _, summary_cells = get_area_info(summary_areas, dataset.propagated_cell_counter)
        summary_cells = list(map(lambda x: (x / dataset.get_total_cells())*100, summary_cells))
        summary_cells_list.append(summary_cells)

        nrmdltn_names, _, nrmdltn_cells = get_area_info(nrmdltn_areas, dataset.propagated_cell_counter)
        nrmdltn_cells = list(map(lambda x: (x / dataset.get_total_cells())*100, nrmdltn_cells))
        nrmdltn_cells_list.append(nrmdltn_cells)
        
        if debug:
            n, i, IO_cells = get_area_info(['IO'], dataset.propagated_cell_counter)
            print(f'{dataset.name}: {IO_cells[0]} cells in {dataset.name} inferior olive, out of {dataset.get_total_cells()} total. {sum(summary_cells):.1f}% cells in non-tract and non-ventricular areas')
    results.summary_cells = summary_cells_list
    results.nrmdltn_cells = nrmdltn_cells_list


def get_cells_in(areas, dataset):
    cells_x, cells_y, cells_z = [], [], []
    if isinstance(areas[0], str):
        area_idxs = get_area_info(areas)[1]
    else:
        area_idxs = areas
    cls_x, cls_y, cls_z = dataset.cells[0], dataset.cells[1], dataset.cells[2]
    for idx, z in enumerate(cls_z):
        x = cls_x[idx]
        y = cls_y[idx]
        cur_area_idx = get_area_index_from(dataset.atlas, z, x, y)
        if cur_area_idx in areas:
            cells_x.append(x)
            cells_y.append(y)
            cells_z.append(z)
    return cells_x, cells_y, cells_z

def project_dataset(ax, dataset, area, s, contour):
    parent, children = children_from(area, depth=0)
    areas = [parent] + children
    
    atlas = np.array(dataset.atlas)
    atlas = np.isin(atlas, areas)
    projection = atlas.any(axis=0)
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
    
    X, Y, _ = get_cells_in(areas, dataset)
    X = [x-y_offset for x in X]
    Y = [y-x_offset for y in Y]
    colour = 'magenta' if dataset.group == datasets[0].group else (0,0.5,1)
    ax.scatter(X, Y, color=colour, s=s, label=dataset.group, zorder=10)