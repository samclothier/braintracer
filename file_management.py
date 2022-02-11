import os, sys, imageio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from bs4 import BeautifulSoup

script_dir = os.getcwd() #path.dirname(os.path.abspath(__file__)) #<-- absolute dir the script is in
atlas = BrainGlobeAtlas('allen_mouse_10um')

def _get_path(file_name):
    if file_name.startswith('cells_'):
        child_dir = 'braintracer\\cellfinder'
    elif file_name.startswith('reg_'):
        child_dir = 'braintracer\\downsampled_data'
    elif file_name.startswith('groundtruth_'):
        child_dir = 'braintracer\\ground_truth'
    elif file_name.startswith('structures'):
        child_dir = 'braintracer'
    else:
        print('Unexpected file name. Braintracer accepts files with the following format:\ncells_[].xml/csv\nreg_[]_[].tiff\ngroundtruth_[].xml\nstructures.csv')
        return None
    return os.path.join(script_dir, child_dir+'\\'+file_name)

# opens xml files from napari containing cell points
def open_file(name): # open files
    all_X, all_Y, all_Z = [], [], []
    neg_X, neg_Y, neg_Z = [], [], []
    pos_X, pos_Y, pos_Z = [], [], []
    file_path = _get_path(name)
    ext = name.split('.')[-1]
    if ext == 'xml':
        # may be cellfinder output or ground truth # TODO: add throw for opening ground truth dir
        with open(file_path, 'r') as f:
            data = BeautifulSoup(f, 'xml')
        types = data.find_all('Marker_Type')
        for t in types:
            type_num = int(t.Type.string) #print(len(list(typ.children)))
            markers = t.find_all('Marker')
            for marker in markers:
                all_X.append(int(marker.MarkerX.contents[0]))
                all_Y.append(int(marker.MarkerY.contents[0]))
                all_Z.append(int(marker.MarkerZ.contents[0]))
                if type_num == 1:
                    neg_X.append(int(marker.MarkerX.contents[0]))
                    neg_Y.append(int(marker.MarkerY.contents[0]))
                    neg_Z.append(int(marker.MarkerZ.contents[0]))
                elif type_num == 2:
                    pos_X.append(int(marker.MarkerX.contents[0]))
                    pos_Y.append(int(marker.MarkerY.contents[0]))
                    pos_Z.append(int(marker.MarkerZ.contents[0]))
                else:
                    print(f'Unexpected marker type number: {type_num}')
        return ([all_X, all_Y, all_Z],
                [neg_X, neg_Y, neg_Z],
                [pos_X, pos_Y, pos_Z])
    elif ext == 'tiff':
        reader = imageio.get_reader(file_path)
        images = []
        for frame in reader:
            images.append(frame)
        return images
    elif ext == 'csv':
        if name.startswith('cells_'):
            cell_df = pd.read_csv(file_path)
            z_coords = cell_df['coordinate_atlas_axis_0'].to_list()
            y_coords = cell_df['coordinate_atlas_axis_1'].to_list()
            x_coords = cell_df['coordinate_atlas_axis_2'].to_list()
            return [x_coords, y_coords, z_coords]
        elif name.startswith('structures'):
            area_indexes = pd.read_csv(file_path)
            area_indexes = area_indexes.set_index('id')
            return area_indexes
        else:
            print(f'Cannot load CSV with name {file_name}')
    else:
        print('Unexpected file extension')
        return None

def get_atlas():
    global atlas
    annotation = np.array(atlas.annotation)
    atlas_oriented = annotation[:,:,::-1] # flip atlas along x axis (I know it's pointless, but the cells are also flipped)
    return atlas_oriented

def get_lookup_df():
    df = atlas.lookup_df
    df = df.set_index('id')
    return df

def save(file_name, as_type):
    if file_name.startswith('injection_'):
        dir_name = 'braintracer/TRIO/'
        dir_path = os.path.join(script_dir, dir_name)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        file_name = dir_path + file_name

    if as_type == 'png':
        plt.savefig(f'{file_name}.png', dpi=600, bbox_inches='tight')
    elif as_type == 'pdf':
        pp = PdfPages(f'{file_name}.pdf')
        pp.savefig()
        pp.close()

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout