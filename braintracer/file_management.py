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

import os, sys, imageio, cv2, glob, contextlib, pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from bs4 import BeautifulSoup
from PIL import Image

script_dir = os.getcwd() #<-- dir of the notebook file
package_dir = os.path.dirname(os.path.realpath(__file__)) #<-- dir of the package itself
atlas = BrainGlobeAtlas('allen_mouse_10um')

def _get_path(file_name, vID=None):
	if file_name.startswith('cells_'):
		child_dir = 'braintracer\\cellfinder'
	elif file_name.startswith('reg_'):
		child_dir = 'braintracer\\downsampled_data'
	elif file_name.startswith('groundtruth_'):
		child_dir = 'braintracer\\ground_truth'
	elif file_name.startswith('structures'):
		child_dir = None # local file, now part of the package
	elif file_name.startswith('atlas'):
		child_dir = 'braintracer\\registered_atlases'
	elif file_name.startswith('binary_'):
		child_dir = 'braintracer\\fluorescence'
	elif file_name.startswith('injection_'):
		child_dir = 'braintracer\\TRIO'
	elif file_name.startswith('video_'):
		assert vID is not None, 'video ID variable must be supplied for saving video frames'
		child_dir = f'braintracer\\videos\\{file_name.split("_")[1]}_{vID}'
	else:
		raise ValueError('Unexpected file name. Braintracer accepts files with the following format:\ncells_[].xml/csv\nreg_[]_[].tiff\ngroundtruth_[].xml\nstructures.csv')

	if child_dir is not None:
		if not os.path.isdir(child_dir):
			os.makedirs(child_dir)
		path = os.path.join(script_dir, child_dir+'\\'+file_name)
	else:
		path = os.path.join(package_dir, file_name)
	return path

# opens xml files from napari containing cell points
def open_file(name, atlas_25=False): # open files
	all_X, all_Y, all_Z = [], [], []
	neg_X, neg_Y, neg_Z = [], [], []
	pos_X, pos_Y, pos_Z = [], [], []
	file_path = _get_path(name)
	assert os.path.isfile(file_path), f'Could not find file: {file_path}'
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
		reader = imageio.get_reader(file_path) # importing with tifffile appears to occasionally lead to wrong dimensions
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
			hemisphere = cell_df['hemisphere'].to_list()
			if atlas_25:
				z_coords = list(np.floor(np.array(z_coords) * 2.5).astype(int)) # convert coords in 25um atlas space to 10um
				y_coords = list(np.floor(np.array(y_coords) * 2.5).astype(int))
				x_coords = list(np.floor(np.array(x_coords) * 2.5).astype(int))
			return [x_coords, y_coords, z_coords, hemisphere] # flip to x, y, z
		elif name.startswith('structures'):
			area_indexes = pd.read_csv(file_path)
			area_indexes = area_indexes.set_index('id')
			return area_indexes
		else:
			print(f'Cannot load CSV with name {name}')
	elif ext == 'npy':
		coordinates = np.load(file_path)
		coordinates = np.c_[ np.repeat(None, coordinates.shape[0]), coordinates ] # Add an extra column for hemisphere = None
		coordinates = np.flip(coordinates.T, axis=0) # flip to x, y, z
		return coordinates.tolist()
	elif ext == 'pkl':
		return pickle.load(open(f'{file_path}', 'rb'))
	else:
		print('Unexpected file extension')
		return None

def open_transformed_brain(dataset):
	name = dataset.name
	path = os.path.join(script_dir, name+'\\'+'transform\\*')
	assert os.path.isdir(path), f'Please provide transformed stack at {path}'
	files = glob.glob(path)
	return files

def open_registered_stack(dataset):
	if dataset.fluorescence:
		if dataset.skimmed:
			name = f'binary_registered_stack_skimmed_{dataset.name}_{dataset.channels[0]}.npy'
		else:
			name = f'binary_registered_stack_{dataset.name}_{dataset.channels[0]}.npy'
		path = _get_path(name)
		return np.load(path)
	else:
		stack = np.array(open_file(f'reg_{dataset.name}_{dataset.channels[0]}.tiff'))[0]
		return stack

def get_atlas():
	global atlas
	return np.array(atlas.annotation)

def get_reference():
	global atlas
	return np.array(atlas.reference)

def get_lookup_df():
	df = atlas.lookup_df
	df = df.set_index('id')
	return df

def save(file_name, as_type, dpi=600, vID=None, file=None):
	if vID is not None:
		dir_path = _get_path(file_name, vID)
	else:
		dir_path =  os.path.join(script_dir, 'braintracer/figures/', file_name)

	if as_type == 'png':
		if vID is None:
			plt.savefig(f'{dir_path}.png', dpi=dpi, bbox_inches='tight')
		else:
			plt.savefig(f'{dir_path}.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
	elif as_type == 'jpg':
		plt.savefig(f'{dir_path}.jpg', dpi=dpi, bbox_inches='tight')
	elif as_type == 'pdf':
		pp = PdfPages(f'{dir_path}.pdf')
		pp.savefig(dpi=dpi)
		pp.close()
	elif as_type == 'pkl':
		assert file != None, 'pkl file must be provided when saving pickle files.'
		pickle.dump(file, open(f'{dir_path}.pkl', 'wb'))

def create_video(dataset_name, vID, fps=30):
	dir_name = f'braintracer/videos/{dataset_name}_{vID}/'
	dir_path = os.path.join(script_dir, dir_name)
	video_name = f'video_{dataset_name}_{vID}_{fps}fps.avi'

	images = [img for img in os.listdir(dir_path) if img.endswith(".png")]
	frame = cv2.imread(os.path.join(dir_path, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(f'braintracer/videos/{video_name}', 0, fps, (width,height))

	for image in images:
		video.write(cv2.imread(os.path.join(dir_path, image)))

	cv2.destroyAllWindows()
	video.release()

def create_gif(dataset_name, vID, fps=30):
	fp_in = f'/braintracer/videos/{dataset_name}_{vID}/video_*.png'
	fp_out = f'/braintracer/videos/video_{dataset_name}_{vID}_{fps}fps.gif'

	# use exit stack to automatically close opened images
	with contextlib.ExitStack() as stack:
		# lazily load images
		imgs = (stack.enter_context(Image.open(f))
				for f in sorted(glob.glob(fp_in)))
		# extract  first image from iterator
		img = next(imgs)
		# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
		img.save(fp=fp_out, format='GIF', append_images=imgs,
				 save_all=True, duration=int((1/fps)*1000), loop=0)

def verify_image_integrity(dataset):
	def code_comparisons(pre_code1, cur_code1, pre_code2, cur_code2):
		pre_code1, cur_code1 = int(pre_code1), int(cur_code1)
		pre_code2, cur_code2 = int(pre_code2), int(cur_code2) # convert all string codes to ints for comparison
		if cur_code1 == pre_code1:
			pass
		elif (cur_code1 == pre_code1 + 1) and cur_code2 == 1:
			pass
		else:
			return False
		if cur_code2 == pre_code2 + 1:
			return True
		elif cur_code2 == 1 and pre_code2 == 8:
			return True
		else:
			return False
	for i in range(3):
		channel_num = i + 1
		path = os.path.join(script_dir, dataset, str(channel_num))
		if os.path.isdir(path):
			for root, dirs, files in os.walk(path):
				prev_name = 'section_000_00.tif'
				for file in files:
					pre_code1 = prev_name.split('_')[1]
					pre_code2 = prev_name.split('_')[-1][1]
					cur_code1 = file.split('_')[1]
					cur_code2 = file.split('_')[-1][1]
					assert code_comparisons(pre_code1, cur_code1, pre_code2, cur_code2), f'Channel {channel_num} warning: {file} does not follow previous file: {prev_name}'
					prev_name = file
			print(f'Channel {channel_num} successfully verified.')
		else:
			print(f'Channel {channel_num} does not exist. Try renaming channels to 1,2, and 3.')

class HiddenPrints:
	def __enter__(self):
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		sys.stdout.close()
		sys.stdout = self._original_stdout