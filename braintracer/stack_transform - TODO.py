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

import os, sys
from skimage.io import imread, imread_collection, concatenate_images
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import ndimage
from matplotlib import colors
from tqdm import tqdm

script_dir = os.getcwd()

stack1 = imread_collection('1/*.tif')
stack2 = imread_collection('3/*.tif')

print(len(stack1), stack1[0].shape)

foreg = imread('reg_SC033_r.tiff')
backg = imread('reg_SC033_b.tiff')
print(foreg.shape, backg.shape)

def analyse_fluorescence():
        def fit_model(model, y, x, name, ax=None):
            X = np.array(x).reshape(len(x),1)
            line_X = np.arange(min(x), max(x))[:, np.newaxis]
            model.fit(X, y)
            line_y = model.predict(line_X)
            try:
                if ax != None:
                    ax.plot(line_X, line_y, color='r', linewidth=1, label=name)
                return model.coef_, model.intercept_
            except Exception:
                if ax != None:
                    ax.plot(line_X, line_y, color='m', linewidth=1, label=name)
                return model.estimator_.coef_, model.estimator_.intercept_

        def get_subtraction_coeffs(ax):
            ref_vals = backg.flatten()[::10_000] # every 1000 values of the array
            sig_vals = foreg.flatten()[::10_000] # same coordinates chosen from both image stacks

            if ax != None:
                ax.hist2d(ref_vals, sig_vals, bins=500, cmap=plt.cm.jet, norm=colors.LogNorm())
                ax.set_ylim(0,7000)
                ax.set_xlim(0,2000)

            ransac = linear_model.RANSACRegressor(max_trials=30)
            m, c = fit_model(ransac, sig_vals, ref_vals, 'RANSAC', ax=ax)
            m = m[0] # extract from array
            return m, c
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
        plt.style = plt.tight_layout()

        binary_thresh = np.sqrt(1000)
        med_filter_iters = 3

        m, c = get_subtraction_coeffs(ax1)
        plt.savefig(f'fit.png', dpi=600, bbox_inches='tight')
        print(f'Subtraction parameters are: m={m}, c={c}')

        for i in tqdm(range(len(stack1))):
            subtracted_im = stack1[i] - ((stack2[i] * m) + c)
            sqt_im = np.sqrt(subtracted_im)
            bin_im = np.where(sqt_im > binary_thresh, 1, 0)
            mfl_im = ndimage.median_filter(bin_im, med_filter_iters)

            binary_im = mfl_im.astype(bool)

            file_name = stack1.files[i][1:-4]
            path = 'transform'
            dir_path = os.path.join(script_dir, path)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            dest = dir_path + file_name
            #print(dest)
            np.save(dest, binary_im)

analyse_fluorescence()