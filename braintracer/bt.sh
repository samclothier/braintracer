#!/bin/bash
#
# Copyright (C) 2021-2023  Sam Clothier
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

FOLDER="braintracer"

if [ ! -d "$FOLDER" ]; then mkdir "$FOLDER"; fi
if [ ! -d "$FOLDER/downsampled_data" ]; then mkdir "$FOLDER/downsampled_data"; fi
if [ ! -d "$FOLDER/cellfinder" ]; then mkdir "$FOLDER/cellfinder"; fi
if [ ! -d "$FOLDER/figures" ]; then mkdir "$FOLDER/figures"; fi
if [ ! -d "$FOLDER/registered_atlases" ]; then mkdir "$FOLDER/registered_atlases"; fi
if [ ! -d "$FOLDER/videos" ]; then mkdir "$FOLDER/videos"; fi

read -p "Dataset name: " dataset
read -p "Signal channels (include spaces for multiple): " s_chs
read -p "Background channel: " b
read -p "Trained network nickname: " nn

while true; do
    read -p "Do results already exist ready for copying? (y/n) " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) 
            read -p "Trained network path: " np
            read -p "Resolution (z x y; eg. 5 2 2): " res
            read -p "Threshold: " t
            read -p "Atlas resolution (um; eg. 10): " atres
            read -p "Also perform cell detection in background channel against the signal channel (only works for one signal channel)? (y/n) " reverse
            for s in $s_chs; do
                echo "Running brainmapper..."
                cd $dataset
                if [ -z "$np" ]; then
                    if [ ! -d "cellfinder_$s" ]; then
                        brainmapper -s $s -b $b -o cellfinder_${s} -v $res --orientation psr --threshold $t --atlas allen_mouse_${atres}um --ball-xy-size 10
                    else
                        echo "Results already exist for signal channel $s!"
                    fi
                else
                    if [ ! -d "cellfinder_${s}_${nn}" ]; then
                        brainmapper -s $s -b $b -o cellfinder_${s}_${nn} -v $res --orientation psr --threshold $t --atlas allen_mouse_${atres}um --trained-model $np --ball-xy-size 10
                    fi
                fi
                cd ..
                echo "Copying results..."
                cp $dataset/cellfinder_${s}_${nn}/registration/downsampled_standard_channel_0.tiff $FOLDER/downsampled_data/reg_${dataset}_${s}.tiff
		cp $dataset/cellfinder_${s}_${nn}/registration/downsampled_standard.tiff $FOLDER/downsampled_data/reg_${dataset}_${b}.tiff
		cp $dataset\cellfinder_${s}_${nn}/analysis/all_points.csv $FOLDER/cellfinder/cells_${dataset}_${nn}_${s}.csv
            done
            if [ "$reverse" == "y" ]; then
                echo "Running brainmapper..."
                cd $dataset
                if [ -z "$np" ]; then
                    if [ ! -d "cellfinder_$b" ]; then
                        brainmapper -s $b -b $s_chs -o cellfinder_${b} -v $res --orientation psr --threshold $t --atlas allen_mouse_${atres}um --ball-xy-size 10
                    else
                        echo "Results already exist for background channel $b!"
                    fi
                else
                    if [ ! -d "cellfinder_${b}_${nn}" ]; then
                        brainmapper -s $b -b $s_chs -o cellfinder_${b}_${nn} -v $res --orientation psr --threshold $t --atlas allen_mouse_${atres}um --trained-model $np --ball-xy-size 10
                    fi
                fi
                cd ..
                echo "Copying results..."
                cp $dataset/cellfinder_$b/analysis/all_points.csv $FOLDER/cellfinder/cells_$dataset_$nn_$b.csv
            fi
            exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "Copying files to directory/$FOLDER..."
cp $dataset/cellfinder_${s_chs}_${nn}/registration/downsampled_standard_channel_0.tiff $FOLDER/downsampled_data/reg_${dataset}_${s_chs}.tiff
cp $dataset/cellfinder_${s_chs}_${nn}/registration/downsampled_standard.tiff $FOLDER/downsampled_data/reg_${dataset}_${b}.tiff
cp $dataset/cellfinder_${s_chs}_${nn}/analysis/all_points.csv $FOLDER/cellfinder/cells_${dataset}_${nn}_${s_chs}.csv
cp $dataset/cellfinder_${s_chs}_${nn}/registration/registered_atlas.tiff $FOLDER/registered_atlases/atlas_${dataset}_${s_chs}.tiff
cp $dataset/cellfinder_${s_chs}/registration/downsampled_standard_channel_0.tiff $FOLDER/downsampled_data/reg_${dataset}_${s_chs}.tiff
cp $dataset/cellfinder_${s_chs}/registration/downsampled_standard.tiff $FOLDER/downsampled_data/reg_${dataset}_${b}.tiff
cp $dataset/cellfinder_${s_chs}/analysis/all_points.csv $FOLDER/cellfinder/cells_${dataset}_Unet_${s_chs}.csv
cp $dataset/cellfinder_${s_chs}/registration/registered_atlas.tiff $FOLDER/registered_atlases/atlas_${dataset}_${s_chs}.tiff
echo "Warnings may appear, but 4 files should have been copied successfully"
echo "Done!"
echo "[35mOpen bt_visualiser.ipynb in Jupyter to view the results.[0m"
sleep
jupyter-lab