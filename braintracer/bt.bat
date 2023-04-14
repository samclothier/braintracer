goto comment
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
:comment

ECHO OFF
start cmd.exe /c
title Braintracer

set folder=braintracer

if not exist %folder% mkdir %folder%
if not exist %folder%\downsampled_data mkdir %folder%\downsampled_data
if not exist %folder%\cellfinder mkdir %folder%\cellfinder
if not exist %folder%\figures mkdir %folder%\figures
if not exist %folder%\registered_atlases mkdir %folder%\registered_atlases
if not exist %folder%\videos mkdir %folder%\videos

set /p dataset=Dataset name: 
set /p s_chs=Signal channels (include spaces for multiple): 
set /p b=Background channel: 
set /p nn=Trained network nickname: 

:choice
set /p c=Do results already exist ready for copying? 
if /i "%c%" EQU "Y" goto :no_run
if /i "%c%" EQU "N" goto :run
goto :choice

:run
set /p np=Trained network path: 
set /p res=Resolution (z x y; eg. 5 2 2): 
set /p orient=Orientation (post-ant, sup-inferior, l-r; eg. psr): 
set /p t=Threshold (std-dev; eg. 20): 
set /p bxy=Ball xy size (px; eg. 10): 
set /p atres=Atlas resolution (um; eg. 10): 
set /p reverse=Also perform cell detection in background channel against the signal channel (only works for one signal channel)? 

(for %%s in (%s_chs%) do (
	ECHO Running cellfinder...
	cd %dataset%
	if [%np%] == [] ( :: if no trained network
		if not exist cellfinder_%s% (
			cellfinder -s %%s -b %b% -o cellfinder_%%s -v %res% --orientation %orient% --threshold %t% --atlas allen_mouse_%atres%um --ball-xy-size %bxy%
		) else (
			ECHO Results already exist for signal channel %%s!
	)) else (
		if not exist cellfinder_%s%_%nn% (
			cellfinder -s %%s -b %b% -o cellfinder_%%s_%nn% -v %res% --orientation %orient% --threshold %t% --atlas allen_mouse_%atres%um --trained-model %np% --ball-xy-size %bxy%
	))
	cd ..
	ECHO Copying results...
	copy %dataset%\cellfinder_%%s_%nn%\registration\downsampled_standard_channel_0.tiff %folder%\downsampled_data\reg_%dataset%_%%s.tiff
	copy %dataset%\cellfinder_%%s_%nn%\registration\downsampled_standard.tiff %folder%\downsampled_data\reg_%dataset%_%b%.tiff
	copy %dataset%\cellfinder_%%s_%nn%\analysis\all_points.csv %folder%\cellfinder\cells_%dataset%_%nn%_%%s.csv
))
:: do the same but without copying the downsampled images (they are identical to the forward run)
if reverse EQU "Y" (
	ECHO Running cellfinder...
	cd %dataset%
	if [%np%] == [] ( :: if no trained network
		if not exist cellfinder_%b% (
			cellfinder -s %b% -b %s_chs% -o cellfinder_%b% -v %res% --orientation %orient% --threshold %t% --atlas allen_mouse_%atres%um --ball-xy-size %bxy%
		) else (
			ECHO Results already exist for background channel %b%!
	)) else (
		if not exist cellfinder_%b%_%n_n% (
			cellfinder -s %b% -b %s_chs% -o cellfinder_%b%_%nn% -v %res% --orientation %orient% --threshold %t% --atlas allen_mouse_%atres%um --trained-model %np% --ball-xy-size %bxy%
	))
	cd ..
	ECHO Copying results...
	copy %dataset%\cellfinder_%b%\analysis\all_points.csv %folder%\cellfinder\cells_%dataset%_%nn%_%b%.csv
)

goto :end

:no_run :: try copying from folder with and without custom network
ECHO Copying files to directory\%folder%...
copy %dataset%\cellfinder_%s_chs%_%nn%\registration\downsampled_standard_channel_0.tiff %folder%\downsampled_data\reg_%dataset%_%s_chs%.tiff
copy %dataset%\cellfinder_%s_chs%_%nn%\registration\downsampled_standard.tiff %folder%\downsampled_data\reg_%dataset%_%b%.tiff
copy %dataset%\cellfinder_%s_chs%_%nn%\analysis\all_points.csv %folder%\cellfinder\cells_%dataset%_%nn%_%s_chs%.csv
copy %dataset%\cellfinder_%s_chs%_%nn%\registration\registered_atlas.tiff %folder%\registered_atlases\atlas_%dataset%_%s_chs%.tiff
copy %dataset%\cellfinder_%s_chs%\registration\downsampled_standard_channel_0.tiff %folder%\downsampled_data\reg_%dataset%_%s_chs%.tiff
copy %dataset%\cellfinder_%s_chs%\registration\downsampled_standard.tiff %folder%\downsampled_data\reg_%dataset%_%b%.tiff
copy %dataset%\cellfinder_%s_chs%\analysis\all_points.csv %folder%\cellfinder\cells_%dataset%_Unet_%s_chs%.csv
copy %dataset%\cellfinder_%s_chs%\registration\registered_atlas.tiff %folder%\registered_atlases\atlas_%dataset%_%s_chs%.tiff
ECHO Warnings may appear, but 4 files should have been copied successfully
:end
ECHO Done!
ECHO [35mOpen bt_visualiser.ipynb in Jupyter to view the results.[0m
PAUSE
jupyter-lab