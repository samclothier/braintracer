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
if not exist %folder%\structures.csv copy %USERPROFILE%\.brainglobe\allen_mouse_10um_v1.2\structures.csv %folder%

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
set /p t=Threshold: 
set /p atres=Atlas resolution (um; eg. 10): 
set /p reverse=Also perform cell detection in background channel against the signal channel (only works for one signal channel)? 

(for %%s in (%s_chs%) do (
	ECHO Running cellfinder...
	cd %dataset%
	if [%np%] == [] ( :: if no trained network
		if not exist cellfinder_%s% (
			cellfinder -s %%s -b %b% -o cellfinder_%%s -v %res% --orientation psr --threshold %t% --atlas allen_mouse_%atres%um --batch-size 64 --ball-xy-size 10
		) else (
			ECHO Results already exist for signal channel %%s!
	)) else (
		if not exist cellfinder_%s%_%nn% (
			cellfinder -s %%s -b %b% -o cellfinder_%%s_%nn% -v %res% --orientation psr --threshold %t% --atlas allen_mouse_%atres%um --batch-size 64 --trained-model %np% --ball-xy-size 10
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
			cellfinder -s %b% -b %s_chs% -o cellfinder_%b% -v %res% --orientation psr --threshold %t% --atlas allen_mouse_%atres%um --batch-size 64 --ball-xy-size 10
		) else (
			ECHO Results already exist for background channel %b%!
	)) else (
		if not exist cellfinder_%b%_%n_n% (
			cellfinder -s %b% -b %s_chs% -o cellfinder_%b%_%nn% -v %res% --orientation psr --threshold %t% --atlas allen_mouse_%atres%um --batch-size 64 --trained-model %np% --ball-xy-size 10
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