ECHO OFF
start cmd.exe /c
title Braintracer

set folder=braintracer

if not exist %folder% mkdir %folder%
if not exist %folder%\downsampled_data mkdir %folder%\downsampled_data
if not exist %folder%\cellfinder mkdir %folder%\cellfinder
if not exist %folder%\atlas.tiff copy %USERPROFILE%\.brainglobe\allen_mouse_10um_v1.2\annotation.tiff %folder%\atlas.tiff
if not exist %folder%\structures.csv copy %USERPROFILE%\.brainglobe\allen_mouse_10um_v1.2\structures.csv %folder%

set /p dataset=Dataset name: 
set /p s=Signal channel: 
set /p b=Background channel: 

:choice
set /p c=Cellfinder already run? 
if /i "%c%" EQU "Y" goto :no_run
if /i "%c%" EQU "N" goto :run
goto :choice

:run
set /p res=Resolution (z x y; eg. 5 2 2): 
ECHO Running cellfinder...
cd %dataset%
if not exist cellfinder_%s% (
cellfinder -s %s% -b %b% -o cellfinder_%s% -v %res% --orientation psl --threshold 35 --atlas allen_mouse_10um --batch-size 128
) else (
ECHO Results already exist for signal channel %s%!
)
cd ..
ECHO Copying results...
copy %dataset%\cellfinder_%s%\registration\downsampled_standard.tiff %folder%\downsampled_data\reg_%s%_%dataset%.tiff
copy %dataset%\cellfinder_%s%\analysis\all_points.csv %folder%\cellfinder\cells_%s%_%dataset%.csv

ECHO Running cellfinder...
cd %dataset%
if not exist cellfinder_%b% (
cellfinder -s %b% -b %s% -o cellfinder_%b% -v %res% --orientation psl --threshold 35 --atlas allen_mouse_10um --batch-size 128
) else (
ECHO Results already exist for background channel %b%!
)
cd ..
ECHO Copying results...
copy %dataset%\cellfinder_%b%\registration\downsampled_standard.tiff %folder%\downsampled_data\reg_%b%_%dataset%.tiff
copy %dataset%\cellfinder_%b%\analysis\all_points.csv %folder%\cellfinder\cells_%b%_%dataset%.csv

goto :end

:no_run
ECHO Copying cellfinder output...
copy %dataset%\cellfinder_%s%\registration\downsampled_standard.tiff %folder%\downsampled_data\reg_%s%_%dataset%.tiff
copy %dataset%\cellfinder_%s%\analysis\all_points.csv %folder%\cellfinder\cells_%s%_%dataset%.csv
copy %dataset%\cellfinder_%b%\registration\downsampled_standard.tiff %folder%\downsampled_data\reg_%b%_%dataset%.tiff
copy %dataset%\cellfinder_%b%\analysis\all_points.csv %folder%\cellfinder\cells_%b%_%dataset%.csv

:end
ECHO Done!
ECHO [35mOpen bt_visualiser.ipynb in Jupyter to view the results.[0m
PAUSE
jupyter notebook