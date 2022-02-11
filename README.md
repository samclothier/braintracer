# Braintracer
Braintracer is a processing pipeline extension for the BrainGlobe API. It enables high-throughput processing with cellfinder, quantifies cell positions and produces figures for visualising cell distributions across datasets.

---
Installation:  
First, install Anaconda on your machine.  
Within the base environment: `pip install cellfinder`  
Then, `pip install bg-atlasapi`  
View your current BrainGlobe atlases with `brainglobe list`  
Ensure you have the 10um atlas installed.  
If you don't, run `brainglobe install allen_mouse_10um`  
This should download the atlas to `%USERPROFILE%\.brainglobe\allen_mouse_10um_v1.2\annotation.tiff`, along with the file `structures.csv`  

Save the following files at `%USERPROFILE%\anaconda3\Lib\site-packages\braintracer`:  
• `__init__.py`  
• `analysis.py`  
• `file_management.py`  
• `plotting.py`  

Add your data into your working directory as follows:  
```
├── WorkingDirectory
│   ├── bt.bat
│   ├── bt_visualiser.py
│   ├── DatasetName1
│   |   ├── SignalChannelName
│   |   |   ├── section_001_01
│   |   |   ├── section_001_02
│   |   ├── BackgroundChannelName
│   ├── DatasetName2
```

`SignalChannelName` and `BackgroundChannelName` should be folders containing the images that make up the stack.  

Usage:  
• Open Anaconda Prompt  
• Type `bt.bat` while in `WorkingDirectory`  
• Follow the instructions in the terminal  
• Play with your results and save figures within Jupyter Notebook!  

---
If you don't have access to any raw data, you can use the sample data provided.  
Set up everything as above bar the `DatasetName` directories and contained files.  
Attempt to run `bt.bat` from `WorkingDirectory` in the terminal once, so that the atlas and other files can be organised correctly.  
Move the sample data files into the `braintracer\cellfinder\` directory.  
You should then be able to explore this data with the bt_visualiser.ipynb notebook with `jupyter notebook`

---
To assess the classifier's performance, you will need to generate ground truth data.
Braintracer requires ground truth coordinates in atlas space, so these should be generated in napari with the cellfinder curation plugin.
• Open napari with `napari`
• Navigate to `dataset\\cellfinder_[]\\registration`
• Load the signal channel `downsampled_standard_channel_0` and background channel `downsampled_standard`
• Open the cellfinder curation plugin and select these layers as the signal and background channels
• Click 'Add training data layers' and select some cells in the cells layer!
• Select both cell layers and go to File... Save selected layer(s)
• Save the file in the following format: `groundtruth_[].xml` (you must type .xml!) within `braintracer\\ground_truth`