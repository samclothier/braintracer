# Braintracer
Braintracer is a processing pipeline extension for the BrainGlobe API. It enables high-throughput processing with cellfinder, quantifies cell positions and produces figures for visualising cell distributions across datasets.

---
## Installation
First, install Anaconda or Miniconda on your machine.  
Open Anaconda Prompt.  
Create a Python environment and install braintracer:  
`conda create -n env_name python=3.10.6`  
`conda activate env_name`  
`pip install braintracer`  

View your downloaded BrainGlobe atlases with `brainglobe list`  
Install the 10um Allen mouse brain atlas: `brainglobe install -a allen_mouse_10um`  

Add your data into your working directory as follows:  
```
├── WorkingDirectory
│   ├── bt.bat
│   ├── bt_visualiser.ipynb
│   ├── DatasetName1
│   |   ├── SignalChannelName
│   |   |   ├── section_001_01
│   |   |   ├── section_001_02
│   |   ├── BackgroundChannelName
│   ├── DatasetName2
```

As you can see, for now `bt.bat` and `bt_visualiser.py` must be copied into the working directory.  
On Windows, these files are found here:  
`Users/USERNAME/miniconda3/envs/ENV_NAME/Lib/site-packages/braintracer/braintracer`  

It is also recommended to install CUDA for usage of the GPU in cellfinder:  
`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`  
Then confirm the GPU is detected by tensorflow:  
`python`  
`import tensorflow as tf`  
`tf.config.list_physical_devices('GPU')`    

To generate the braintracer directory structure inside `WorkingDirectory`:  
• Open Anaconda Prompt  
• Activate your environment: `conda activate env_name`  
• Navigate to `WorkingDirectory`  
• Run the braintracer pre-processing tool: `bt.bat`  
• The tool can then be closed - the directories are generated immediately  

---
## Usage
braintracer has two main workflows - pre-processing and visualisation.  

### Pre-processing
• Open Anaconda Prompt  
• Activate your environment: `conda activate env_name`  
• Navigate to `WorkingDirectory`  
• Run the braintracer pre-processing tool: `bt.bat`  
• Follow the instructions in the terminal  

If you already have a .csv from cellfinder containing cell coordinates, follow the above steps but answer `y` when asked `Do results already exist ready for copying?`  

### Visualisation
• Open Anaconda Prompt  
• Activate your environment: `conda activate env_name`  
• Navigate to `WorkingDirectory`  
• Open Jupyter with `jupyter-lab`  
• In the browser tab that appears, open `bt_visualiser.ipynb`  
• Play with your results and save figures all within Jupyter Notebook!  

---
## Sample data
If you don't have access to any raw data, you can use the sample data provided.  
Move the sample data files into the `WorkingDirectory\braintracer\cellfinder\` directory.  
You should then be able to explore this data with the bt_visualiser.ipynb notebook with `jupyter notebook` or `jupyter-lab`  

---
## Measure performance with ground truth
To assess the classifier's performance, you will need to generate ground truth data.  
Braintracer requires ground truth coordinates in atlas space, so these should be generated in napari with the cellfinder curation plugin.  
• Open napari with `napari`  
• Navigate to `dataset\\cellfinder_[]\\registration`  
• Load the signal channel `downsampled_standard_channel_0` and background channel `downsampled_standard`  
• Open the cellfinder curation plugin and select these layers as the signal and background channels  
• Click 'Add training data layers' and select some cells in the cells layer!  
• Select both cell layers and go to File... Save selected layer(s)  
• Save the file in the following format: `groundtruth_[].xml` (you must type .xml!) within `braintracer\\ground_truth`  

---
## Generate training data to improve the classifier
The classifier requires some feedback to be improved, or retrained.  
You can generate training data easily in napari.
• Open napari with `napari`  
• Drag the `dataset\\cellfinder_[]` folder onto the napari workspace  
• Drag the folders containing your signal and background channels  
• Move the signal and background channel layers down to the bottom of the layer manager (with signal channel above the background!)  
• Make the atlas layer (`allen_mouse_10um`) visible and decrease the opacity to reveal areas during curation  
• Go to `Plugins > cellfinder > Curation`  
• Set the signal and background image fields to your signal and background layers  
• Click `Add training data layers`  
• Select the layer you are interested in (`Cells` to mark false positives; `Non cells` for false negatives)  
• Select the magnifying glass to move the FOV such that the entire area to be curated is visible but cell markers can still large enough  
• You are then able to select the arrow icon to make markers selectable and not have to switch back and forth between the two tools  
• Begin curation from the caudal end (towards slice #0) and work your way through each slice, switching between the `Cells` and `Non cells` layers depending on the type of false label  
• Depending on the strategy, either review all cells (even confirming correct classifications by selecting `Mark as cell(s)` for the `Cells` layer or `Mark as non cell(s)` for the `Non cells` layer) or only the subset of cells that appear to be classified incorrectly  
• When finished, click `Save training data` and select the output folder  
• The plugin will create a file called `training.yml` and folders called `cells` and `non_cells` containing the TIFFs that the classifier will be shown  
• Additionally, select both training data layers and go to File... Save selected layer(s)  
• Save the file as `name.xml` (you must type .xml!)  
The YML file can then be used to retrain the network.  
