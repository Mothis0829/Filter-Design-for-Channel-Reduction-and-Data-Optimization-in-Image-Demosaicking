# Filter-Design-for-Channel-Reduction-and-Data-Optimization-in-Image-Demosaicking

## Dataset Preparation
### Tokyo Tech Dataset
Download the data from this link. Move its extracted folder matto dataset/TokyoTech/ and then run the script dataset/TokyoTech/generate_hdf5.py to convert the data to hdf5 format.

### TCPDNet Pre-trained Models
In this Dropbox link you will find 2 folders:

**CPDNetDataset:** The pre-trained models on CPDNet dataset.

**TokyoTechDataset:** The pre-trained models on Tokyo Tech dataset.
The result in the paper is the average of 5 trained models at the epoch 200th. Therefore, in each dataset folder, you will find 5 sub-folders each of which corresponds to one trained model.
