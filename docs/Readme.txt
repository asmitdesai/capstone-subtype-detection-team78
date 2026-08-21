To reproduce analysis:

Use Docker Image andreaphp/datascience:10_05_2023 with GPU access.
We used for our analysis an NVIDIA Quadro RTX 5000 GPU

Recreate the conda enviroment with conda create --name <env_name> --file conda/spec_list.txt, after that you will use your new conda env as a jupyter kernel.

Download GDC data using the GDC download tool and the provided manifest in the "GDC Data Download" folder

Change the .env file to reflect where you put code and data in the code
1. CODE_PATH - the absolute path of code subfolder.
2. GDC_PATH - the absolute path of TCGA folder where you downloades cases using GDC donwload tool.
3. RAW_THYROID_PATH - the absolute path of rawdata for thyroid dataset creation (for reference only, npy files are provided on zenodo).
4. THYROID_PATH - the absolute path of dataset folder.

For your convenience the dataset folder contains a dataframe with relative paths for GDC cases for use with the Neural Network Data Generator Class.

If you want to recreate Thyroid Datasets, run code/preprocessing/Thyroid.ipynb using the conda kernel.




