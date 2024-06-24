# CT Scan Segmentation - 3D

Segment the CT scans images, images in nifti format based on four classes:

These are the classes and their mask color.

```
edema (yellow), 
non-enhancing solid core (red), 
necrotic/cystic core (green), 
enhancing core (blue).
```

Must ensure copy this folder [model](https://drive.google.com/drive/folders/1qquW47fMUCl7zUs7LzP77ACdjj1YKumR?usp=sharing) 

and put in directory and pass the path in .env file. Kindly following file structure

## .env file structure
```
ASSETS_DIR=<Directory Path> 
```

# How to Setup

Create a conda environment
```
conda create -n heal_img
conda activate heal_img
```

# Install Dependencies
```
pip install -r requirements.txt
```

Now Run the main file.
```
python main.py
```

**Note**: Testing images must be nii.gz format. Right now test the model from one of the testing images . Output images also available.


