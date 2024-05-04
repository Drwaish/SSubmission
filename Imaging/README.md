# CT Scan Segmentation - 3D

Segment the CT scans images, images in nifti format based on four classes:

These are the classes and their mask color.

```
edema (yellow), 
non-enhancing solid core (red), 
necrotic/cystic core (green), 
enhancing core (blue).
```

Must ensure copy this folder [model](https://drive.google.com/drive/folders/1qquW47fMUCl7zUs7LzP77ACdjj1YKumR?usp=sharing) and put in direcotry and pass the path in .env file.

# How to Setup

Create a conda environment
```
conda create -n heal_img
conda activate heal_img
```

Now Run the main file.
```
python main.py
```

**Note**: Testing images must be nii.gz format. Right now test the models from its testing images quota. Output images also available.


