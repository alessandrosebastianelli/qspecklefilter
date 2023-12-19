# Quantum Speckle Filtering



## History



## Dataset
[The original dataset can ba found here](https://github.com/alessandrosebastianelli/sentinel_1_GRD_dataset)

From the original dataset several smaller patches have been extracted to facilitate the training process of our quantum based solution.

To extract patches, firstly copy ```dataset_v2``` into ```datasets``` folder

```
qspecklefilter/
├─ datasets/
│  ├─ dataset_v2/
│  │  ├─ training/
│  │  ├─ testing/
│  │  ├─ validation/
├─ ...../
├─ scripts/
├─ README.md
├─ .girignore

```

then run 

```
python scripts/extract_patches.py
```

the new dataset will be located at ```datasets/ dataset_v3```
