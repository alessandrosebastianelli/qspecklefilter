
from rasterio.windows import Window
from tqdm.auto import tqdm
import numpy as np
import rasterio
import glob
import os


IMG_SHAPE  = (256, 256, 1)
PATCH_SIZE = (64, 64)
ROWS       = IMG_SHAPE[0]//PATCH_SIZE[0]
COLUMNS    = IMG_SHAPE[1]//PATCH_SIZE[1]


def patchify(dataset):
    os.makedirs(dataset.replace('2', '3'), exist_ok=True)

    for path in tqdm(glob.glob(os.path.join(dataset, "*.tif"))):
        with rasterio.open(path) as src:
            for row in range(ROWS):
                for col in range(COLUMNS):

                    save_path = path.replace('2', '3')
                    save_path = save_path.replace('.tif', f'_{row}_{col}.tif')

                    window = Window(col*PATCH_SIZE[0], row*PATCH_SIZE[1], PATCH_SIZE[0], PATCH_SIZE[1])
                    windowed_data = src.read(window=window)
                    meta  = src.meta.copy()
                    # Update metadata based on the new window size and location
                    meta['width'], meta['height'] = window.width, window.height
                    meta['transform'] = src.window_transform(window)

                    with rasterio.open(save_path, 'w', **meta) as dst:
                        dst.write(windowed_data)
                
if __name__== '__main__':

    ROOT  = os.path.join('datasets', 'dataset_v2')

    patchify(dataset=os.path.join(ROOT, 'training'))
    patchify(dataset=os.path.join(ROOT, 'validation'))
    patchify(dataset=os.path.join(ROOT, 'testing'))