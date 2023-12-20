
from tqdm.auto import tqdm
import pennylane as qml
import numpy as np
import rasterio
import torch
import glob
import os

from hqm.circuits.angleencoding import BasicEntangledCircuit
from hqm.layers.quanvolution import Quanvolution2D

def reject_outliers(s1, perc = 95):
    p = np.percentile(s1, perc)
    s1 = np.clip(s1, 0.0, p)
    return s1.astype(np.float32)

def min_max(s1):
    return (s1 - s1.min()) / (s1.max() - s1.min()).astype(np.float32)

def add_speckle(s1, looks = 4):
    # Numpy Gamma Distribution is defined in the shape-scale form
    # Mean 1 Var 1/looks
    gamma_shape = looks
    gamma_scale = 1/looks
    noise = np.random.gamma(gamma_shape, 
                            gamma_scale, 
                            s1.shape)
    s1 = s1*noise
    return s1.astype(np.float32), noise.astype(np.float32)

def process(quanv, dataset):
    os.makedirs(os.path.join(dataset.replace('3', '4'), 'input'), exist_ok=True)
    os.makedirs(os.path.join(dataset.replace('3', '4'), 'ground'), exist_ok=True)

    for i, path in enumerate(tqdm(glob.glob(os.path.join(dataset, "*.tif")))):
        with rasterio.open(path) as src:
            speckle_free = src.read()

        speckle_full, _ = add_speckle(speckle_free, looks=4)
        speckle_full    = reject_outliers(speckle_full)
        speckle_full    = min_max(speckle_full)
        speckle_free    = reject_outliers(speckle_free)
        speckle_free    = min_max(speckle_free)

        with torch.no_grad():
            speckle_full = quanv(torch.tensor(speckle_full[None,...]))[0,...].numpy()
        
        np.save(os.path.join(dataset.replace('3', '4'), 'input', f'patch_{i}.npy'), speckle_full)
        np.save(os.path.join(dataset.replace('3', '4'), 'ground', f'patch_{i}.npy'), speckle_free)

        
if __name__== '__main__':

    ROOT  = os.path.join('datasets', 'dataset_v3')

    N_QUBITS      = 9
    N_LAYERS      = 1
    FITLERS       = 9
    KERNELSIZE    = 3
    STRIDE        = 1

    dev           = qml.device('lightning.qubit', wires=N_QUBITS)
    qcircuit      = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    quanv         = Quanvolution2D(qcircuit=qcircuit, filters=FITLERS, kernelsize=KERNELSIZE, stride=STRIDE, padding='same')

    process(quanv, dataset=os.path.join(ROOT, 'training'))
    process(quanv, dataset=os.path.join(ROOT, 'validation'))
    process(quanv, dataset=os.path.join(ROOT, 'testing'))