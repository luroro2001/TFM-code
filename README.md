# Foundation Model for Solar Spectropolarimetry

This master's thesis (TFM) presents the development and evaluation of a foundation model for solar spectropolarimetry employing contrastive training. The model consists of two residual network encoders, one for Stokes profiles and one for physical atmospheric models, trained together to project both modalities into a shared latent space using a CLIP style contrastive loss, as well as reconstruction losses from the two corresponding decoders. The training database consists of synthetic profiles of the Fe I doublet at 630.15 and 630.25 nm, computed from perturbations of semi-empirical solar atmospheric models, covering a wide range of physical conditions that are representative of different solar regions.

<img width="1541" height="893" alt="Image" src="https://github.com/user-attachments/assets/afd27cf2-10f9-42d6-99d6-df799ef9c85c" />

---

## Table of contents

- [Project Structure](#project-structure)
- [How Does It Work?](#how-does-it-work)
- [Downstream Tasks](#downstream-tasks)
- [Requirements](#requirements)

---

## Project structure

There are more files included in this repository, but below I only show the ones that are actually used in the model.

```
TFM/code/
│ 
├── database # NOT INCLUDED IN THIS REPOSITORY
│   ├── good_profiles_testing.npy
│   ├── good_profiles_training.npy
│   ├── good_profiles_validation.npy
│   ├── models_testing.h5
│   ├── models_training.h5
│   ├── models_validation.h5
│   ├── stokes_testing.h5
│   ├── stokes_training.h5
│   └── stokes_validation.h5
│ 
├── modules 
│   ├── dataset.py # to load the data
│   ├── normalize.py # normalization and denormalization fucntions
│   ├── resnet.py # defines the residual networks
│   └── symlog.py # symmetric logarithm transform
│   
├── train 
│   ├── weights/ # saved model checkpoints. NOT INCLUDED/UPDATED HERE
│   ├── train_clip.py # CLIP style contrastive training
│   └── conf.yaml # hyperparameters file
│ 
├── validate
│   ├── validate.py # validation script
│   └── validate_trial.py # only for testing, not part of the model
│
└── README.md
```
---

## How does it work?

The inner workings of the model can be separated into three phases based on the main scripts: data loading and preprocessing, contrastive training and validation.

### 1. Loading the data (`modules/dataset.py`)

The `Dataset` loads the paired data samples from two HDF5 files: one that contains synthetic Stokes profiles (I, Q, U, V) sampled over 112 wavelength points, and one that contains the corresponding physical atmospheric models, described by temperature, microturbulent velocity, line of sight velocity and the three components of the magnetic field (over 80 atmospheric depth layers). Then, a `.npy` file is used to select only the good (physically coherent) profiles from the full database.

Before being passed to the network, each parameter is reescaled to the range [-1,1] using fixed bounds (specified on Table 1 of the TFM). The Stokes parameters Q, U, and V are also normalized by I, as is typically done in solar spectropolarimetry. Additionally, Gaussian noise can be added to the Stokes profiles to bring them closer to what one might expect from oberservations. In this work, $\sigma=10^{-3}$ (in terms of continuum intensity) was added.

### 2. Training the model (`train/train_clip.py`)

The training parameters are configured using the file `conf.yaml`, which establishes the noise level, batch size, learning rate, number of epochs, etc. The `Training` class makes two residual network encoders (using `modules/resnet.py`), one for the Stokes profiles and one for the physical modles. Both project their respective inputs into a shared latent space of dimension `latent_dim` (set in the configuration file). Similarly, two decoders are used to reconstruct each modality from its latent representation.

At each training step, both modalities are flattened, encoded, and L2-normalized to lie on the unit hypersphere. The main goal is the contrastive loss, which encourages matched pairs to have close representations while pushing unmatched pairs apart. When decoders are enabled, MSE reconstruction losses for both modalities are added to the total loss, weighted by coefficients (which were set empirically to keep the finallosses at comparable orders of magnitude). The optimizer used is Adam with a cosine annealing learning rate schedule. Model checkpoints are saved after every epoch, and the best validation checkpoint is tracked.

### 3. Validation (`validate/validate.py`)

The `Testing` loads a saved checkpoint and runs the model on the test set. For each sample, Stokes profiles and physical models are independently encoded
and normalized to obtain their latent representations. Different methods make it possible to: encode and decode (reconstruct) each type of data and compare the output to the ground truth; visualize a 2D projection of the latent space using t-SNE (to assess the sucess of the contrastive learning); perform fast Stokes synthesis and inversion (see following section).

---

## Downstream tasks

The model allows the execution of two downstream applications:

### Fast Stokes inversion

The fast Stokes inverter implements the cross-modal path that is illustrated in the figure below. Given a Stokes profiles as input, the Stokes encoder projects it into the shared latent space, producing a latent vector $\textbf{z}_s \in \mathbb{R}^{64}$. This vector is then decoded by the model decoder to produce an estimation of the atmospheric stratification for each of the physical parameters. The main advantage of the approach is that it avoids the iterative nature of classical inversion codes; once the model is trained, the inversion of a profile only requires two passes through two neural networks, so it can be carried out faster than in traditional methods.

<img width="2718" height="841" alt="Image" src="https://github.com/user-attachments/assets/47cb75c9-ad89-4e79-b9f7-e515bb6d8294" />

### Fast Stokes synthesis

The fast Stokes synthesizer implements the cross-modal path that is illustrated in the figure below. Given a physical atmospheric model as input, the model encoder projects it into the shared latent space, producing a latent vector $\textbf{z}_m \in \mathbb{R}^{64}$, which is then decoded by the Stokes decoder to produce a synthetic Stokes profile. As with the inverter, this path was not part of the explicit training, so its performance reflects the quality of the contrastive alignment between the two encoders. The synthesizer represents the forward problem: given known physical conditions, producing the corresponding observational parameters.

<img width="2716" height="788" alt="Image" src="https://github.com/user-attachments/assets/f39e6fec-7bd2-4b64-9e0c-eee9c984e5d9" />

---

## Requirements

The model was implemented in Python using the following packages:

- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyTorch](https://pytorch.org/)
- [h5py](https://www.h5py.org/)
- [scikit-learn](https://scikit-learn.org/)
- [einops](https://github.com/arogozhnikov/einops)
- [tqdm](https://github.com/tqdm/tqdm)
- [PyYAML](https://pyyaml.org/)