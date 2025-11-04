# Project structure

```
TFM/code/
│ 
├── database 
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
│   ├── dataset.py
│   ├── encoder_decoder.py
│   ├── encoding.py
│   ├── mlp.py
│   ├── normalize.py
│   ├── resnet.py
│   ├── siren.py
│   └── symlog.py
│   
├── train 
│   ├── weights
│   ├── train_clip.py
│   └── train_vicreg.py
│ 
├── validate
│   ├── validate.py
│   └── validate_trial.py
│ 
├── validate_old
│   ├── deconvolution_hinode.py
│   ├── doplots_clip.py
│   ├── invert_clip.py
│   ├── invert_vicreg.py
│   ├── invert.py
│   ├── noise_svd.py
│   ├── validate_2d.py
│   ├── validate.py
│   ├── view_models.py
│   └── view.py
│
└── README.md
```

In the following sections, each script that makes up the code will be explained.

# **`database`**

**PENDING**: add explanation of what each file is.

# **`modules directory`**

## **`dataset.py`**

Starts off by defining two helper functions to normalize and denormalize data. 

**normalize_input** scales input data $x$ from the range $[xmin, xmax]$ to $[-1, 1]$. This is because neural networks work better when the inputs are normalized. The formula is simply:

$x_{norm}=2 \cdot \frac{x-x_{min}}{x_{max}-x_{min}} - 1$

**denormalize_input** reverts normalized values from $[-1, 1]$ back to the original range $[xmin, xmax]$.

### **`Dataset class`**

Provides both Stokes profiles abd physical model parameters for training.

Its inputs are:
- `filename_stokes`: HDF5 file containing Stokes I, Q, U, V profiles.
- `filename_model`: HDF5 file containing the physical model parameters (logtau, T, Pe, vmic, v, Bx, By, Bz).
- `good_profiles_filename`: .npy file indexing "good" profiles to use.
- `n_training`: Optional, number of examples to train on.
- `noise`: Amount of gaussian noise to add for augmentation. 

#### `__init__(self, etc)`

- opens the hdf5 files containing the data and then loads indices of good profiles (ind). Filters out only the good profiles using 'ind'.
- sets dataset length: either all available samples or a subset (`n_training`).
- defines normalization bounds for all Stokes parameters (I, Q, U, V) and model parameters (T, vmic, v, Bx, By, Bz). These are later used in `normalize_input()`.

#### `__getitem__(self, index)`

This method is called by Pytorch for eahc sample during training.

- Extracts the Stokes parameters for the given index and adds noise to them for data augmentation, which helps the model generalize better. Also converts the I, Q, U, V into fractions Q/I, U/I, V/I, as is standard in spectropolarimetry. 
- Rescales all inputs with `normalize_input` into `[-1,1]`for neural network stability. Then, does the same thing for model parameters (T, vmic, v, Bx, By, Bz).
- Returns two arrays per sample: out_stokes and out_model.

#### `__len__(self, index)`

Required by Pytorch, returns the number of training samples in the dataset.

### **`DatasetHinode class`**
Similar to Dataset but tailored for real Hinode solar data (no physical model parameters). I imagine this is used during validation. The key differences are:
- No physical model parameters (because it's used for indference, not supervised training).
- Extracts subsets from 2D solar images.
- This is used for validation or for checking inversion capacity with real observations.

Finally, the script does a quick test to see if the dataset loads correctly and prints out the first sample and te¡hen the data size.

# **`train directory`**

## **`train_clip.py'**

This script uses a CLIP contrastive learning training loop that learns to align two modalities:
- Stokes profiles: 4x112 wavelengths, flattened to 4*112 (4 parameters sampled in 112 points).
- Models (physical parameters): 6x80 layers, flattened to 6*80 (6 parameters sampled in 80 points).
It uses two encoders (encoder_stokes and encoder_models) to map these into a common latent space (latent_dim) and optimizes a symmetric contrastives loss (CLIP). Optionally it also uses decoders and saves checkpoints.

**Note**: The code starts off by defining `merge_images', which isn't actually used anywhere.

### **'CLIPLoss'**

Simple contrastive loss.
**PENDING**: get a deeper understanding.

### **'CLIPLossMultiModal'**

Supports multi-modal contrastive learning between 'n' modalities (implemented for n=2 and n=3).

### **'Training'** 