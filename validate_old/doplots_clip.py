import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
try:
    from nvitop import Device
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
import matplotlib.pyplot as pl
import sys
sys.path.append('../modules')
import datasets
import normalize
import symlog
import resnet
import glob
import pyqtgraph as pg
import platform
import h5py
import pynndescent
import os
import pickle

"""
This script defines a Training class for zero-shot inversion of solar Stokes images using 
pre-trained neural networks. It loads ResNet-based encoders and decoders, encodes Stokes images 
into latent vectors, performs nearest-neighbor (k-NN) search in latent space, reconstructs both 
Stokes images and physical models (temperature, velocity, magnetic fields), and generates comparison plots.

The workflow is designed to either process synthetic data (project_obs) or Hinode-reconstructed data (project_hinode).
"""    
    
class Training(object):
    def __init__(self, checkpoint, gpu, batch_size):

        print(f"Loading model {checkpoint}")
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        self.loss = chk['loss']
        self.loss_val = chk['loss_val']

        chk = torch.load(checkpoint+'.best', map_location=lambda storage, loc: storage)

        self.config = chk['config']

        self.cuda = torch.cuda.is_available()
        self.gpu = gpu        
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            self.handle = Device.all()[self.gpu]
            
            print("Computing in {0} : {1}".format(self.device, self.handle.name()))
        
        self.batch_size = batch_size
        
        # Model        
        self.encoder_models = resnet.ResidualNet(in_features=6*51, 
                      out_features=self.config['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=0.1,
                      use_batch_norm=True).to(self.device)
        
        self.encoder_stokes = resnet.ResidualNet(in_features=4*112, 
                      out_features=self.config['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=0.1,
                      use_batch_norm=True).to(self.device)
        
        self.decoder_models = resnet.ResidualNet(in_features=self.config['latent_dim'],
                      out_features=6*51,
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=0.1,
                      use_batch_norm=True).to(self.device)
        
        self.decoder_stokes = resnet.ResidualNet(in_features=self.config['latent_dim'],
                    out_features=4*112,
                    hidden_features=self.config['mlp']['n_hidden_mlp'],
                    num_blocks=self.config['mlp']['num_layers_mlp'],
                    activation=F.gelu,
                    dropout_probability=0.1,
                    use_batch_norm=True).to(self.device)
        

        print("Setting weights of the model...")        
        self.encoder_models.load_state_dict(chk['encoder_models_dict'])
        self.encoder_stokes.load_state_dict(chk['encoder_stokes_dict'])
        self.decoder_models.load_state_dict(chk['decoder_models_dict'])
        self.decoder_stokes.load_state_dict(chk['decoder_stokes_dict'])

        self.stats = chk['stats']
        
        self.encoder_stokes.eval()
        self.encoder_models.eval()
        self.decoder_stokes.eval()
        self.decoder_models.eval()    

    def denormalize_models(self, models):        
        models = models * self.stats['std_models'][:, None, None] + self.stats['mn_models'][:, None, None]
        models[2, ...] = symlog.inv_symlog(models[2, ...])
        models[4:, ...] = symlog.inv_symlog(models[4:, ...])

        models[2, ...] = np.log10(models[2, ...])
        models[3, ...] = models[3, ...] * 1e-5
        return models
    
    def read_db(self):
        
        self.training_dataset = datasets.Dataset1D(self.config['training_set'], 
                    pctx=[0,80], 
                    pcty=[0,100], 
                    stats=None, 
                    step=1)
        
        print("Stacking Stokes...")
        stokes_all = []
        for i in range(4):                        
            stokes_all.append(self.training_dataset.stokes[i].reshape((4, 112, -1)))
        self.stokes_all = np.concatenate(stokes_all, axis=-1)
            
        print("Stacking models...")        
        models_all = []
        for i in range(4):            
            models = self.training_dataset.models[i][:, ...]
            models_all.append(models.reshape((7, 51, -1)))
        self.models_all = np.concatenate(models_all, axis=-1)        

        print("Encoding models...")
        self.z_models, self.z_normal_models = self.project(self.models_all[1:, ...], 256, which='models')        

        if os.path.isfile('pynnindex'):
            print("Reading nearest neighbors...")
            with open('pynnindex','rb') as f:
                self.index = pickle.load(f)
        else:
            print("Building nearest neighbors...")
            # We use L2 normalized vectors which gives the same ordering as cosine similarity     
            self.index = pynndescent.NNDescent(self.z_normal_models)
            self.index.prepare()
            with open('pynnindex','wb') as f:
                pickle.dump(self.index,f)                                
        return
    
    def project(self, array, batch_size, which='stokes'):
        ns, nl, n = array.shape
        array_flat = array.reshape((ns*nl, n)).T
        array_flat = np.array_split(array_flat, batch_size, axis=0)
        z_all = []
        z_normal_all = []
        with torch.no_grad():
            for i in tqdm(range(len(array_flat))):
                tmp = torch.tensor(array_flat[i].astype('float32')).to(self.device)
                if (which == 'stokes'):
                    z = self.encoder_stokes(tmp)                    
                if (which == 'models'):
                    z = self.encoder_models(tmp)
                
                z_all.append(z.cpu().numpy())

                z = F.normalize(z, dim=-1)

                z_normal_all.append(z.cpu().numpy())
                
        z_all = np.concatenate(z_all, axis=0)
        z_normal_all = np.concatenate(z_normal_all, axis=0)
        
        return z_all, z_normal_all
    
    def synthesize(self, z, batch_size, which='stokes'):        
        z_split = np.array_split(z, batch_size, axis=0)        
        rec_all = []
        with torch.no_grad():
            for i in tqdm(range(len(z_split))):
                tmp = torch.tensor(z_split[i].astype('float32')).to(self.device)
                if (which == 'stokes'):                    
                    rec = self.decoder_stokes(tmp)
                if (which == 'models'):                    
                    rec = self.decoder_models(tmp)
                
                rec_all.append(rec.cpu().numpy())
                
        rec_all = np.concatenate(rec_all, axis=0).transpose(1, 0)
        if which == 'models':
            rec_all = rec_all.reshape((6, 51, -1))
            rec_all = np.pad(rec_all, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        if which == 'stokes':
            rec_all = rec_all.reshape((4, 112, -1))

        return rec_all
                    
    def project_obs(self):
        if platform.node() == 'gpu1':
            root = '/swap/aasensio/datasets/sims'        
        elif platform.node() == 'drogon.ll.iac.es':
            root = '/scratch1/aasensio/hinode_sims'
        elif platform.node() == 'vena.dyn.iac.es':
            root = '/net/drogon/scratch1/aasensio/hinode_sims'        
        else:            
            root = '/home/aasensio/datasets/hinode_simulations'
        
        print(f'Reading Stokes...')
        f = h5py.File(f'{root}/cheung_stokes_6301_Hinode_degraded.h5', 'r')
        stokes = f['stokes'][:]

        print(f'Reading ground truth models...')
        f = h5py.File(f'{root}/cheung_model_degraded.h5', 'r')
        self.models = f['model'][:, :, :, ::1]
        models = f['model'][:, :, :, ::1]
        Bx = models[:, :, 4, :]
        By = models[:, :, 5, :]
        Bperp1 = np.sign(Bx**2-By**2)*np.sqrt(np.abs(Bx**2-By**2))
        Bperp2 = np.sign(Bx*By)*np.sqrt(np.abs(Bx*By))
        models[:, :, 4, :] = Bperp1
        models[:, :, 5, :] = Bperp2
        models[:, :, 2, :] = np.log10(models[:, :, 2, :])
        models[:, :, 3, :] = models[:, :, 3, :] * 1e-5

        print(f'Adding noise to Stokes for observations...')
        stokes += np.random.normal(loc=0.0, scale=3e-4, size=stokes.shape)
        
        print(f'Symlogging Stokes for observations...')
        for j in range(1, 4):
            stokes[:, :, j, :] = symlog.symlog(stokes[:, :, j, :] * self.stats['symlog_stokes'][j])        

        self.stokes_obs = (stokes - self.stats['mn_stokes'][None, None, :, None]) / self.stats['std_stokes'][None, None, :, None]

        self.stokes_obs = np.transpose(self.stokes_obs, (2, 3, 0, 1))
        self.models_obs = np.transpose(models, (2, 3, 0, 1))

        ns, nl, nx, ny = self.stokes_obs.shape
        self.stokes_obs = self.stokes_obs.reshape((ns, nl, nx*ny))
                
        print("Encoding Stokes for observations...")
        self.z_stokes_obs, self.z_normal_stokes_obs = self.project(self.stokes_obs, 256, which='stokes')

        # k-NN search
        print("Zero-shot inversion...")
        self.indices, self.distances = self.index.query(self.z_normal_stokes_obs, k=1)
        
        # Inferred zero-shot models
        self.z_models_inv = self.z_models[self.indices.flatten(), :]
        
        print("Using decoders...")
        self.rec_models = self.synthesize(self.z_models_inv, 256, which='models')
        self.rec_stokes = self.synthesize(self.z_stokes_obs, 256, which='stokes')

        self.z_stokes_obs = self.z_stokes_obs.reshape((nx, ny, -1))
        self.stokes_obs = self.stokes_obs.reshape((ns, nl, nx, ny))        

        print("Denormalizing models...")
        self.rec_models = self.denormalize_models(self.rec_models)
        
        self.rec_models = self.rec_models.reshape((7, 51, nx, ny))
        self.rec_stokes = self.rec_stokes.reshape((4, 112, nx, ny))

    def project_hinode(self):
        
        print(f'Reading Stokes...')
        f = h5py.File(f'reconstructed.h5', 'r')
        stokes = f['reconstructed'][:].transpose((1, 0, 2, 3))
        
        print(f'Symlogging Stokes for observations...')
        for j in range(1, 4):
            stokes[j, :, :, :] = symlog.symlog(stokes[j, :, :, :] * self.stats['symlog_stokes'][j])        

        self.stokes_obs = (stokes - self.stats['mn_stokes'][:, None, None, None]) / self.stats['std_stokes'][:, None, None, None]
        
        ns, nl, nx, ny = self.stokes_obs.shape
        self.stokes_obs = self.stokes_obs.reshape((ns, nl, nx*ny))
                
        print("Encoding Stokes for observations...")
        self.z_stokes_obs, self.z_normal_stokes_obs = self.project(self.stokes_obs, 256, which='stokes')

        # k-NN search
        print("Zero-shot inversion...")
        self.indices, self.distances = self.index.query(self.z_normal_stokes_obs, k=1)
        
        # Inferred zero-shot models
        self.z_models_inv = self.z_models[self.indices.flatten(), :]

        print("Using decoders...")
        self.rec_models = self.synthesize(self.z_models_inv, 256, which='models')
        self.rec_stokes = self.synthesize(self.z_stokes_obs, 256, which='stokes')
        
        self.z_stokes_obs = self.z_stokes_obs.reshape((nx, ny, -1))
        self.stokes_obs = self.stokes_obs.reshape((ns, nl, nx, ny))        

        print("Denormalizing models...")
        self.rec_models = self.denormalize_models(self.rec_models)

        self.rec_models = self.rec_models.reshape((7, 51, nx, ny))
        self.rec_stokes = self.rec_stokes.reshape((4, 112, nx, ny))
        
    def plot(self):
        loc = [0, 26, 26, 24]
        labels = ['I', 'Q', 'U', 'V']
        fig, ax = pl.subplots(nrows=4, ncols=2, figsize=(10, 10))
        for j in range(4):            
            ax[j, 0].imshow(self.stokes_obs[j, loc[j], ...])
            ax[j, 1].imshow(self.rec_stokes[j, loc[j], ...])
            ax[j, 0].axis('off')
            ax[j, 1].axis('off')
            ax[j, 0].text(0.05, 0.9, f'Stokes {labels[j]}', color='white', fontsize=12, fontweight='bold', transform=ax[j, 0].transAxes)
        pl.tight_layout()

        pl.savefig('stokes.png')

        labels = ['T', 'vz', 'Bz']
        which = [1, 3, 6]
        loc = [0, 10, 20]

        fig, ax = pl.subplots(nrows=3, ncols=6, figsize=(23, 7))
        for i in range(3):
            for j in range(3):
                ax[j, 2*i].imshow(self.models_obs[which[i], loc[j], ...])
                ax[j, 2*i+1].imshow(self.rec_models[which[i], loc[j], ...])
                
                ax[j, 2*i].axis('off')
                ax[j, 2*i+1].axis('off')
            ax[0, 2*i].set_title(f'{labels[i]} - original')
            ax[0, 2*i+1].set_title(f'{labels[i]} - zero-shot')
        pl.tight_layout()
        pl.savefig('models.png')

    def plot_hinode(self):
        loc = [0, 26, 26, 24]
        labels = ['I', 'Q', 'U', 'V']
        fig, ax = pl.subplots(nrows=4, ncols=2, figsize=(10, 10))
        for j in range(4):            
            ax[j, 0].imshow(self.stokes_obs[j, loc[j], ...])
            ax[j, 1].imshow(self.rec_stokes[j, loc[j], ...])
            ax[j, 0].axis('off')
            ax[j, 1].axis('off')
            ax[j, 0].text(0.05, 0.9, f'Stokes {labels[j]}', color='white', fontsize=12, fontweight='bold', transform=ax[j, 0].transAxes)
        pl.tight_layout()

        pl.savefig('stokes_hinode.png')

        labels = ['T', 'vz', 'Bz']
        which = [1, 3, 6]
        loc = [0, 10, 20]

        fig, ax = pl.subplots(nrows=3, ncols=3, figsize=(12, 10))
        for i in range(3):
            for j in range(3):                
                im = ax[j, i].imshow(self.rec_models[which[i], loc[j], ...])
                pl.colorbar(im, ax=ax[j, i])
                
                ax[j, i].axis('off')
            ax[0, i].set_title(f'{labels[i]}')
        pl.tight_layout()
        pl.savefig('models_hinode.png')
                

if (__name__ == '__main__'):

    files = glob.glob('../train/weights/*_clip.pth')
    files.sort()
    checkpoint = files[-1]
    
    deepnet = Training(checkpoint, gpu=0, batch_size=1024*16)

    deepnet.read_db()

    # deepnet.project_obs()
    # deepnet.plot()
        
    deepnet.project_hinode()    
    deepnet.plot_hinode()
    