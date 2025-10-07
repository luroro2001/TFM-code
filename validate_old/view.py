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
import mlp
import datasets
import normalize
import symlog
import resnet
import glob
from einops import rearrange
import pyqtgraph as pg
    
class CLIPLoss(nn.Module):
    """ Simple contrastive loss for CLIP
    """
    def get_logits(self, z1_features, z2_features, logit_scale):
        logits_per_z1 = logit_scale * z1_features @ z2_features.T
        logits_per_z2 = logit_scale * z2_features @ z1_features.T
        return logits_per_z1, logits_per_z2

    def forward(self, z1_features, z2_features, logit_scale):
        logits_per_z1, logits_per_z2 = self.get_logits(z1_features, z2_features, logit_scale)        
        labels = torch.arange(logits_per_z1.shape[0], device=z1_features.device, dtype=torch.long)
        total_loss = 0.5 * (F.cross_entropy(logits_per_z1, labels) + F.cross_entropy(logits_per_z2, labels))
        return total_loss
    
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
        self.encoding_models = resnet.ResidualNet(in_features=6*11, 
                      out_features=self.config['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=0.1,
                      use_batch_norm=True).to(self.device)
        
        self.encoding_stokes = resnet.ResidualNet(in_features=4*112, 
                      out_features=self.config['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=0.1,
                      use_batch_norm=True).to(self.device)
        
        self.decoding_models = resnet.ResidualNet(in_features=self.config['latent_dim'],
                      out_features=6*11,
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=0.1,
                      use_batch_norm=True).to(self.device)
        
        self.decoding_stokes = resnet.ResidualNet(in_features=self.config['latent_dim'],
                      out_features=4*112,
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=0.1,
                      use_batch_norm=True).to(self.device)

        print("Setting weights of the model...")        
        self.encoding_models.load_state_dict(chk['encoding_models_dict'])
        self.encoding_stokes.load_state_dict(chk['encoding_stokes_dict'])
        self.decoding_models.load_state_dict(chk['decoding_models_dict'])
        self.decoding_stokes.load_state_dict(chk['decoding_stokes_dict'])

        self.stats = chk['stats']
        
        self.encoding_stokes.eval()
        self.encoding_models.eval()
        self.decoding_stokes.eval()
        self.decoding_models.eval()

        self.loss_fn = CLIPLoss()

    def denormalize_models(self, models):        
        models = models * self.stats['std_models'][:, :, None] + self.stats['mn_models'][:, :, None]
        models[2, ...] = symlog.inv_symlog(models[2, ...])
        models[4:, ...] = symlog.inv_symlog(models[4:, ...])

        models[2, ...] = np.log10(models[2, ...])
        models[3, ...] = models[3, ...] * 1e-5
        return models

    def project(self, array, batch_size, which='stokes'):
        ns, nl, n = array.shape
        array_flat = array.reshape((ns*nl, n)).T
        array_flat = np.array_split(array_flat, batch_size, axis=0)
        z_all = []
        rec_all = []
        with torch.no_grad():
            for i in tqdm(range(len(array_flat))):
                tmp = torch.tensor(array_flat[i].astype('float32')).to(self.device)
                if (which == 'stokes'):
                    z = self.encoding_stokes(tmp)                    
                if (which == 'models'):
                    z = self.encoding_models(tmp)

                z = F.normalize(z, dim=-1)

                if (which == 'stokes'):
                    rec = self.decoding_stokes(z).reshape((-1, 4, 112))
                if (which == 'models'):
                    rec = self.decoding_models(z).reshape((-1, 6, 11))

                z_all.append(z.cpu().numpy())
                rec_all.append(rec.cpu().numpy())
        
        z_all = np.concatenate(z_all, axis=0)
        rec_all = np.concatenate(rec_all, axis=0).transpose((1, 2, 0))

        # Artificially pad the first element of the array with zeros to mimick log(tau), that is not predicted
        rec_all = np.pad(rec_all, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

        return z_all, rec_all
                
    def process(self):
        
        self.training_dataset = datasets.Dataset1D(self.config['training_set'], 
                    pctx=[0,80], 
                    pcty=[0,100], 
                    stats=None, 
                    step=5)
        
        print("Stacking Stokes...")
        stokes_all = []
        for i in range(4):                        
            stokes_all.append(self.training_dataset.stokes[i].reshape((4, 112, -1)))
        self.stokes_all = np.concatenate(stokes_all, axis=-1)
        
        print("Encoding Stokes...")
        self.z_stokes, self.rec_stokes = self.project(self.stokes_all, 256, which='stokes')        

        print("Stacking models...")        
        models_all = []
        for i in range(4):            
            models = self.training_dataset.models[i][:, ...]
            models_all.append(models.reshape((7, 11, -1)))
        self.models_all = np.concatenate(models_all, axis=-1)        

        print("Encoding models...")
        self.z_models, self.rec_models = self.project(self.models_all[1:, ...], 256, which='models')

        self.z_stokes_2d = []
        self.z_models_2d = []
        left = 0
        for i in range(4):
            nx, ny = self.training_dataset.stokes[i].shape[2:]
            nz = self.z_stokes.shape[1]
            right = left + nx*ny
            self.z_stokes_2d.append(self.z_stokes[left:right, :].reshape((nx, ny, nz)))
            self.z_models_2d.append(self.z_models[left:right, :].reshape((nx, ny, nz)))
            left += nx*ny

        print("Denormalizing models...")                
        self.models_all = self.denormalize_models(self.models_all)
        self.rec_models = self.denormalize_models(self.rec_models)
        
        nq, nh, nx, ny = self.training_dataset.models[0].shape
        self.models = self.denormalize_models(self.training_dataset.models[0].reshape((nq, nh, nx*ny))).reshape((nq, nh, nx, ny))
        
        return
    
    def plot(self):
        app = pg.mkQApp("Crosshair Example")

        win1 = pg.GraphicsLayoutWidget(show=True)
        win1.setWindowTitle('pyqtgraph example: crosshair')
        label = pg.LabelItem(justify='right')
        win1.addItem(label)
        p1 = win1.addPlot(row=1, col=0, rowspan=2)
        img = pg.ImageItem()
        p1.addItem(img)
        img.setImage(self.training_dataset.stokes[0][0, 0, :, :])

        win = pg.GraphicsLayoutWidget(show=True)
        win.setWindowTitle('pyqtgraph example: crosshair')
        label = pg.LabelItem(justify='right')
        win.addItem(label)
        p1 = win.addPlot(row=1, col=0)

        # customize the averaged curve that can be activated from the context menu:
        p1.avgPen = pg.mkPen('#FFFFFF')
        p1.avgShadowPen = pg.mkPen('#8080DD', width=10)

        # p2 = win.addPlot(row=2, col=0)
        

        win.nextRow()
        pStokesI = win.addPlot(row=2, col=0)
        pStokesQ = win.addPlot(row=2, col=1)
        pStokesU = win.addPlot(row=2, col=2)
        pStokesV = win.addPlot(row=2, col=3)

        pT = win.addPlot(row=3, col=0)
        plogP = win.addPlot(row=3, col=1)
        pvz = win.addPlot(row=3, col=2)
        pBp1 = win.addPlot(row=3, col=3)
        pBp2 = win.addPlot(row=3, col=4)
        pBz = win.addPlot(row=3, col=5)

        win.resize(1200, 1200)
        win.show()

        def imageHoverEvent(event):
            pos = event.pos()
            i, j = pos.x(), pos.y()
            i = int(np.clip(i, 0, self.training_dataset.stokes[0][0, 0, :, :].shape[0] - 1))
            j = int(np.clip(j, 0, self.training_dataset.stokes[0][0, 0, :, :].shape[0] - 1))        

            ppos = img.mapToParent(pos)
            x, y = ppos.x(), ppos.y()

            pStokesI.plot(self.training_dataset.stokes[0][0, :, i, j], pen=[255, 255, 255], clear=True)
            pStokesQ.plot(self.training_dataset.stokes[0][1, :, i, j], pen=[255, 255, 255], clear=True)
            pStokesU.plot(self.training_dataset.stokes[0][2, :, i, j], pen=[255, 255, 255], clear=True)
            pStokesV.plot(self.training_dataset.stokes[0][3, :, i, j], pen=[255, 255, 255], clear=True)

            z1 = self.z_stokes_2d[0][i, j, :]
            sim = np.sum(z1[None, :] * self.z_models, axis=-1)
            ind = np.argsort(sim)[::-1][0:5]            

            pT.plot(self.models[1, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            plogP.plot(self.models[2, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pvz.plot(self.models[3, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pBp1.plot(self.models[4, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pBp2.plot(self.models[5, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pBz.plot(self.models[6, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')

            for i in range(5):
                pStokesI.plot(self.stokes_all[0, :, ind[i]], pen=[255, 0, 0])
                pStokesQ.plot(self.stokes_all[1, :, ind[i]], pen=[255, 0, 0])
                pStokesU.plot(self.stokes_all[2, :, ind[i]], pen=[255, 0, 0])
                pStokesV.plot(self.stokes_all[3, :, ind[i]], pen=[255, 0, 0])

                pT.plot(self.models_all[1, :, ind[i]], pen=[255, 0, 0])
                plogP.plot(self.models_all[2, :, ind[i]], pen=[255, 0, 0])
                pvz.plot(self.models_all[3, :, ind[i]], pen=[255, 0, 0])
                pBp1.plot(self.models_all[4, :, ind[i]], pen=[255, 0, 0])
                pBp2.plot(self.models_all[5, :, ind[i]], pen=[255, 0, 0])
                pBz.plot(self.models_all[6, :, ind[i]], pen=[255, 0, 0])

                pT.plot(self.rec_models[1, :, ind[i]], pen=[255, 255, 0])
                plogP.plot(self.rec_models[2, :, ind[i]], pen=[255, 255, 0])
                pvz.plot(self.rec_models[3, :, ind[i]], pen=[255, 255, 0])
                pBp1.plot(self.rec_models[4, :, ind[i]], pen=[255, 255, 0])
                pBp2.plot(self.rec_models[5, :, ind[i]], pen=[255, 255, 0])
                pBz.plot(self.rec_models[6, :, ind[i]], pen=[255, 255, 0])


        img.hoverEvent = imageHoverEvent
        pg.exec()

if (__name__ == '__main__'):

    files = glob.glob('../train/weights/*.pth')
    files.sort()
    checkpoint = files[-1]
    
    deepnet = Training(checkpoint, gpu=0, batch_size=1024*16)
    
    deepnet.process()

    deepnet.plot()

    # z_stokes, z_models, loss, loss_val, models, stokes = deepnet.test()

    # which = 252000

    # z1 = z_stokes[which, :]
    
    # sim = np.sum(z1[None, :] * z_models, axis=-1)

    # ind = np.argsort(sim)[::-1][0:5]

    # labels = ['T', 'logP', 'vz', 'Bp1', 'Bp2', 'Bz']

    # fig, ax = pl.subplots(nrows=3, ncols=2, figsize=(10, 15))
    # for i in range(6):
    #     ax.flat[i].plot(models[i, :, which], linewidth=2, color='black')
    #     ax.flat[i].set_ylabel(labels[i])
    #     for j in range(5):
    #         ax.flat[i].plot(models[i, :, ind[j]])

    # fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
    # for i in range(4):
    #     ax.flat[i].plot(stokes[i, :, which], linewidth=2, color='black')
    #     for j in range(5):
    #         ax.flat[i].plot(stokes[i, :, ind[j]])