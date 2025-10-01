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
import mlp
import glob
import pyqtgraph as pg
import platform
import h5py
    
    
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
                      hidden_features=self.config['encoder']['n_hidden'],
                      num_blocks=self.config['encoder']['num_layers'],
                      activation=F.relu,
                      dropout_probability=0.0,
                      use_batch_norm=True).to(self.device)
        
        self.encoder_stokes = resnet.ResidualNet(in_features=4*112, 
                      out_features=self.config['latent_dim'],
                      hidden_features=self.config['encoder']['n_hidden'],
                      num_blocks=self.config['encoder']['num_layers'],
                      activation=F.relu,
                      dropout_probability=0.0,
                      use_batch_norm=True).to(self.device)
        
        self.projector_models = mlp.MLP(n_input=self.config['latent_dim'], 
                      n_output=self.config['embedding_dim'],
                      dim_hidden=self.config['projector']['n_hidden'],
                      n_hidden=self.config['projector']['num_layers'],
                      activation=nn.ReLU(),
                      last_bias=False,
                      bn=True).to(self.device)
        
        self.projector_stokes = mlp.MLP(n_input=self.config['latent_dim'], 
                      n_output=self.config['embedding_dim'],
                      dim_hidden=self.config['projector']['n_hidden'],
                      n_hidden=self.config['projector']['num_layers'],
                      activation=nn.ReLU(),
                      last_bias=False,
                      bn=True).to(self.device)
        

        print("Setting weights of the model...")        
        self.encoder_models.load_state_dict(chk['encoder_models_dict'])
        self.encoder_stokes.load_state_dict(chk['encoder_stokes_dict'])
        self.projector_models.load_state_dict(chk['projector_models_dict'])
        self.projector_stokes.load_state_dict(chk['projector_stokes_dict'])

        self.stats = chk['stats']
        
        self.encoder_stokes.eval()
        self.encoder_models.eval()
        self.projector_stokes.eval()
        self.projector_models.eval()

    def denormalize_models(self, models):        
        models = models * self.stats['std_models'][:, None, None] + self.stats['mn_models'][:, None, None]
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
        with torch.no_grad():
            for i in tqdm(range(len(array_flat))):
                tmp = torch.tensor(array_flat[i].astype('float32')).to(self.device)
                if (which == 'stokes'):
                    z = self.projector_stokes(self.encoder_stokes(tmp))
                    # z = self.encoder_stokes(tmp)
                if (which == 'models'):
                    z = self.projector_models(self.encoder_models(tmp))
                    # z = self.encoder_models(tmp)

                z_all.append(z.cpu().numpy())
        
        z_all = np.concatenate(z_all, axis=0)

        return z_all
                
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
        
        print("Encoding Stokes...")
        self.z_stokes = self.project(self.stokes_all, 256, which='stokes')        

        print("Stacking models...")        
        models_all = []
        for i in range(4):            
            models = self.training_dataset.models[i][:, ...]
            models_all.append(models.reshape((7, 51, -1)))
        self.models_all = np.concatenate(models_all, axis=-1)        

        print("Encoding models...")
        self.z_models = self.project(self.models_all[1:, ...], 256, which='models')
        
        print("Denormalizing models...")                
        self.models_all = self.denormalize_models(self.models_all)
                
        return
    
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
        self.z_stokes_obs = self.project(self.stokes_obs, 256, which='stokes')

        self.z_stokes_obs = self.z_stokes_obs.reshape((nx, ny, -1))
        self.stokes_obs = self.stokes_obs.reshape((ns, nl, nx, ny))
        
    def plot(self):
        app = pg.mkQApp("Crosshair Example")

        win1 = pg.GraphicsLayoutWidget(show=True)
        win1.setWindowTitle('pyqtgraph example: crosshair')
        label = pg.LabelItem(justify='right')
        win1.addItem(label)
        p1 = win1.addPlot(row=1, col=0, rowspan=2)
        img = pg.ImageItem()
        p1.addItem(img)
        img.setImage(self.stokes_obs[0, 0, :, :])

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
            i = int(np.clip(i, 0, self.stokes_obs[0, 0, :, :].shape[0] - 1))
            j = int(np.clip(j, 0, self.stokes_obs[0, 0, :, :].shape[0] - 1))        

            ppos = img.mapToParent(pos)
            x, y = ppos.x(), ppos.y()

            pStokesI.plot(self.stokes_obs[0, :, i, j], pen=[255, 255, 255], clear=True)
            pStokesQ.plot(self.stokes_obs[1, :, i, j], pen=[255, 255, 255], clear=True)
            pStokesU.plot(self.stokes_obs[2, :, i, j], pen=[255, 255, 255], clear=True)
            pStokesV.plot(self.stokes_obs[3, :, i, j], pen=[255, 255, 255], clear=True)

            z1 = self.z_stokes_obs[i, j, :]
            sim = np.sum((z1[None, :] - self.z_models)**2, axis=-1)
            ind = np.argsort(sim)[0:5]            

            pT.plot(self.models_obs[1, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            plogP.plot(self.models_obs[2, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pvz.plot(self.models_obs[3, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pBp1.plot(self.models_obs[4, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pBp2.plot(self.models_obs[5, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pBz.plot(self.models_obs[6, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')

            avg_model = np.mean(self.models_all[:, :, ind], axis=-1)
            avg_stokes = np.mean(self.stokes_all[:, :, ind], axis=-1)

            for i in range(5):

                factor = 1.0 - i / 5.0
                pStokesI.plot(self.stokes_all[0, :, ind[i]], pen=[255 * factor, 0, 0])
                pStokesQ.plot(self.stokes_all[1, :, ind[i]], pen=[255 * factor, 0, 0])
                pStokesU.plot(self.stokes_all[2, :, ind[i]], pen=[255 * factor, 0, 0])
                pStokesV.plot(self.stokes_all[3, :, ind[i]], pen=[255 * factor, 0, 0])

                pT.plot(self.models_all[1, :, ind[i]], pen=[255 * factor, 0, 0])
                plogP.plot(self.models_all[2, :, ind[i]], pen=[255 * factor, 0, 0])
                pvz.plot(self.models_all[3, :, ind[i]], pen=[255 * factor, 0, 0])
                pBp1.plot(self.models_all[4, :, ind[i]], pen=[255 * factor, 0, 0])
                pBp2.plot(self.models_all[5, :, ind[i]], pen=[255 * factor, 0, 0])
                pBz.plot(self.models_all[6, :, ind[i]], pen=[255 * factor, 0, 0])

            pStokesI.plot(avg_stokes[0, :], pen=[0, 0, 255])
            pStokesQ.plot(avg_stokes[1, :], pen=[0, 0, 255])
            pStokesU.plot(avg_stokes[2, :], pen=[0, 0, 255])
            pStokesV.plot(avg_stokes[3, :], pen=[0, 0, 255])

            pT.plot(avg_model[1, :], pen=[0, 0, 255])
            plogP.plot(avg_model[2, :], pen=[0, 0, 255])
            pvz.plot(avg_model[3, :], pen=[0, 0, 255])
            pBp1.plot(avg_model[4, :], pen=[0, 0, 255])
            pBp2.plot(avg_model[5, :], pen=[0, 0, 255])
            pBz.plot(avg_model[6, :], pen=[0, 0, 255])
            

        img.hoverEvent = imageHoverEvent
        pg.exec()

if (__name__ == '__main__'):

    files = glob.glob('../train/weights/*_vicreg.pth')
    files.sort()
    checkpoint = files[-1]
    
    deepnet = Training(checkpoint, gpu=0, batch_size=1024*16)
    
    deepnet.project_obs()
    deepnet.read_db()
    

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