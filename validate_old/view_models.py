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
    
    
class Training(object):
    def __init__(self, checkpoint, gpu, batch_size):

        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)

        chk = torch.load(checkpoint+'.best', map_location=lambda storage, loc: storage)

        self.config = chk['config']

        self.training_dataset = datasets.Dataset1D(self.config['training_set'], pctx=[0,100], pcty=[0,100], stats=None, step=5)

    def denormalize(self, models):
        models = normalize.denormalize(models, self.stats['min_models'][:,None,None], self.stats['max_models'][:,None,None])
        models[2, ...] = symlog.inv_symlog(models[2, ...])
        models[4:, ...] = symlog.inv_symlog(models[4:, ...])

        models[2, ...] = np.log10(models[2, ...])
        models[3, ...] = models[3, ...] * 1e-5
        return models
                    
    def plot(self, which=0):
        app = pg.mkQApp("Crosshair Example")

        win1 = pg.GraphicsLayoutWidget(show=True)
        win1.setWindowTitle('pyqtgraph example: crosshair')
        label = pg.LabelItem(justify='right')
        win1.addItem(label)
        p1 = win1.addPlot(row=1, col=0, rowspan=2)
        img = pg.ImageItem()
        p1.addItem(img)
        img.setImage(self.training_dataset.stokes[which][0, 0, :, :])

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
            i = int(np.clip(i, 0, self.training_dataset.stokes[which][0, 0, :, :].shape[0] - 1))
            j = int(np.clip(j, 0, self.training_dataset.stokes[which][0, 0, :, :].shape[0] - 1))        

            ppos = img.mapToParent(pos)
            x, y = ppos.x(), ppos.y()

            pStokesI.plot(self.training_dataset.stokes[which][0, :, i, j], pen=[255, 255, 255], clear=True)
            pStokesQ.plot(self.training_dataset.stokes[which][1, :, i, j], pen=[255, 255, 255], clear=True)
            pStokesU.plot(self.training_dataset.stokes[which][2, :, i, j], pen=[255, 255, 255], clear=True)
            pStokesV.plot(self.training_dataset.stokes[which][3, :, i, j], pen=[255, 255, 255], clear=True)

            pT.plot(self.training_dataset.models[which][1, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            plogP.plot(self.training_dataset.models[which][2, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pvz.plot(self.training_dataset.models[which][3, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pBp1.plot(self.training_dataset.models[which][4, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pBp2.plot(self.training_dataset.models[which][5, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')
            pBz.plot(self.training_dataset.models[which][6, :, i, j], pen=[255, 255, 255], clear=True, symbol='x')


        img.hoverEvent = imageHoverEvent
        pg.exec()

if (__name__ == '__main__'):

    files = glob.glob('../train/weights/*.pth')
    files.sort()
    checkpoint = files[-1]
    
    deepnet = Training(checkpoint, gpu=0, batch_size=1024*16)
    
    deepnet.plot(which=3)

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