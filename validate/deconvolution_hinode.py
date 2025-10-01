import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.utils.data
import matplotlib.pyplot as pl
import h5py
from astropy.io import fits
from collections import OrderedDict
from tqdm import tqdm
from noise_svd import noise_estimation
from kornia.filters import median_blur, spatial_gradient
from nvitop import Device
import platform

class Classic(nn.Module):
    def __init__(self, config):
        """

        Parameters
        ----------
        npix_apodization : int
            Total number of pixel for apodization (divisible by 2)
        device : str
            Device where to carry out the computations
        batch_size : int
            Batch size
        """
        super().__init__()

        self.config = config          
        
        self.cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{self.config['gpus'][0]}" if self.cuda else "cpu")        

        self.handle = Device.all()[self.config['gpus'][0]]
            
        print(f"Computing in {self.device} : {self.handle.name()} - mem: {self.handle.memory_used_human()}/{self.handle.memory_total_human()}")
        
        # Generate Hamming window function for WFS correlation
        self.npix_apod = self.config['npix_apodization']
        win = np.hanning(self.npix_apod)
        winOut = np.ones(self.config['n_pixel'])
        winOut[0:self.npix_apod//2] = win[0:self.npix_apod//2]
        winOut[-self.npix_apod//2:] = win[-self.npix_apod//2:]
        window = np.outer(winOut, winOut)
                
        # Define Zernike modes
        self.window = torch.tensor(window.astype('float32')).to(self.device)        
        
        self.cutoff = self.config['diameter'] / (self.config['wavelength'] * 1e-8) / 206265.0
        freq = np.fft.fftfreq(self.config['n_pixel'], d=self.config['pix_size']) / self.cutoff
        
        xx, yy = np.meshgrid(freq, freq)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        mask = rho <= 0.85
        mask_shift = np.fft.fftshift(mask)
        self.mask = torch.tensor(mask.astype('float32')).to(self.device)
        self.mask_shift = torch.tensor(mask_shift.astype('float32')).to(self.device)

        if self.config['precision'] == 'float16':
            print("Working in float16...")
            self.use_amp = True
        else:
            print("Working in float32...")
            self.use_amp = False

        # Define the scaler for the automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)                         

    def lofdahl_scharmer_filter(self, image_ft, psf_ft):
                
        num = torch.abs(psf_ft)**2
        denom = torch.abs(image_ft.detach() * torch.conj(psf_ft))**2
        H = 1.0 - self.mask * self.config['n_pixel']**2 * self.sigma**2 * (num / denom)
                
        H[H > 1.0] = 1.0
        H[H < 0.2] = 0.0

        H = self.mask * median_blur(H, (3, 3)).squeeze()
        H = torch.nan_to_num(H)

        H[H < 0.2] = 0.0
        
        return H

    def forward(self, image):
        # Apodize frames and compute FFT
        mean_val = torch.mean(image, dim=(2, 3), keepdim=True)
        image_apod = image - mean_val
        image_apod *= self.window
        image_apod += mean_val
        image_ft = torch.fft.fft2(image)

        # Threshold
        if (self.regularize_fourier == 'scharmer'):
            H = self.lofdahl_scharmer_filter(image_ft, self.psf_ft)
        if (self.regularize_fourier == 'mask'):            
            H = self.mask[None, None, :, :]

        # Convolve estimated image with PSF            
        convolved = torch.fft.ifft2(H * image_ft * self.psf_ft).real

        # Regularization
        grad = spatial_gradient(image, mode='sobel', order=1)

        return convolved, grad

    
    def deconvolve_torch(self, obs, psf, sigma, regularize_fourier='mask', lambda_grad=0.1, lambda_obj=0.0, lambda_spectra=0.0):
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        self.psf = psf.to(self.device)
        self.psf_ft = torch.fft.fft2(self.psf)

        obs = obs.to(self.device)
        sigma = sigma.to(self.device)

        if nl == 1:
            lambda_spectral = 0.0

        lambda_grad = torch.tensor(lambda_grad).to(self.device)
        lambda_obj = torch.tensor(lambda_obj).to(self.device)
        lambda_spectra = torch.tensor(lambda_spectra).to(self.device)        

        self.sigma = sigma
        self.weight = 1.0 / sigma
        self.regularize_fourier = regularize_fourier
                            
        image = obs.clone().detach().requires_grad_(True).to(self.device)

        optimizer = torch.optim.Adam([image], lr=0.1)

        losses = []

        t = tqdm(range(self.config['gradient_steps']))
        
        for loop in t:

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):

                if self.config['checkpointing']:
                    convolved, grad = torch.utils.checkpoint.checkpoint(self.forward, image, use_reentrant=False)
                else:
                    convolved, grad = self.forward(image)

                regul_grad = lambda_grad * torch.mean(grad**2)
                regul_obj = lambda_obj * torch.mean(image**2)

                # 1D filter
                if (lambda_spectra > 0.0):                
                    grad_spectra = image[1:, ...] - image[:-1, ...]
                    regul_spectra = lambda_spectra * torch.mean(grad_spectra**2)
                else:
                    regul_spectra = torch.tensor(0.0).to(self.device)

                # Compute the loss
                loss_mse = torch.mean( ( obs - convolved )**2)

                loss = loss_mse + regul_grad + regul_obj + regul_spectra
                                                                
            self.scaler.scale(loss).backward()

            # Update the parameters
            self.scaler.step(optimizer)
            self.scaler.update()
                        
            tmp = OrderedDict()
            tmp['gpu'] = f'{self.handle.gpu_utilization()}'                
            tmp['mem'] = f' {self.handle.memory_used_human()}/{self.handle.memory_total_human()}'
            tmp['loss_mse'] = f'{loss_mse.item():.8f}'
            tmp['reg_grad'] = f'{regul_grad.item():.8f}'
            tmp['reg_obj'] = f'{regul_obj.item():.8f}'
            tmp['reg_spec'] = f'{regul_spectra.item():.8f}'
            tmp['loss'] = f'{loss.item():.8f}'
            t.set_postfix(ordered_dict=tmp)

            losses.append(loss.item())

        losses = np.array(losses)        
        
        mean_val = torch.mean(image, dim=(2, 3), keepdim=True)
        image_apod = image - mean_val
        image_apod *= self.window
        image_apod += mean_val

        image = image_apod.detach().cpu().numpy()

        return image, losses
    
if (__name__ == '__main__'):

    f = fits.open('hinode_psf_size_256_def_-0.32_hinode.fits')
    psf = f[0].data[0:-1, 0:-1]

    if platform.node() == 'gpu1':
        root = '/swap/aasensio/datasets/sims'        
    elif platform.node() == 'drogon.ll.iac.es':
        root = '/net/drogon/scratch1/aasensio/hinode_spots/neural_hinode'
        f = h5py.File(f'{root}/10953_0.h5','r')
        stokes = f['stokes'][:, 100:500,100:500, :].transpose((3, 0, 1, 2))
    elif platform.node() == 'vena.dyn.iac.es':
        root = '/net/drogon/scratch1/aasensio/hinode_spots/neural_hinode'
        f = h5py.File(f'{root}/10953_0.h5','r')
        stokes = f['stokes'][:, 100:500,100:500, :].transpose((3, 0, 1, 2))
    else:
        root = '/home/aasensio/datasets/hinode_sunspots'
        f = h5py.File(f'{root}/10921_0.h5', 'r')
        stokes = f['stokes'][:, 600:1000,100:500, :].transpose((3, 0, 1, 2))    
    
    nl, ns, nx, ny = stokes.shape

    pad_width = (nx - psf.shape[0]) // 2

    psf = np.pad(psf, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=0)
    psf = psf / np.sum(psf)
    psf = np.fft.fftshift(psf)
    
    config = {
        'gpus': [0],
        'npix_apodization': 24,        
        'gradient_steps' : 50,
        'wavelength': 6302.5,
        'diameter': 50.0,
        'pix_size': 0.16,        
        'precision': 'float32',
        'checkpointing': True,
        'n_pixel': nx,
    }

    lambda_grad_all = [0.0005, 0.0005, 0.0005, 0.0005]
    lambda_obj_all = [0.0, 0.01, 0.01, 0.01]
    lambda_spectra_all = [0.2, 0.2, 0.2, 0.2]

    classic = Classic(config)

    reconstructed_all = []

    for j in range(4):
        im = stokes[0:, j:j+1, :, :] / np.mean(stokes[0, 0, 0:20, 0:20])

        print("Estimating noise...")
        sigma = noise_estimation(im[0, 0, :, :])
        print(sigma)           
                
        frames = torch.tensor(im.astype('float32'))
        sigma = torch.tensor(sigma.astype('float32'))
        
        if (j == 0):
            psf = torch.tensor(psf.astype('float32'))
            
        # For Stokes QUV
        reconstructed, loss = classic.deconvolve_torch(frames, 
                                                       psf, 
                                                       sigma, 
                                                       regularize_fourier='mask', 
                                                       lambda_grad=lambda_grad_all[j],
                                                       lambda_obj=lambda_obj_all[j],
                                                       lambda_spectra=lambda_spectra_all[j])
        
        reconstructed_all.append(reconstructed)

    reconstructed = np.concatenate(reconstructed_all, axis=1)

    f = h5py.File('reconstructed.h5', 'w')
    f.create_dataset('reconstructed', data=reconstructed)
    f.close()

    # n = config['npix_apodization'] // 2
    # im = im[:, :, n:-n, n:-n]
    # reconstructed = reconstructed[:, :, n:-n, n:-n]

    # fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
    # ax[0, 0].imshow(im[0, 0, :, :], cmap='gray')
    # ax[0, 1].imshow(reconstructed[0, 0, :, :], cmap='gray')

    # ax[1, 0].imshow(im[30, 0, :, :], cmap='gray')
    # ax[1, 1].imshow(reconstructed[30, 0, :, :], cmap='gray')

    # fig, ax = pl.subplots()
    # ax.plot(im[:, 0, 20, 20], color='C0')
    # ax.plot(reconstructed[:, 0, 20, 20], color='C1')

    # ax.plot(im[:, 0, 200, 200], color='C0')
    # ax.plot(reconstructed[:, 0, 200, 200], color='C1')