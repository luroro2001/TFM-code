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
import resnet
import dataset
import normalize
import symlog
import resnet
import glob
from einops import rearrange

def normalize_input(x, xmin, xmax):
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

def denormalize_output(x, xmin, xmax):
    return 0.5 * (x + 1.0) * (xmax - xmin) + xmin

    
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
    
class Testing(object):
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
            try:
                self.handle = Device.all()[self.gpu]
                print("Computing in {0} : {1}".format(self.device, self.handle.name()))
            except Exception:
                print("NVIDIA device not found, running on CPU instead.")
                self.device = torch.device("cpu")
                self.handle = None
            else:
                print("Running on CPU (no GPU or NVML detected).")
                self.device = torch.device("cpu")
                self.handle = None

        self.batch_size = batch_size
        self.decoders = self.config['training']['use_decoders']
        
        # Model
        # self.encoding_pars = mlp.MLP(n_input=9,
        #                             n_output=64,
        #                             dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
        #                             n_hidden=self.hyperparameters['mlp']['num_layers_mlp'],
        #                             activation=nn.ReLU()).to(self.device)
        
        # self.encoding_stokes = mlp.MLP(n_input=400,
        #                             n_output=64,
        #                             dim_hidden=self.hyperparameters['mlp']['n_hidden_mlp'],                                 
        #                             n_hidden=self.hyperparameters['mlp']['num_layers_mlp'],
        #                             activation=nn.ReLU()).to(self.device)
        
        self.encoder_models = resnet.ResidualNet(in_features=6*80, 
                      out_features=self.config['mlp']['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=self.config['mlp']['dropout_probability'],
                      use_batch_norm=True).to(self.device)
        
        self.encoder_stokes = resnet.ResidualNet(in_features=4*112, 
                      out_features=self.config['mlp']['latent_dim'],
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=self.config['mlp']['dropout_probability'],
                      use_batch_norm=True).to(self.device)
        
        if self.decoders:
            self.decoder_models = resnet.ResidualNet(in_features=self.config['mlp']['latent_dim'],
                      out_features=6*80,
                      hidden_features=self.config['mlp']['n_hidden_mlp'],
                      num_blocks=self.config['mlp']['num_layers_mlp'],
                      activation=F.gelu,
                      dropout_probability=self.config['mlp']['dropout_probability'],
                      use_batch_norm=True).to(self.device)

            self.decoder_stokes = resnet.ResidualNet(in_features=self.config['mlp']['latent_dim'],
                        out_features=4*112,
                        hidden_features=self.config['mlp']['n_hidden_mlp'],
                        num_blocks=self.config['mlp']['num_layers_mlp'],
                        activation=F.gelu,
                        dropout_probability=self.config['mlp']['dropout_probability'],
                        use_batch_norm=True).to(self.device)

        print("Setting weights of the model...")        
        self.encoder_models.load_state_dict(chk['encoder_models_dict'])
        self.encoder_stokes.load_state_dict(chk['encoder_stokes_dict'])

        self.encoder_stokes.eval()
        self.encoder_models.eval()

        if self.decoders:
            self.decoder_models.load_state_dict(chk['decoder_models_dict'])
            self.decoder_stokes.load_state_dict(chk['decoder_stokes_dict'])
        
    def denormalize(self, models):
        lower = [2000., 0.0, -10.0, 0.0, -1000.0, -1000.0]
        upper = [25000., 3.0, 10.0, 1000.0, 1000.0, 1000.0]
        
        for i in range(models.shape[1]):
            models[:, i, :] = denormalize_output(models[:, i, :], lower[i], upper[i])

        return models
                
    def test(self):

        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}

        self.test_dataset = dataset.Dataset('stokes_testing.h5', 
                                                   'models_testing.h5', 
                                                   'good_profiles_testing.npy',
                                                   noise=self.config['training']['noise'])
        
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False, 
                    **kwargs)
        
        t = tqdm(self.test_loader)

        z_stokes = []
        z_models = []
        models_all = []
        stokes_all = []
        decoded_stokes_all = []
        decoded_models_all = []

        with torch.no_grad():

            for batch_idx, (stokes, models) in enumerate(t):
                models = models.to(self.device)
                stokes = stokes.to(self.device)

                stokes_flat = rearrange(stokes, 'b c h -> b (c h)')
                models_flat = rearrange(models, 'b c h -> b (c h)')

                z_s = self.encoder_stokes(stokes_flat)
                z_m = self.encoder_models(models_flat)

                z_s = F.normalize(z_s, dim=-1)
                z_m = F.normalize(z_m, dim=-1)

                z_stokes.append(z_s.cpu().numpy())
                z_models.append(z_m.cpu().numpy())

                models_all.append(models.cpu().numpy())
                stokes_all.append(stokes.cpu().numpy())

                if self.decoders:
                    decoded_stokes = self.decoder_stokes(z_s)
                    decoded_models = self.decoder_models(z_m)

                    decoded_stokes = rearrange(decoded_stokes, 'b (c h) -> b c h', c=4)
                    decoded_models = rearrange(decoded_models, 'b (c h) -> b c h', c=6)
                    
                    decoded_stokes_all.append(decoded_stokes.cpu().numpy())
                    decoded_models_all.append(decoded_models.cpu().numpy())

                
        z_stokes = np.concatenate(z_stokes, axis=0)
        z_models = np.concatenate(z_models, axis=0)
        models_all = np.concatenate(models_all, axis=0)
        stokes_all = np.concatenate(stokes_all, axis=0)
        decoded_models_all = np.concatenate(decoded_models_all, axis=0) if self.decoders else None
        decoded_stokes_all = np.concatenate(decoded_stokes_all, axis=0) if self.decoders else None

        models_all = self.denormalize(models_all)
        decoded_models_all = self.denormalize(decoded_models_all) if self.decoders else None

        return z_stokes, z_models, models_all, stokes_all, decoded_models_all, decoded_stokes_all


if (__name__ == '__main__'):

    files = glob.glob('../train/weights/*.pth')
    files.sort()
    checkpoint = files[-1]
    
    deepnet = Testing(checkpoint, gpu=0, batch_size=1024)
    z_stokes, z_models, models, stokes, decoded_models, decoded_stokes = deepnet.test()