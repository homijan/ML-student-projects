import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import numpy as np

class AlphaModel(pl.LightningModule): 

### Model ###
    def __init__(self, Nfeatures, N1, N2, Ntargets, scaling, Nfields):
        super(AlphaModel, self).__init__() # TODO: if not "cannot assign module before Module.__init__() call"
        # Initialize layers
        self.fc1 = torch.nn.Linear(Nfeatures, N1)
        self.fc2 = torch.nn.Linear(N1, N2)
        self.fc3 = torch.nn.Linear(N2, Ntargets)
        # Keep data scaling
        self.scaling = scaling
        # TODO: improve evaluation of Npoints
        self.Npoints = int(Nfeatures / Nfields)
        # TODO: better place to define mse_loss
        self.mse_loss = torch.nn.MSELoss(reduction = 'mean')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

### The Optimizer ### 
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.05)#l_rate) # TODO: should be a parameter
        optimizer = torch.optim.SGD(self.parameters(), lr=0.05)#l_rate) # TODO: should be a parameter
        return optimizer

### Training ### 
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Evaluate physical model using data scaling
        logits = self.forward(x)
        # Evaluate loss comparing to the kinetic heat flux in y
        loss = self.mse_loss(logits, y)
        # Add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

### Validation ### 
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Evaluate physical model using data scaling
        logits = self.forward(x)
        # Evaluate loss comparing to the kinetic heat flux in y
        loss = self.mse_loss(logits, y)
        return {'val_loss': loss}

    # Define validation epoch end
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


class DirectModel(AlphaModel): 

    def __init__(self, Nfeatures, N1, N2, scaling, Nfields):
        super(DirectModel, self).__init__(Nfeatures, N1, N2, 2, scaling, Nfields)

    def scaled_heatflux_model(self, x):
        Q = self.forward(x)[:, 0]
        return Q

    def scaled_nln_model(self, x):
        print(self.forward(x).size())
        nln = self.forward(x)[:, 1]
        return nln
    
    def heatflux_model(self, x):
        mean = self.scaling['Q']['mean']
        std = self.scaling['Q']['std']
        return self.scaled_heatflux_model(x) * std + mean    
    
    def local_heatflux_model(self, x):
        # Extract features for modified local heat flux
        # The order MUST correspond to 'generate_QimpactTrainingData.py' T, gradT, Z, n
        featureName = ['T', 'gradT', 'Z']
        feature = {}
        ip = int(self.Npoints / 2)
        i = 0
        for name in featureName:
            mean, std = self.scaling[name]['mean'], self.scaling[name]['std']
            # Batch values of the feature at xc
            feature[name] = x[:, i * self.Npoints + ip] * std + mean
            i = i+1
        # Get local heat flux if no large Kn in the kernel interval
        Z = feature['Z']; T = feature['T']; gradT = feature['gradT']
        kQSH = 6.1e+02 # scaling constant consistent with SCHICK
        ### TODO
        # Workaround to get proper tensor dimensions
        local_heatflux_model = - kQSH / Z[:] * ((Z[:] + 0.24) / (Z[:] + 4.2))\
          * T[:]**2.5 * gradT[:]
        ###
        return local_heatflux_model