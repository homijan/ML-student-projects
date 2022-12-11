import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import Profiler, AdvancedProfiler
import numpy as np

class AlphaModel(pl.LightningModule): 

### Model ###
    def __init__(self, Nfeatures, N1, N2, Ntargets, scaling):
        # Initialize layers
        self.fc1 = nn.Linear(Nfeatures, N1)
        self.fc2 = nn.Linear(N1, N2)
        self.fc3 = nn.Linear(N2, Ntargets)
        # Keep data scaling
        self.scaling = scaling    

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x    
    
### Data loader ### 
    def train_dataloader(self):
        train_dataset = \
          TensorDataset(torch.tensor(train_features.values).float(),\
            torch.tensor(train_targets[target_fields].values).float())
        train_loader = DataLoader(dataset = train_dataset, batch_size = 128)
        return train_loader
        
    def val_dataloader(self):
        validation_dataset = \
          TensorDataset(torch.tensor(validation_features.values).float(),\
            torch.tensor(validation_targets[target_fields].values).float())
        validation_loader = DataLoader(dataset = validation_dataset,\
          batch_size = 128)
        return validation_loader
    
    def test_dataloader(self):
        test_dataset = \
          TensorDataset(torch.tensor(test_features.values).float(),\
            torch.tensor(test_targets[target_fields].values).float())
        test_loader = DataLoader(dataset = test_dataset, batch_size = 128)
        return test_loader

### The Optimizer ### 
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=l_rate)

### Training ### 
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Evaluate physical model and apply data-scaling
        mean = self.scaling['Qimpact']['mean']
        sted = self.scaling['Qimpact']['std']
        logits = (self.heatflux_model(x) - mean) / std
        # Evaluate loss comparing to the kinetic heat flux in y
        loss = mse_loss(logits, y)
        # Add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

### Validation ### 
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Evaluate physical model and apply data-scaling
        mean = self.scaling['Qimpact']['mean']
        sted = self.scaling['Qimpact']['std']
        logits = (self.heatflux_model(x) - mean) / std
        # Evaluate loss comparing to the kinetic heat flux in y
        loss = mse_loss(logits, y)
        return {'val_loss': loss}

    # Define validation epoch end
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

### Testing ###     
    def test_step(self, batch, batch_idx):
        x, y = batch
        # Evaluate physical model and apply data-scaling
        heatflux_model = self.heatflux_model(x)
        mean = self.scaling['Qimpact']['mean']
        sted = self.scaling['Qimpact']['std']
        logits = (heatflux_model - mean) / std
        # Evaluate loss comparing to the kinetic heat flux in y
        loss = mse_loss(logits, y)
        # Compare model to the kinetic heat flux in y
        correct = torch.sum(logits == y.data)
        
        predictions_pred.append(heatflux_model)
        predictions_actual.append((y.data - Q_mean) / Q_std)
        return {'test_loss': loss, 'test_correct': correct, 'logits': logits}
    
    # Define test end
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}      
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs }


class AlphacModel(AlphaModel): 

    def __init__(self, Nfeatures, N1, N2, scaling):
        super(AlphacModel, self).__init__(Nfeatures, N1, N2, 1, scaling)

    def heatflux_model(self, x):
        # Extract features for modified local heat flux
        featureName = ['Z', 'T', 'gradT', 'Kn', 'n']
        feature = {}
        ip = int(Npoints / 2)
        i = 0
        for name in featureName:
            mean, std = self.scaling[name]['mean'], self.scaling[name]['std']
            # Batch values of the feature at xc
            feature[name] = x[:, i * Npoints + ip] * std + mean
            #feature[name] = x[:, i * Npoints:(i+1) * Npoints] * std + mean
            i = i+1
        # Get alphas
        alphac = self.forward(x)
        # Get local heat flux if no large Kn in the kernel interval
        #Kn = feature['Kn']
        Z = feature['Z']; T = feature['T']; gradT = feature['gradT']
        kQSH = 6.1e+02 # scaling constant consistent with SCHICK
        heatflux_model = - alphac * kQSH / Z * ((Z + 0.24)/(Z + 4.2))\
          * T**2.5 * gradT
        return heatflux_model


class AlphacAlphanModel(AlphaModel): 

    def __init__(self, Nfeatures, N1, N2, scaling):
        super(AlphacAlphanModel, self).__init__(Nfeatures, N1, N2, 2, scaling)

    def heatflux_model(self, x):
        # Extract features for modified local heat flux
        featureName = ['Z', 'T', 'gradT', 'Kn', 'n']
        feature = {}
        ip = int(Npoints / 2)
        i = 0
        for name in featureName:
            mean, std = self.scaling[name]['mean'], self.scaling[name]['std']
            # Batch values of the feature at xc
            feature[name] = x[:, i * Npoints + ip] * std + mean
            #feature[name] = x[:, i * Npoints:(i+1) * Npoints] * std + mean
            i = i+1
        # Get alphas
        alphas = self.forward(x)
        alphac = alphas[:, 0]; alphac = alphan[:, 1]
        # Get local heat flux if no large Kn in the kernel interval
        #Kn = feature['Kn']
        Z = feature['Z']; T = feature['T']; gradT = feature['gradT']
        kQSH = 6.1e+02 # scaling constant consistent with SCHICK
        heatflux_model = - alphac * kQSH / Z * ((Z + 0.24)/(Z + 4.2))\
          * T**(2.5 / (1.0 + np.exp(alphan)) * gradT
        return heatflux_model