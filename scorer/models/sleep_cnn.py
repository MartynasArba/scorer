import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepCNN(nn.Module):
    """
    Uses ephys and derived channels (8 total)
    """
    def __init__(self, num_classes: int, mean_std = None):
        super().__init__()             
    
        self.mean_std = mean_std if mean_std is not None else None  #standardize
        
        # "Feature extraction"
        self.conv1 = nn.Conv1d(8, 16, kernel_size = 11, padding = 5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size = 5, padding = 2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
        
        self.pool = nn.MaxPool1d(2) #maxpooling should also improve overfitting, but at this stage might not be worth it
        self.dropout = nn.Dropout(0.5) #try to prevent overfitting
        
        # Global average pooling to reduce parameters after convolutions
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Smaller FC layers - classifier
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        
        if self.mean_std is not None:
            mean, std = self.mean_std
            x = (x - mean[None, :, None]) / std[None, :, None]
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class EphysSleepCNN(nn.Module):
    def __init__(self, num_classes: int, mean_std = None):
        super().__init__()             
    
        self.mean_std = mean_std if mean_std is not None else None  #standardize
        
        # "Feature extraction"
        self.conv1 = nn.Conv1d(2, 16, kernel_size = 11, padding = 5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size = 5, padding = 2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
        
        self.pool = nn.MaxPool1d(2) #maxpooling should also improve overfitting, but at this stage might not be worth it
        self.dropout = nn.Dropout(0.5) #try to prevent overfitting
        
        # Global average pooling to reduce parameters after convolutions
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Smaller FC layers - classifier
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):               
        #ephys only
        x = x[:, :2, :]
        
        if self.mean_std is not None:
            mean, std = self.mean_std
            mean = mean[:2]
            std  = std[:2]
            x = (x - mean[None, :, None]) / std[None, :, None]
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
class FreqSleepCNN(nn.Module):
    """
    Uses FFT of ephys channels (0 and 1)
    """
    def __init__(self, num_classes: int):
        super().__init__()             

        # "Feature extraction"
        self.conv1 = nn.Conv1d(2, 16, kernel_size = 11, padding = 5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size = 5, padding = 2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
        
        self.pool = nn.MaxPool1d(2) #maxpooling should also improve overfitting, but at this stage might not be worth it
        self.dropout = nn.Dropout(0.5) #try to prevent overfitting
        
        # Global average pooling to reduce parameters after convolutions
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Smaller FC layers - classifier
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def _freqtransform(self, x):
        X = torch.fft.rfft(x, dim = -1)
        power = X.real.square() + X.imag.square()
        #drop 0th bin
        power = power[..., 1:]
        #normalize
        power = torch.log1p(power)
        return power
        
    def forward(self, x):               
        #ephys only
        x = x[:, :2, :]
        
        #get fft
        x = self._freqtransform(x)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
if __name__ == "__main__":
    print(torch.__version__)