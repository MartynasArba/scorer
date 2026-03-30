import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepCNN(nn.Module):
    """
    mostly depreciated
    Uses ephys and derived channels (7 total)
    """
    def __init__(self, num_classes: int, mean_std = None):
        super().__init__()             
    
        self.mean_std = mean_std if mean_std is not None else None
        
        # "Feature extraction"
        self.conv1 = nn.Conv1d(7, 16, kernel_size = 11, padding = 5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size = 5, padding = 2)
        self.conv4 = nn.Conv1d(64, 128, kernel_size = 3, padding = 1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
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
    def __init__(self, num_classes: int):
        super().__init__()            
        
        self.conv1 = nn.Conv1d(2, 32, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2) 
        self.dropout = nn.Dropout(0.5)
        
        self.pool5 = nn.MaxPool1d(2)
        
        self.feature_proj = nn.Linear(256 * 31, 128)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):               
        x = x[:, :2, :]

        #standardize by batch, but use median scaling 
        B, C, T = x.shape
        x_flat = x.transpose(0, 1).reshape(C, -1)
        batch_median  = torch.quantile(x_flat, q = 0.5, dim = 1).view(1, C, 1)
        batch_q1 = torch.quantile(x_flat, q = 0.25, dim = 1).view(1, C, 1)
        batch_q3 = torch.quantile(x_flat, q = 0.75, dim = 1).view(1, C, 1)
        batch_iqr = (batch_q3 - batch_q1) + 1e-8
        x = (x - batch_median) / batch_iqr
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.feature_proj(x))
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """return features only for contrastive learning"""             
        x = x[:, :2, :]
        
        #standardize by batch, but use median scaling 
        B, C, T = x.shape
        x_flat = x.transpose(0, 1).reshape(C, -1)       
        batch_median  = torch.quantile(x_flat, q = 0.5, dim = 1).view(1, C, 1)
        batch_q1 = torch.quantile(x_flat, q = 0.25, dim = 1).view(1, C, 1)
        batch_q3 = torch.quantile(x_flat, q = 0.75, dim = 1).view(1, C, 1)
        batch_iqr = (batch_q3 - batch_q1) + 1e-8
        x = (x - batch_median) / batch_iqr
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)  
        x = F.relu(self.feature_proj(x))
        
        return x

class DualStreamSleepCNN(nn.Module):
    """
    two-stream model: combining phase-invariant FFT power with time-domain features
    """
    def __init__(self, num_classes: int):
        super().__init__()             
        
        #time series stream
        self.time_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1) # flattens to [batch, 256]
        )
        
        # freq domain stream, architecture is basically the same
        self.freq_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1) # flattens to [batch, 256]
        )
        
        # fusion, classification
        self.dropout = nn.Dropout(0.5)
        
        self.feature_proj = nn.Linear(512, 128) #added bottleneck to support contrastive learning
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def _robust_scale(self, x):
        """median-based batch scaling for the time stream"""
        B, C, T = x.shape
        x_flat = x.transpose(0, 1).reshape(C, -1) 
        median = torch.quantile(x_flat, q=0.50, dim=1).view(1, C, 1)
        q75 = torch.quantile(x_flat, q=0.75, dim=1).view(1, C, 1)
        q25 = torch.quantile(x_flat, q=0.25, dim=1).view(1, C, 1)
        iqr = (q75 - q25) + 1e-8 
        return (x - median) / iqr
        
    def _freqtransform(self, x):
        """log-power spectrum"""
        X = torch.fft.rfft(x, dim=-1)
        power = X.real.square() + X.imag.square()
        power = power[..., 1:] # drop 0th (DC) bin to prevent baseline spikes
        return torch.log1p(power) # Log scale for stable gradients
    
    def get_features(self, x):               
        # take ephys channels
        x = x[:, :2, :]
        
        x_time = self._robust_scale(x)
        feat_time = self.time_conv(x_time).squeeze(-1) # shape: [batch, 256]
        
        x_freq = self._freqtransform(x)
        feat_freq = self.freq_conv(x_freq).squeeze(-1) # shape: [batch, 256]
        
        fused = torch.cat([feat_time, feat_freq], dim=1)  # shape: [batch, 512]
        
        # fuse
        fused = torch.cat([feat_time, feat_freq], dim=1)  # shape: [batch, 128]
        #pass through projection 
        fused = F.relu(self.feature_proj(fused))
        
        return fused
        
    def forward(self, x):               
        
        fused = self.get_features(x)
        
        # classification head
        out = self.dropout(fused)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class SCDSSleepCNN(nn.Module):
    """
    Alternative single channel dual stream sleep CNN
    two-stream model: combining phase-invariant FFT power with time-domain features
    """
    def __init__(self, num_classes: int):
        super().__init__()             
        
        #time series stream
        self.time_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1) # flattens to [batch, 256]
        )
        
        # freq domain stream, architecture is basically the same
        self.freq_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1) # flattens to [batch, 256]
        )
        
        # fusion, classification
        self.dropout = nn.Dropout(0.5)
        
        self.feature_proj = nn.Linear(512, 128) #added bottleneck to support contrastive learning
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def _robust_scale(self, x):
        """median-based batch scaling for the time stream"""
        B, C, T = x.shape
        x_flat = x.transpose(0, 1).reshape(C, -1) 
        median = torch.quantile(x_flat, q=0.50, dim=1).view(1, C, 1)
        q75 = torch.quantile(x_flat, q=0.75, dim=1).view(1, C, 1)
        q25 = torch.quantile(x_flat, q=0.25, dim=1).view(1, C, 1)
        iqr = (q75 - q25) + 1e-8 
        return (x - median) / iqr
        
    def _freqtransform(self, x):
        """log-power spectrum"""
        X = torch.fft.rfft(x, dim=-1)
        power = X.real.square() + X.imag.square()
        power = power[..., 1:] # drop 0th (DC) bin to prevent baseline spikes
        return torch.log1p(power) # Log scale for stable gradients
    
    def get_features(self, x):               
        # take SINGLE channel
        x = x[:, :1, :]
        
        x_time = self._robust_scale(x)
        feat_time = self.time_conv(x_time).squeeze(-1) # shape: [batch, 256]
        
        x_freq = self._freqtransform(x)
        feat_freq = self.freq_conv(x_freq).squeeze(-1) # shape: [batch, 256]
        
        fused = torch.cat([feat_time, feat_freq], dim=1)  # shape: [batch, 512]
        
        # fuse
        fused = torch.cat([feat_time, feat_freq], dim=1)  # shape: [batch, 128]
        #pass through projection 
        fused = F.relu(self.feature_proj(fused))
        
        return fused
        
    def forward(self, x):               
        
        fused = self.get_features(x)
        
        # classification head
        out = self.dropout(fused)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    print(torch.__version__)