import matplotlib.pyplot as plt
from scipy.signal import welch
import torch
from scipy.stats import kurtosis
import numpy as np
from sklearn.decomposition import PCA
from scorer.models.sleep_cnn import SCDSSleepCNN

from scorer.data.loaders import robust_normalize

import scipy.signal as sig
import seaborn as sns
import torch

def plot_embeddings(x_good, x_bad):
    
    cnn = SCDSSleepCNN(num_classes=3).to('cuda')
    cnn.load_state_dict(torch.load(r"C:\Users\marty\Projects\scorer\scorer\models\weights\adversarial_adjusted_encoder20260429.pt")) 
    cnn.eval()

    with torch.no_grad():
        # pass features through cnn
        good_features = cnn(x_good.to('cuda')).cpu().numpy()
        bad_features = cnn(x_bad.to('cuda')).cpu().numpy()
        
    pca = PCA(n_components=2)
    all_features = np.vstack([good_features, bad_features])
    pca_result = pca.fit_transform(all_features)
    plt.figure(figsize=(8, 8))
    plt.scatter(pca_result[:1000, 0], pca_result[:1000, 1], alpha=0.5, label='Good Data', s=10)
    plt.scatter(pca_result[1000:, 0], pca_result[1000:, 1], alpha=0.5, label='OOD Data', s=10)
    plt.legend()
    plt.title("CNN Latent Space Comparison")
    plt.show()


def plot_domain_diagnostics(clean_data, ood_data, fs=250.0):
    """
    clean_data: A clean tensor from validation set [Channels, Time]
    ood_data: The processed tensor from your OOD inference set [Channels, Time]
    """
    clean_flat = clean_data.cpu().numpy().flatten()
    ood_flat = ood_data.cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Amplitude Distribution (Histogram)
    sns.kdeplot(clean_flat, label="Clean Validation", color="blue", ax=axes[0])
    sns.kdeplot(ood_flat, label="Processed OOD", color="red", ax=axes[0])
    axes[0].set_title("Amplitude Distribution (Density)")
    axes[0].set_xlim([-8, 8])
    axes[0].legend()
    
    # PSD
    # nperseg defines frequency resolution (e.g., 2-4 seconds of data)
    f_clean, pxx_clean = sig.welch(clean_flat, fs=fs, nperseg=int(fs*2))
    f_ood, pxx_ood = sig.welch(ood_flat, fs=fs, nperseg=int(fs*2))
    
    axes[1].semilogy(f_clean, pxx_clean, label="Clean Validation", color="blue")
    axes[1].semilogy(f_ood, pxx_ood, label="Processed OOD", color="red")
    axes[1].set_title("Power Spectral Density (PSD)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_xlim([0, 50])
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def check_tensors(raw_tensor, processed_tensor):
    """
    raw_tensor shape: [10800, 1, 1000]
    processed_tensor shape: [10800, 1, 1000]
    """
    # calculate sd per-window, channel is irrelevant
    raw_std = raw_tensor.std(dim=2).squeeze().cpu().numpy()
    proc_std = processed_tensor.std(dim=2).squeeze().cpu().numpy()
    
    # rolling variance
    plt.figure(figsize=(15, 4))
    plt.plot(raw_std, label='Raw Window Std Dev', alpha=0.6, color='gray')
    plt.plot(proc_std, label='Processed Window Std Dev', alpha=0.9, color='red')
    plt.yscale('log')
    plt.title('Rolling Variance')
    plt.xlabel('Window Index (Chronological)')
    plt.ylabel('Standard Deviation (Log Scale)')
    plt.legend()
    plt.show()
    
    # find "best/worst" windows
    worst_idx = np.argmax(raw_std)
    best_idx = np.argmin(raw_std)
    
    # plot windows
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # "worst"
    axes[0].plot(raw_tensor[worst_idx, 0, :].cpu().numpy(), label='Raw Artifact', color='gray')
    axes[0].plot(processed_tensor[worst_idx, 0, :].cpu().numpy(), label='Filtered Attempt', color='red')
    axes[0].set_title(f'The Poison Pill (Worst Window: Index {worst_idx})')
    axes[0].legend()
    
    # "best"
    axes[1].plot(raw_tensor[best_idx, 0, :].cpu().numpy(), label='Raw Quiet', color='gray')
    axes[1].plot(processed_tensor[best_idx, 0, :].cpu().numpy(), label='Filtered', color='blue')
    axes[1].set_title(f'The Baseline (Quietest Window: Index {best_idx})')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


good_path = r"C:\Users\marty\Desktop\train_sets\final_test\F\windowed_pilot_ch0\X_pilot_ch0_chunk0.pt"
ood_path = r"G:\sleep-ecog-DOWNSAMPLED\20251124-1_g0_t0obx0obx_box3\processed\windowed_20260303164717 20251124-1_g0_t0.obx0.obx_box3\X_chunk0.pt"


x_val_raw = torch.load(good_path)[0:1, :, :].permute(1, 0, 2).float()
x_ood_raw = torch.load(ood_path)[0:1, :, :].permute(1, 0, 2).float()

x_good, x_bad = robust_normalize(x_val_raw), robust_normalize(x_ood_raw, scale = 1.3)

plot_embeddings(x_good, x_bad)

check_tensors(x_ood_raw, x_bad)


clean_data = x_good[0, : , :].flatten()
ood_data =  x_bad[0, : , :].flatten()
plot_domain_diagnostics(clean_data, ood_data, fs=250.0)



# #CHECK FFTS AND RAW SIGNALS

# x_train = torch.load(good_path).detach().cpu().numpy()[0, :, :].ravel()
# x_ood = torch.load(ood_path).detach().cpu().numpy()[0, :, :].ravel()

# x_train = (x_train - x_train.mean())/x_train.std()
# x_ood = (x_ood - x_ood.mean())/x_ood.std()

# #clipping check
# good_clip_pct = np.mean(x_train == np.max(x_train)) * 100
# bad_clip_pct = np.mean(x_ood == np.max(x_ood)) * 100

# print(f"Good Data Clipping: {good_clip_pct:.4f}% | Kurtosis: {kurtosis(x_train.flatten()):.2f}")
# print(f"Bad Data Clipping:  {bad_clip_pct:.4f}% | Kurtosis: {kurtosis(x_ood.flatten()):.2f}")

# f_train, pxx_train = welch(x_train[:2000], fs=250, nperseg=256)
# f_ood, pxx_ood = welch(x_ood[:2000], fs=250, nperseg=256)

# plt.figure(figsize=(10,4))
# plt.plot(f_train, pxx_train, label='Training Data')
# plt.plot(f_ood, pxx_ood, label='OOD Data')
# plt.xlim(0, 20)
# plt.legend()
# plt.title("Frequency Distribution Mismatch Check")
# plt.show()

# plt.figure(figsize=(10,4))
# plt.plot(x_train[:2000] - 1.5, label='Training Data')
# plt.plot(x_ood[:2000] + 1.5, label='OOD Data')
# plt.title("timeseries comparison")
# plt.show()

# #CHECK THETA-DELTA RATIO
# def get_tdr(data, sr=250):
#     f, pxx = welch(data, fs=sr, nperseg=256, axis=-1)
    
#     delta_mask = (f >= 1) & (f <= 4)
#     theta_mask = (f >= 6) & (f <= 9)
    
#     delta_power = np.sum(pxx[..., delta_mask], axis=-1)
#     theta_power = np.sum(pxx[..., theta_mask], axis=-1)
    
#     return theta_power / (delta_power + 1e-9)

# good_tdr = get_tdr(x_train)
# bad_tdr = get_tdr(x_ood)

# plt.hist(good_tdr.flatten()[:100], bins=50, alpha=0.5, density=True, label='Good Data')
# plt.hist(bad_tdr.flatten()[:100], bins=50, alpha=0.5, density=True, label='OOD Data')
# plt.legend()
# plt.title("Theta/Delta Ratio Distribution")
# plt.show()

# #check for sample rate mismatch - it's probably ok
# import pandas as pd
# df = pd.read_csv(r"G:\sleep-ecog-DOWNSAMPLED\20260327-1_g0_t0obx0obx_box2\20260327-1_g0_t0.obx0.obx_box2.csv")
# #should be around 19 to 10 based on motionsensor csv
# known_duration_seconds = 15* 60 * 60
# actual_sample_rate = len(df) / known_duration_seconds
# print(f"The actual sample rate of the CSV is: {actual_sample_rate} Hz")