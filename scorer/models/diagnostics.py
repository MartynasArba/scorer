import matplotlib.pyplot as plt
from scipy.signal import welch
import torch
from scipy.stats import kurtosis
import numpy as np
from sklearn.decomposition import PCA
from scorer.models.sleep_cnn import SCDSSleepCNN

#CHECK EMBEDDINGS

good_path = r"C:\Users\marty\Desktop\train_sets\final_test\F\windowed_pilot_ch0\X_pilot_ch0_chunk0.pt"
ood_path = r"G:\sleep-ecog-DOWNSAMPLED\20251124-1_g0_t0obx0obx_box3\processed\windowed_20260303164717 20251124-1_g0_t0.obx0.obx_box3\X_chunk0.pt"

cnn = SCDSSleepCNN(num_classes=3).to('cuda')
cnn.load_state_dict(torch.load(r"C:\Users\marty\Projects\scorer\scorer\models\weights\Aligned_Encoder_20260427.pt")) 
cnn.eval()

x_good = torch.load(good_path)[0:1, :, :].permute(1, 0, 2).float()
x_bad = torch.load(ood_path)[0:1, :, :].permute(1, 0, 2).float()

x_good = (x_good - x_good.mean())/x_good.std()
x_bad = (x_bad - x_bad.mean())/x_bad.std()
print(x_bad.size())

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


#CHECK FFTS AND RAW SIGNALS

x_train = torch.load(good_path).detach().cpu().numpy()[0, :, :].ravel()
x_ood = torch.load(ood_path).detach().cpu().numpy()[0, :, :].ravel()

x_train = (x_train - x_train.mean())/x_train.std()
x_ood = (x_ood - x_ood.mean())/x_ood.std()

#clipping check
good_clip_pct = np.mean(x_train == np.max(x_train)) * 100
bad_clip_pct = np.mean(x_ood == np.max(x_ood)) * 100

print(f"Good Data Clipping: {good_clip_pct:.4f}% | Kurtosis: {kurtosis(x_train.flatten()):.2f}")
print(f"Bad Data Clipping:  {bad_clip_pct:.4f}% | Kurtosis: {kurtosis(x_ood.flatten()):.2f}")

f_train, pxx_train = welch(x_train[:2000], fs=250, nperseg=256)
f_ood, pxx_ood = welch(x_ood[:2000], fs=250, nperseg=256)

plt.figure(figsize=(10,4))
plt.plot(f_train, pxx_train, label='Training Data')
plt.plot(f_ood, pxx_ood, label='OOD Data')
plt.xlim(0, 20)
plt.legend()
plt.title("Frequency Distribution Mismatch Check")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(x_train[:2000] - 1.5, label='Training Data')
plt.plot(x_ood[:2000] + 1.5, label='OOD Data')
plt.title("timeseries comparison")
plt.show()

#CHECK THETA-DELTA RATIO
def get_tdr(data, sr=250):
    f, pxx = welch(data, fs=sr, nperseg=256, axis=-1)
    
    delta_mask = (f >= 1) & (f <= 4)
    theta_mask = (f >= 6) & (f <= 9)
    
    delta_power = np.sum(pxx[..., delta_mask], axis=-1)
    theta_power = np.sum(pxx[..., theta_mask], axis=-1)
    
    return theta_power / (delta_power + 1e-9)

good_tdr = get_tdr(x_train)
bad_tdr = get_tdr(x_ood)

plt.hist(good_tdr.flatten()[:100], bins=50, alpha=0.5, density=True, label='Good Data')
plt.hist(bad_tdr.flatten()[:100], bins=50, alpha=0.5, density=True, label='OOD Data')
plt.legend()
plt.title("Theta/Delta Ratio Distribution")
plt.show()

# #check for sample rate mismatch - it's probably ok
# import pandas as pd
# df = pd.read_csv(r"G:\sleep-ecog-DOWNSAMPLED\20260327-1_g0_t0obx0obx_box2\20260327-1_g0_t0.obx0.obx_box2.csv")
# #should be around 19 to 10 based on motionsensor csv
# known_duration_seconds = 15* 60 * 60
# actual_sample_rate = len(df) / known_duration_seconds
# print(f"The actual sample rate of the CSV is: {actual_sample_rate} Hz")