import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def extract_features(file_path, sr=22050, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    features = {}
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    features['centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['bw'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    features['tonnetz'] = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))
    features['chroma'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
    features['duration'] = librosa.get_duration(y=y, sr=sr)
    return features

if __name__ == '__main__':
    # Test with single audio
    sample_file = r'C:\Users\Tejash\OneDrive\Desktop\coughvid_20211012\fixed\00a1e1ea-9725-47fb-ad18-b17a30b3a145_fixed.wav'
    feats = extract_features(sample_file)
    print("Features for sample file:")
    for k, v in feats.items():
        print(f"{k}: {v}")

    # Bulk extraction
    audio_dir = r'C:\Users\Tejash\OneDrive\Desktop\coughvid_20211012\fixed'
    data = []
    short_files = []

    for root, dirs, files in os.walk(audio_dir):
        for fname in files:
            if fname.lower().endswith(('.wav', '.ogg')):
                fpath = os.path.join(root, fname)
                try:
                    feats = extract_features(fpath)
                    feats['label'] = os.path.basename(root)
                    feats['file'] = fpath
                    print(f"{fpath}: duration = {feats['duration']:.2f} sec")
                    if feats['duration'] < 0.5:
                        short_files.append(fpath)
                    else:
                        data.append(feats)
                except Exception as e:
                    print(f"Error with {fpath}: {e}")
                    short_files.append(fpath + " (error)")

    # Save/explain results
    if data:
        df = pd.DataFrame(data)
        print("First few rows of extracted features:")
        print(df.head())
        df.to_csv("audio_features_updated.csv", index=False)
        print("Features saved to audio_features_updated.csv")
    else:
        print("No features extracted (check your paths and file formats).")

    if short_files:
        with open("short_files.txt", "w") as f:
            for sf in short_files:
                f.write(sf + "\n")
        print(f"{len(short_files)} files are too short (<0.5s) or errored, see short_files.txt.")
    else:
        print("No short or corrupted files detected.")

