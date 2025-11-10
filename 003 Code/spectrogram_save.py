import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import librosa
import os

A_data = glob('/root/ssd/Yeonseo/ESS/data/numpy_file_Blind/*')

for A_name in tqdm(A_data, desc = 'data loading'):
    all_file = glob(A_name+'/*.npy')
    
    for data_path in tqdm(all_file, desc = 'path loading'):
        data = np.load(data_path, allow_pickle=True)
        data = data.astype('float32')
        leng = data.shape[0]
        data = data.flatten()
        stft_train = librosa.stft(data, n_fft = 1024, hop_length=leng*2)
        stft_train = stft_train.astype('float32')
        frequency = librosa.fft_frequencies(n_fft = 1024)
        max_freq = 1000
        freq_mask = frequency <= max_freq
        D = librosa.amplitude_to_db(np.abs(stft_train), ref = np.max)
        S_b = D[freq_mask, :]
        librosa.display.specshow(S_b)
        path = f'/root/ssd/Yeonseo/ESS/data/spectrogram_image_1024/Test/{data_path.split("/")[-2]}/'
        os.makedirs(path, exist_ok = True)
        name = data_path.split('/')[-1].split('.')[0]
        plt.savefig(path+name+'.png')
        plt.close()
