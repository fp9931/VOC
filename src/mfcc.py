import soundfile as sf
import pandas as pd
import numpy as np
import os
import librosa
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from natsort import natsorted

general_path = os.path.dirname(os.getcwd())
data_path = os.path.join(general_path, 'Results', 'burst_time')
result_path = os.path.join(general_path, 'Results')
if not os.path.exists(result_path):
    os.makedirs(result_path)

info_path = os.path.join(general_path, 'VOC-ALS.xlsx')
info = pd.read_excel(info_path)
sex = info['Sex'].values
catergory = info['Category'].values

timings_path = os.path.join("C:/Users/mimos/Projects/VOC/Results/test_burst","VOT.xlsx")
timings = pd.read_excel(timings_path)
starts = timings['Start'].values
stops = timings['Stop'].values

mfcc_0 = []
mfcc_0 = []
mfcc_1 = []
mfcc_2 = []
mfcc_3 = []
mfcc_4 = []
mfcc_5 = []
mfcc_6 = []
mfcc_7 = []
mfcc_8 = []
mfcc_9 = []
mfcc_10 = []
mfcc_11 = []
mfcc_12 = []
name = []

list_of_files = os.listdir(data_path)
list_of_files = natsorted(list_of_files)
i = 0
for aux, file in enumerate(list_of_files):
    if file.endswith('.wav'):
        file_path = os.path.join(data_path, file)
        audio, sr = librosa.load(file_path)
        name.append(file.split('.')[0])
        print(name[-1])

        y = audio[int(stops[i]*sr):int((stops[i] + 0.2)* sr)]
        i += 1

        if len(y) < int(0.02 * sr):
            segment = np.pad(y, (0, int(0.02 * sr) - len(y)))

        pre_emphasis = 0.97
        y_preemphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        y_preemphasized *= np.hamming(len(y_preemphasized)) 

        NFFT = 256
        mag = np.absolute(np.fft.rfft(y_preemphasized, NFFT)) 
        pow = ((1.0 / NFFT) * ((mag) ** 2))  # Power Spectrum
        
        nfilt = 40
        low_freq = 0
        high_freq = sr / 2
        low_mel = 1125 * np.log(1 + low_freq / 700)  # Convert Hz to Mel
        high_mel = 1125 * np.log(1 + high_freq / 700)
        mel_points = np.linspace(low_mel, high_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = 700 * (np.exp(mel_points / 2595) - 1)  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / sr).astype(int)  # Convert to bin numbers

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        mel_spec = np.dot(pow, fbank.T)
        mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)  # Numerical stability
        log_mel_spec = np.log(mel_spec)

        mfcc = dct(log_mel_spec)[:12]

        low_energy = np.log(np.sum(pow) if np.sum(pow) > 0 else np.finfo(float).eps)
        mfcc_0.append(mfcc[0])
        mfcc_1.append(mfcc[1])
        mfcc_2.append(mfcc[2])
        mfcc_3.append(mfcc[3])
        mfcc_4.append(mfcc[4])
        mfcc_5.append(mfcc[5])
        mfcc_6.append(mfcc[6])
        mfcc_7.append(mfcc[7])
        mfcc_8.append(mfcc[8])
        mfcc_9.append(mfcc[9])
        mfcc_10.append(mfcc[10])
        mfcc_11.append(mfcc[11])
        mfcc_12.append(low_energy)

        # if i == 10:
        #     break

        # Save the MFCC features to a CSV file
        mfcc_df = pd.DataFrame({
            'Name': name,
            'mfcc_0': mfcc_0,
            'mfcc_1': mfcc_1,
            'mfcc_2': mfcc_2,
            'mfcc_3': mfcc_3,
            'mfcc_4': mfcc_4,
            'mfcc_5': mfcc_5,
            'mfcc_6': mfcc_6,
            'mfcc_7': mfcc_7,
            'mfcc_8': mfcc_8,
            'mfcc_9': mfcc_9,
            'mfcc_10': mfcc_10,
            'mfcc_11': mfcc_11,
            'low_energy': mfcc_12
        })
        mfcc_df.to_excel(os.path.join(result_path, f'mfcc_syllables.xlsx'), index=False)

