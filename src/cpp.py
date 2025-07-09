import soundfile as sf
import pandas as pd
import numpy as np
import os
import librosa
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from natsort import natsorted
from parselmouth.praat import call
import parselmouth
import wave

general_path = os.path.dirname(os.getcwd())
data_path = os.path.join(general_path, 'Results', 'burst_time')
result_path = os.path.join(general_path, 'Results')
if not os.path.exists(result_path):
    os.makedirs(result_path)

info_path = os.path.join(general_path, 'VOC-ALS.xlsx')
info = pd.read_excel(info_path)
sex = info['Sex'].values
catergory = info['Category'].values
ID = info['ID'].values

timings_path = os.path.join("C:/Users/mimos/Projects/VOC/Results/test_burst","VOT_2.xlsx")
timings = pd.read_excel(timings_path)
starts = timings['Start'].values
stops = timings['Stop'].values

cp = []
cpp = []
name = []

list_of_files = os.listdir(data_path)
list_of_files = natsorted(list_of_files)
i = 0
for aux, file in enumerate(list_of_files):
    if file.endswith('.wav'):
        file_path = os.path.join(data_path, file)
        wav_file = os.path.join(file_path)
        wf = wave.open(wav_file, "rb")
        name_aux = file.split('.')[0]
        name.append(file.split('.')[0])
        name_subj = name_aux.split('_')[0]
        idx = np.where(ID == name_subj)[0]

        min_freq = 50 if sex[idx] == 'M' else 100
        max_freq = 600 if sex[idx] == 'M' else 800

        snd = parselmouth.Sound(wav_file)
        point_process = call(snd, "To PointProcess (periodic, cc)", min_freq, max_freq)
        pitch = call(snd, "To Pitch (cc)", 0.0, min_freq, 15, False, 0.03, 0.45, 0.01, 0.35, 0.14, max_freq)

        # Cepstral Peak Prominence
        spectrum = call(snd, "To Spectrum")
        cepstrum = call(spectrum, "To PowerCepstrum")
        cpp.append(call(cepstrum, "Get peak prominence", min_freq, max_freq, "parabolic", 0.001, 0.05, "Exponential decay", "Robust slow"))
        cp.append(call(cepstrum, "Get peak", min_freq, max_freq, "parabolic"))

        # print(f"File: {file} - CP: {cp[-1]} - CPP: {cpp[-1]}")
        print(i)
        i += 1
            
# Save the MFCC features to a CSV file
cp_df = pd.DataFrame({
    'Name': name,
    'CP': cp,
    'CPP': cpp,
    })
cp_df.to_excel(os.path.join(result_path, 'cpp_syllables.xlsx'), index=False)

