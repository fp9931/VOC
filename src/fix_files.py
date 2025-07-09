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

vocali_path = "C:/Users/mimos/Projects/VOC/Results/phonation_features.xlsx"
df_vocali = pd.read_excel(vocali_path)


list_of_files = os.listdir(data_path)
list_of_files = natsorted(list_of_files)
i = 0
for aux, file in enumerate(list_of_files):
    if file.endswith('.wav'):
        file_path = os.path.join(data_path, file)
        wav_file = os.path.join(file_path)
        wf = wave.open(wav_file, "rb")
        name_aux = file.split('.')[0]
        name_subj = name_aux.split('_')[0]

        # Trovare dove si trova il soggetto corrente, contare quante ripetizioni


