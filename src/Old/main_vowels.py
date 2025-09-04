from parselmouth.praat import call
import soundfile as sf
import pandas as pd
import numpy as np
import parselmouth
import wave
import os


general_path = os.path.dirname(os.getcwd())
data_path = os.path.join(general_path, 'Data')
feature_path = os.path.join(general_path, 'Features')
if not os.path.exists(feature_path):
    os.makedirs(feature_path)

info_path = os.path.join(general_path, 'VOC-ALS.xlsx')
info = pd.read_excel(info_path)
sex = info['Sex'].values
catergory = info['Category'].values

timings_path = os.path.join(general_path, 'timings')

result = []

count = 0
for folder in os.listdir(data_path):
    if folder.startswith('phonation'):
        phonation_path = os.path.join(data_path, folder)
        timings = os.path.join(timings_path, folder+'.xlsx')
        timings_table = pd.read_excel(timings)

        for i, file in enumerate(os.listdir(phonation_path)):
            print(file[:-4])
            name = file[:5]
            task = folder[-1]
            gender = sex[i]
            cat = catergory[i]

            wav_file = os.path.join(phonation_path, file)
            wf = wave.open(wav_file, "rb")

            ind = np.where(timings_table['ID'] == file[:5])[0]
            start = timings_table['Start'].values[ind][0]
            end = timings_table['Stop'].values[ind][0]

            min_freq = 50 if sex[i] == 'M' else 100
            max_freq = 600 if sex[i] == 'M' else 800

            snd = parselmouth.Sound(wav_file)
            point_process = call(snd, "To PointProcess (periodic, cc)", min_freq, max_freq)
            pitch = call(snd, "To Pitch (cc)", 0.0, min_freq, 15, False, 0.03, 0.45, 0.01, 0.35, 0.14, max_freq)

            # Temporal features
            duration = end - start

            # Extract central 2/4 of the duration
            start_central = start + duration / 4
            end_central = end - duration / 4

            # F0 metrics
            f0_mean = call(pitch, "Get mean", start_central, end_central, "Hertz")
            f0_std = call(pitch, "Get standard deviation", start_central, end_central, "Hertz")
            f0_min = call(pitch, "Get minimum", start_central, end_central, "Hertz", "parabolic")
            f0_max = call(pitch, "Get maximum", start_central, end_central, "Hertz", "parabolic")
            f0_median = call(pitch, "Get quantile", start_central, end_central, 0.5, "Hertz")
            f0_25 = call(pitch, "Get quantile", start_central, end_central, 0.25, "Hertz")
            f0_75 = call(pitch, "Get quantile", start_central, end_central, 0.75, "Hertz")

            # JItter metrics                             
            jitter_local = call(point_process, "Get jitter (local)", start_central, end_central, 0.0001, 1/min_freq, 1.3)
            jitter_local_absolute = call(point_process, "Get jitter (local, absolute)", start_central, end_central, 0.0001, 1/min_freq, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", start_central, end_central, 0.0001, 1/min_freq, 1.3)
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", start_central, end_central, 0.0001, 1/min_freq, 1.3)
            jitter_ddp = call(point_process, "Get jitter (ddp)", start_central, end_central, 0.0001, 1/min_freq, 1.3)

            # Shimmer metrics
            shimmer_local = call([snd,point_process], "Get shimmer (local)", start_central, end_central, 0.0001, 1/min_freq, 1.3, 1.6)
            shimmer_local_dB = call([snd,point_process], "Get shimmer (local_dB)", start_central, end_central, 0.0001, 1/min_freq, 1.3, 1.6)
            shimmer_apq3 = call([snd,point_process], "Get shimmer (apq3)", start_central, end_central, 0.0001, 1/min_freq, 1.3, 1.6)
            shimmer_apq5 = call([snd,point_process], "Get shimmer (apq5)", start_central, end_central, 0.0001, 1/min_freq, 1.3, 1.6)
            shimmer_apq11 = call([snd,point_process], "Get shimmer (apq11)", start_central, end_central, 0.0001, 1/min_freq, 1.3, 1.6)
            shimmer_dda = call([snd,point_process], "Get shimmer (dda)", start_central, end_central, 0.0001, 1/min_freq, 1.3, 1.6)

            # Harmonicity metrics
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, min_freq, 0.1, 4.5) #take mean, std, range
            hnr_mean = call(harmonicity, "Get mean", start_central, end_central)
            hnr_std = call(harmonicity, "Get standard deviation", start_central, end_central)
            hnr_min = call(harmonicity, "Get minimum", start_central, end_central, 'parabolic')
            hnr_max = call(harmonicity, "Get maximum", start_central, end_central, 'parabolic')

            # Cepstral Peak Prominence
            snd_vowel = snd.extract_part(start_central, end_central)
            spectrum = call(snd_vowel, "To Spectrum")
            cepstrum = call(spectrum, "To PowerCepstrum")
            cpp = call(cepstrum, "Get peak prominence", min_freq, max_freq, "parabolic", 0.001, 0.05, "Exponential decay", "Robust slow")
            cp = call(cepstrum, "Get peak", min_freq, max_freq, "parabolic")

            # Formants
            formant_max = 5000 if sex[i] == 'M' else 5500
            formant = call(snd, "To Formant (burg)", 0.0, 5.0, formant_max, 0.025, 50)

            f1_mean = call(formant, "Get mean", 1, start_central, end_central, "Hertz")
            f2_mean = call(formant, "Get mean", 2,  start_central, end_central, "Hertz")
            f3_mean = call(formant, "Get mean", 3, start_central, end_central, "Hertz")
            f1_std = call(formant, "Get standard deviation", 1, start_central, end_central, "Hertz")
            f2_std = call(formant, "Get standard deviation", 2,  start_central, end_central, "Hertz")
            f3_std = call(formant, "Get standard deviation", 3, start_central, end_central, "Hertz")
            f1_median = call(formant, "Get quantile", 1, start_central, end_central, "Hertz", 0.50)
            f2_median = call(formant, "Get quantile", 2, start_central, end_central, "Hertz", 0.50)
            f3_median = call(formant, "Get quantile", 3, start_central, end_central, "Hertz", 0.50)
            f1_max = call(formant, "Get maximum", 1, start_central, end_central, "Hertz", "parabolic")
            f2_max = call(formant, "Get maximum", 2, start_central, end_central, "Hertz", "parabolic")
            f3_max = call(formant, "Get maximum", 3, start_central, end_central, "Hertz", "parabolic")
            f1_min = call(formant, "Get minimum", 1, start_central, end_central, "Hertz", "parabolic")
            f2_min = call(formant, "Get minimum", 2, start_central, end_central, "Hertz", "parabolic")
            f3_min = call(formant, "Get minimum", 3, start_central, end_central, "Hertz", "parabolic")

            # MFCC features
            mfccs_sentence = call(snd_vowel, "To MFCC", 12, 0.015, 0.005, 100, 100, 0.0)
            mfccs_array =  mfccs_sentence.to_array()  
            mfccs_delta = np.diff(mfccs_array, axis=1)
            mfccs_delta_delta = np.diff(mfccs_delta, axis=1)

            mfcc_mean = np.mean(mfccs_array, axis=1)
            mfcc_std = np.std(mfccs_array, axis=1)
            mfccs_delta_mean = np.mean(mfccs_delta, axis=1)
            mfccs_delta_std = np.std(mfccs_delta, axis=1)
            mfccs_delta_delta_mean = np.mean(mfccs_delta_delta, axis=1)
            mfccs_delta_delta_std = np.std(mfccs_delta_delta, axis=1)

            result.append({
                'name': name,
                'task': task,
                'category': cat,
                'sex': gender,
                'duration': duration,
                'f0 mean': f0_mean,
                'f0 std': f0_std,
                'f0 max': f0_max,
                'f0 min': f0_min,
                'f0 median': f0_median,
                'f0 25': f0_25,
                'f0 75': f0_75,
                'jitter local': jitter_local,
                'jitter local absolute': jitter_local_absolute,
                'jitter rap': jitter_rap,
                'jitter ppq5': jitter_ppq5,
                'jitter ddp': jitter_ddp,
                'shimmer local': shimmer_local,
                'shimmer local dB': shimmer_local_dB,
                'shimmer apq3': shimmer_apq3,
                'shimmer apq5': shimmer_apq5,
                'shimmer apq11': shimmer_apq11,
                'shimmer dda': shimmer_dda,
                'hnr mean': hnr_mean,
                'hnr std': hnr_std,
                'hnr min': hnr_min,
                'hnr max': hnr_max,
                'cpp': cpp,
                'cp': cp,
                'f1 mean': f1_mean,
                'f1 std': f1_std,
                'f1 min': f1_min,
                'f1 max': f1_max,
                'f1 median': f1_median,
                'f2 mean': f2_mean,
                'f2 std': f2_std,
                'f2 min': f2_min,
                'f2 max': f2_max,
                'f2 median': f2_median,
                'f3 mean': f3_mean,
                'f3 std': f3_std,
                'f3 min': f3_min,
                'f3 max': f3_max,
                'f3 median': f3_median,
                'mfcc0 mean': mfcc_mean[0],
                'mfcc0 std': mfcc_std[0],
                'mfcc0 delta mean': mfccs_delta_mean[0],
                'mfcc0 delta std': mfccs_delta_std[0],
                'mfcc0 delta delta mean': mfccs_delta_delta_mean[0],
                'mfcc0 delta delta std': mfccs_delta_delta_std[0],
                'mfcc1 mean': mfcc_mean[1],
                'mfcc1 std': mfcc_std[1],
                'mfcc1 delta mean': mfccs_delta_mean[1],
                'mfcc1 delta std': mfccs_delta_std[1],
                'mfcc1 delta delta mean': mfccs_delta_delta_mean[1],
                'mfcc1 delta delta std': mfccs_delta_delta_std[1],
                'mfcc2 mean': mfcc_mean[2],
                'mfcc2 std': mfcc_std[2],
                'mfcc2 delta mean': mfccs_delta_mean[2],
                'mfcc2 delta std': mfccs_delta_std[2],
                'mfcc2 delta delta mean': mfccs_delta_delta_mean[2],
                'mfcc2 delta delta std': mfccs_delta_delta_std[2],
                'mfcc3 mean': mfcc_mean[3],
                'mfcc3 std': mfcc_std[3],
                'mfcc3 delta mean': mfccs_delta_mean[3],
                'mfcc3 delta std': mfccs_delta_std[3],
                'mfcc3 delta delta mean': mfccs_delta_delta_mean[3],
                'mfcc3 delta delta std': mfccs_delta_delta_std[3],
                'mfcc4 mean': mfcc_mean[4],
                'mfcc4 std': mfcc_std[4],
                'mfcc4 delta mean': mfccs_delta_mean[4],
                'mfcc4 delta std': mfccs_delta_std[4],
                'mfcc4 delta delta mean': mfccs_delta_delta_mean[4],
                'mfcc4 delta delta std': mfccs_delta_delta_std[4],
                'mfcc5 mean': mfcc_mean[5],
                'mfcc5 std': mfcc_std[5],
                'mfcc5 delta mean': mfccs_delta_mean[5],
                'mfcc5 delta std': mfccs_delta_std[5],
                'mfcc5 delta delta mean': mfccs_delta_delta_mean[5],
                'mfcc5 delta delta std': mfccs_delta_delta_std[5],
                'mfcc6 mean': mfcc_mean[6],
                'mfcc6 std': mfcc_std[6],
                'mfcc6 delta mean': mfccs_delta_mean[6],
                'mfcc6 delta std': mfccs_delta_std[6],
                'mfcc6 delta delta mean': mfccs_delta_delta_mean[6],
                'mfcc6 delta delta std': mfccs_delta_delta_std[6],
                'mfcc7 mean': mfcc_mean[7],
                'mfcc7 std': mfcc_std[7],
                'mfcc7 delta mean': mfccs_delta_mean[7],
                'mfcc7 delta std': mfccs_delta_std[7],
                'mfcc7 delta delta mean': mfccs_delta_delta_mean[7],
                'mfcc7 delta delta std': mfccs_delta_delta_std[7],
                'mfcc8 mean': mfcc_mean[8],
                'mfcc8 std': mfcc_std[8],
                'mfcc8 delta mean': mfccs_delta_mean[8],
                'mfcc8 delta std': mfccs_delta_std[8],
                'mfcc8 delta delta mean': mfccs_delta_delta_mean[8],
                'mfcc8 delta delta std': mfccs_delta_delta_std[8],
                'mfcc9 mean': mfcc_mean[9],
                'mfcc9 std': mfcc_std[9],
                'mfcc9 delta mean': mfccs_delta_mean[9],
                'mfcc9 delta std': mfccs_delta_std[9],
                'mfcc9 delta delta mean': mfccs_delta_delta_mean[9],
                'mfcc9 delta delta std': mfccs_delta_delta_std[9],
                'mfcc10 mean': mfcc_mean[10],
                'mfcc10 std': mfcc_std[10],
                'mfcc10 delta mean': mfccs_delta_mean[10],
                'mfcc10 delta std': mfccs_delta_std[10],
                'mfcc10 delta delta mean': mfccs_delta_delta_mean[10],
                'mfcc10 delta delta std': mfccs_delta_delta_std[10],
                'mfcc11 mean': mfcc_mean[10],
                'mfcc11 std': mfcc_std[10],
                'mfcc11 delta mean': mfccs_delta_mean[10],
                'mfcc11 delta std': mfccs_delta_std[10],
                'mfcc11 delta delta mean': mfccs_delta_delta_mean[10],
                'mfcc11 delta delta std': mfccs_delta_delta_std[10],
                'mfcc12 mean': mfcc_mean[11],
                'mfcc12 std': mfcc_std[11],
                'mfcc12 delta mean': mfccs_delta_mean[11],
                'mfcc12 delta std': mfccs_delta_std[11],
                'mfcc12 delta delta mean': mfccs_delta_delta_mean[11],
                'mfcc12 delta delta std': mfccs_delta_delta_std[11],
            })

            result_df = pd.DataFrame(result)
            result_df.to_excel(os.path.join(feature_path,'vowel_features.xlsx'), index=False)
