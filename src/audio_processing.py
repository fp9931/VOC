import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from scipy.stats import iqr
import librosa
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import parselmouth
from parselmouth.praat import call

##################### LAVORO CON GLI INDICI CHE FANNO RIFERIMENTO ALL'AUDIO TAGLIATO. Per ricostruire i valori aggiungere start - 0.5 ########################
general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(general_path, 'Data')
timings_path = os.path.join(data_path, 'Timings')
audio_path = os.path.join(data_path, 'Audio')
info = pd.read_excel(os.path.join(data_path, 'Info.xlsx'))

total_iterations = len(info) * 8
progress_bar = tqdm(total=total_iterations, desc="Processing")
progress_bar.set_postfix_str("Starting...")
start_time = time.time()
completed_iterations = 0 
results_path = os.path.join(general_path, 'Results')

features_syll_tot = []
features_vow_tot = []

for folder in os.listdir(audio_path):
    for subj in os.listdir(os.path.join(audio_path, folder)):
            if subj.startswith('CT'):
                continue

            name_file = subj[:-4]
            idx_sex = np.where(info['ID'] == subj[:5])[0]
            if idx_sex.size == 0:
                tqdm.write(f"Subject {subj} not found in info file. Skipping...")
                continue

            audio_file = os.path.join(audio_path, folder, subj)
            idx_sex = np.where(info['ID'] == subj[:5])[0]
            sex = info['Sex'].values[idx_sex]
            age = info['Age'].values[idx_sex]
            onset = info['Onset'].values[idx_sex]
            disease_duration = info['Disease duration'].values[idx_sex]
            speech_score = info['ALSFRS-R_SpeechSubscore'].values[idx_sex]
            swallowing_score = info['ALSFRS-R_SwallowingSubscore'].values[idx_sex]
            bulbar_score = info['PUMNS_BulbarSubscore'].values[idx_sex]

            if sex == 'M':
                pitch_min = 50
                pitch_max = 600
                formant_max = 5000
            else:
                pitch_min = 100
                pitch_max = 800
                formant_max = 5500

            iter_start = time.time()
            tqdm.write(f"Processing {name_file} ...")

            features_sy = []
            if folder.startswith('rhythm'):
                timings = pd.read_excel(os.path.join(timings_path, folder, f"{subj[:5]}.xlsx"))
                start_times = timings['Start'].values
                stop_times = timings['Stop'].values

                valid_indices = ~np.isnan(start_times)
                start_times = start_times[valid_indices]
                stop_times = stop_times[valid_indices]

                sound = parselmouth.Sound(audio_file)
                sr = sound.sampling_frequency
                pitch = call(sound, "To Pitch (cc)", 0.0, pitch_min, 15, False, 0.03, 0.45, 0.01, 0.35, 0.14, pitch_max)
                point_process = call(sound, "To PointProcess (periodic, cc)", pitch_min, pitch_max)

                for t in range(len(start_times)):   
                    start = float(start_times[t])
                    stop = float(stop_times[t])
                
                    # # F0
                    # f0_mean = call(pitch, "Get mean", start, stop, "Hertz")
                    # f0_median = call(pitch, "Get quantile", start, stop, 0.5, "Hertz")
                    # f0_std = call(pitch, "Get standard deviation", start, stop, "Hertz")
                    # f0_iqr = call(pitch, "Get quantile", start, stop, 0.75, "Hertz") - call(pitch, "Get quantile", start, stop, 0.25, "Hertz")
                    # f0_range = call(pitch, "Get maximum", start, stop, "Hertz", "parabolic") - call(pitch, "Get minimum", start, stop, "Hertz", "parabolic")
                    # f0_max = call(pitch, "Get maximum", start, stop, "Hertz", "parabolic")

                    # # Jitter
                    # jitter_local = call(point_process, "Get jitter (local)", start, stop, 0.0001, 1/pitch_min, 1.3)
                    # jitter_local_absolute = call(point_process, "Get jitter (local, absolute)", start, stop, 0.0001, 1/pitch_min, 1.3)
                    # jitter_rap = call(point_process, "Get jitter (rap)", start, stop, 0.0001, 1/pitch_min, 1.3)
                    # jitter_ppq5 = call(point_process, "Get jitter (ppq5)", start, stop, 0.0001, 1/pitch_min, 1.3)
                    # jitter_ddp = call(point_process, "Get jitter (ddp)", start, stop, 0.0001, 1/pitch_min, 1.3)

                    # # Shimmer
                    # shimmer_local = call([sound,point_process], "Get shimmer (local)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                    # shimmer_local_dB = call([sound,point_process], "Get shimmer (local_dB)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                    # shimmer_apq3 = call([sound,point_process], "Get shimmer (apq3)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                    # shimmer_apq5 = call([sound,point_process], "Get shimmer (apq5)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                    # shimmer_apq11 = call([sound,point_process], "Get shimmer (apq11)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                    # shimmer_dda = call([sound,point_process], "Get shimmer (dda)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)

                    # # HNR
                    # harmonicity = call(sound, "To Harmonicity (cc)", 0.01, pitch_min, 0.1, 4.5)
                    # hnr_mean = call(harmonicity, "Get mean", start, stop)
                    # hnr_std = call(harmonicity, "Get standard deviation", start, stop)
                    # hnr_min = call(harmonicity, "Get minimum", start, stop, 'parabolic')
                    # hnr_max = call(harmonicity, "Get maximum", start, stop, 'parabolic')

                    # # Formants
                    # formant = call(sound, "To Formant (burg)", 0.0, 5.0, formant_max, 0.025, 50)

                    # f1_mean = call(formant, "Get mean", 1, start, stop, "Hertz")
                    # f1_std = call(formant, "Get standard deviation", 1, start, stop, "Hertz")
                    # f1_median = call(formant, "Get quantile", 1, start, stop, "Hertz", 0.50)
                    # f1_iqr = call(formant, "Get quantile", 1, start, stop, "Hertz", 0.75) - call(formant, "Get quantile", 1, start, stop, "Hertz", 0.25)
                    # f1_range = call(formant, "Get maximum", 1, start, stop, "Hertz", "parabolic") - call(formant, "Get minimum", 1, start, stop, "Hertz", "parabolic")
                    # f1_max = call(formant, "Get maximum", 1, start, stop, "Hertz", "parabolic")

                    # f2_mean = call(formant, "Get mean", 2, start, stop, "Hertz")
                    # f2_std = call(formant, "Get standard deviation", 2, start, stop, "Hertz")
                    # f2_median = call(formant, "Get quantile", 2, start, stop, "Hertz", 0.50)
                    # f2_iqr = call(formant, "Get quantile", 2, start, stop, "Hertz", 0.75) - call(formant, "Get quantile", 2, start, stop, "Hertz", 0.25)
                    # f2_range = call(formant, "Get maximum", 2, start, stop, "Hertz", "parabolic") - call(formant, "Get minimum", 2, start, stop, "Hertz", "parabolic")
                    # f2_max = call(formant, "Get maximum", 2, start, stop, "Hertz", "parabolic")

                    # f3_mean = call(formant, "Get mean", 3, start, stop, "Hertz")
                    # f3_std = call(formant, "Get standard deviation", 3, start, stop, "Hertz")
                    # f3_median = call(formant, "Get quantile", 3, start, stop, "Hertz", 0.50)
                    # f3_iqr = call(formant, "Get quantile", 3, start, stop, "Hertz", 0.75) - call(formant, "Get quantile", 3, start, stop, "Hertz", 0.25)
                    # f3_range = call(formant, "Get maximum", 3, start, stop, "Hertz", "parabolic") - call(formant, "Get minimum", 3, start, stop, "Hertz", "parabolic")
                    # f3_max = call(formant, "Get maximum", 3, start, stop, "Hertz", "parabolic")

                    # CP
                    snd_syllable = sound.extract_part(start, stop)
                    # spectrum = call(snd_syllable, "To Spectrum")
                    # cepstrum = call(spectrum, "To PowerCepstrum")
                    # cpp = call(cepstrum, "Get peak prominence", pitch_min, pitch_max, "parabolic", 0.001, 0.05, "Exponential decay", "Robust slow")
                    # cp = call(cepstrum, "Get peak", pitch_min, pitch_max, "parabolic")

                    # # MFCC features
                    mfccs_sentence = call(snd_syllable, "To MFCC", 12, 0.015, 0.005, 100, 100, 0.0)
                    mfcc_matrix = mfccs_sentence.to_array()
                    mfcc_mean = np.mean(mfcc_matrix, axis=1)
                    mfcc_std = np.std(mfcc_matrix, axis=1)
                    mfcc_median = np.median(mfcc_matrix, axis=1)
                    mfcc_iqr = iqr(mfcc_matrix, axis=1)

                    duration = stop - start
                    rep = t+1

                    # Save all the results
                    features_rep = {
                        "rep": rep,
                        "duration": duration,
                        # "f0 mean": f0_mean,
                        # "f0 median": f0_median,
                        # "f0 std": f0_std,
                        # "f0 iqr": f0_iqr,
                        # "f0 range": f0_range,
                        # "f0 max": f0_max,
                        # "jitter local": jitter_local,
                        # "jitter local absolute": jitter_local_absolute,
                        # "jitter rap": jitter_rap,
                        # "jitter ppq5": jitter_ppq5,
                        # "jitter ddp": jitter_ddp,
                        # "shimmer local": shimmer_local,
                        # "shimmer local dB": shimmer_local_dB,
                        # "shimmer apq3": shimmer_apq3,
                        # "shimmer apq5": shimmer_apq5,
                        # "shimmer apq11": shimmer_apq11,
                        # "shimmer dda": shimmer_dda,
                        # "hnr mean": hnr_mean,
                        # "hnr std": hnr_std,
                        # "hnr min": hnr_min,
                        # "hnr max": hnr_max,
                        # "cp": cp,
                        # "cpp": cpp,
                        # "f1 mean": f1_mean,
                        # "f1 std": f1_std,
                        # "f1 median": f1_median,
                        # "f1 iqr": f1_iqr,
                        # "f1 range": f1_range,
                        # 'f1 max': f1_max,
                        # "f2 mean": f2_mean,
                        # "f2 std": f2_std,
                        # "f2 median": f2_median,
                        # "f2 iqr": f2_iqr,
                        # "f2 range": f2_range,
                        # "f2 max": f2_max,
                        # "f3 mean": f3_mean,
                        # "f3 std": f3_std,
                        # "f3 median": f3_median,
                        # "f3 iqr": f3_iqr,
                        # "f3 range": f3_range,
                        # "f3 max": f3_max,
                    }

                    for i in range(len(mfcc_mean)):
                        features_rep.update({
                            f"mfcc{i}_mean": mfcc_mean[i],
                            # f"mfcc{i}_std": mfcc_std[i],
                            # f"mfcc{i}_median": mfcc_median[i],
                            # f"mfcc{i}_iqr": mfcc_iqr[i],
                        })

                    features_sy.append(features_rep)
                
                speech_duration = stop_times[-1] - start_times[0]
                articulation_duration = np.sum([f['duration'] for f in features_sy])
                repetitions = features_sy[-1]['rep']
                speech_rate = repetitions / speech_duration if speech_duration > 0 else 0
                articulation_rate = repetitions / articulation_duration if articulation_duration > 0 else 0

                audio_array, _ = librosa.load(audio_file, sr=sr)
                audio_array = audio_array * 0

                for j in range(0, len(start_times)):
                    audio_array[int(start_times[j]*sr):int(stop_times[j]*sr)] = 1
                y_task = audio_array[int(start_times[0]*sr):int(stop_times[-1]*sr)]
                count = np.sum(y_task)
                y_task_padded = np.pad(y_task, (1, 1), constant_values=0)
                au = np.diff(y_task_padded)
                au_length = len(np.where(au == 1)[0])

                act_rate = count / len(y_task) if len(y_task) > 0 else 0
                act_freq = au_length / speech_duration if speech_duration > 0 else 0

                label = folder[-2:]

                # Mean the values in the dict
                features_s = {
                    "ID": subj[:5],
                    "label": label,
                    # "rep": features_sy[-1]['rep'],
                    # "speech duration": speech_duration,
                    # "articulation duration": articulation_duration,
                    # "speech rate": speech_rate,
                    # "articulation rate": articulation_rate,
                    # "act_rate": act_rate,
                    # "act_freq": act_freq,
                    # "articulation entropy": np.nan,
                    # "f0 mean": np.nanmean([f["f0 mean"] for f in features_sy]),
                    # "f0 std": np.nanmean([f["f0 std"] for f in features_sy]),
                    # "f0 median": np.nanmean([f["f0 median"] for f in features_sy]),
                    # "f0 iqr": np.nanmean([f["f0 iqr"] for f in features_sy]),
                    # "f0 range": np.nanmean([f["f0 range"] for f in features_sy]),
                    # "f0 max": np.nanmean([f["f0 max"] for f in features_sy]),
                    # "jitter local": np.nanmean([f["jitter local"] for f in features_sy]),
                    # "jitter local absolute": np.nanmean([f["jitter local absolute"] for f in features_sy]),
                    # "jitter rap": np.nanmean([f["jitter rap"] for f in features_sy]),
                    # "jitter ppq5": np.nanmean([f["jitter ppq5"] for f in features_sy]),
                    # "jitter ddp": np.nanmean([f["jitter ddp"] for f in features_sy]),
                    # "shimmer local": np.nanmean([f["shimmer local"] for f in features_sy]),
                    # "shimmer local dB": np.nanmean([f["shimmer local dB"] for f in features_sy]),
                    # "shimmer apq3": np.nanmean([f["shimmer apq3"] for f in features_sy]),
                    # "shimmer apq5": np.nanmean([f["shimmer apq5"] for f in features_sy]),
                    # "shimmer apq11": np.nanmean([f["shimmer apq11"] for f in features_sy]),
                    # "shimmer dda": np.nanmean([f["shimmer dda"] for f in features_sy]),
                    # "hnr mean": np.nanmean([f["hnr mean"] for f in features_sy]),
                    # "hnr std": np.nanmean([f["hnr std"] for f in features_sy]),
                    # "hnr min": np.nanmean([f["hnr min"] for f in features_sy]),
                    # "hnr max": np.nanmean([f["hnr max"] for f in features_sy]),
                    # "cp": np.nanmean([f["cp"] for f in features_sy]),
                    # "cpp": np.nanmean([f["cpp"] for f in features_sy]),
                    # "f1 mean": np.nanmean([f["f1 mean"] for f in features_sy]),
                    # "f1 std": np.nanmean([f["f1 std"] for f in features_sy]),
                    # "f1 median": np.nanmean([f["f1 median"] for f in features_sy]),
                    # "f1 iqr": np.nanmean([f["f1 iqr"] for f in features_sy]),
                    # "f1 range": np.nanmean([f["f1 range"] for f in features_sy]),
                    # "f1 max": np.nanmean([f["f1 max"] for f in features_sy]),
                    # "f2 mean": np.nanmean([f["f2 mean"] for f in features_sy]),
                    # "f2 std": np.nanmean([f["f2 std"] for f in features_sy]),
                    # "f2 median": np.nanmean([f["f2 median"] for f in features_sy]),
                    # "f2 iqr": np.nanmean([f["f2 iqr"] for f in features_sy]),
                    # "f2 range": np.nanmean([f["f2 range"] for f in features_sy]),
                    # "f2 max": np.nanmean([f["f2 max"] for f in features_sy]),
                    # "f3 mean": np.nanmean([f["f3 mean"] for f in features_sy]),
                    # "f3 std": np.nanmean([f["f3 std"] for f in features_sy]),
                    # "f3 median": np.nanmean([f["f3 median"] for f in features_sy]),
                    # "f3 iqr": np.nanmean([f["f3 iqr"] for f in features_sy]),
                    # "f3 range": np.nanmean([f["f3 range"] for f in features_sy]),
                    # "f3 max": np.nanmean([f["f3 max"] for f in features_sy])
                }

                for i in range(12):
                    features_s.update({
                        f"mfcc{i}_mean": np.nanmean([f[f"mfcc{i}_mean"] for f in features_sy]),
                        # f"mfcc{i}_std": np.nanmean([f[f"mfcc{i}_std"] for f in features_sy]),
                        # f"mfcc{i}_median": np.nanmean([f[f"mfcc{i}_median"] for f in features_sy]),
                        # f"mfcc{i}_iqr": np.nanmean([f[f"mfcc{i}_iqr"] for f in features_sy]),
                    })

                features_syll_tot.append(features_s)

                results_sy_df = pd.DataFrame(features_syll_tot)
                # results_sy_df.to_excel(os.path.join(general_path, "Features", "Features_syllables.xlsx"), index=False)        

            if folder.startswith('phonation'):
                timings = pd.read_excel(os.path.join(timings_path, folder, f"{folder}.xlsx"))

                idx = np.where(timings['ID'] == subj[:5])[0]
                start = timings['Start'].values[idx][0]
                stop = timings['Stop'].values[idx][0]

                sound = parselmouth.Sound(audio_file)
                # pitch = call(sound, "To Pitch (cc)", 0.0, pitch_min, 15, False, 0.03, 0.45, 0.01, 0.35, 0.14, pitch_max)
                # point_process = call(sound, "To PointProcess (periodic, cc)", pitch_min, pitch_max)

                # duration = stop - start
                #  # F0
                # f0_mean = call(pitch, "Get mean", start, stop, "Hertz")
                # f0_median = call(pitch, "Get quantile", start, stop, 0.5, "Hertz")
                # f0_std = call(pitch, "Get standard deviation", start, stop, "Hertz")
                # f0_iqr = call(pitch, "Get quantile", start, stop, 0.75, "Hertz") - call(pitch, "Get quantile", start, stop, 0.25, "Hertz")
                # f0_range = call(pitch, "Get maximum", start, stop, "Hertz", "parabolic") - call(pitch, "Get minimum", start, stop, "Hertz", "parabolic")
                # f0_max = call(pitch, "Get maximum", start, stop, "Hertz", "parabolic")

                # # Jitter
                # jitter_local = call(point_process, "Get jitter (local)", start, stop, 0.0001, 1/pitch_min, 1.3)
                # jitter_local_absolute = call(point_process, "Get jitter (local, absolute)", start, stop, 0.0001, 1/pitch_min, 1.3)
                # jitter_rap = call(point_process, "Get jitter (rap)", start, stop, 0.0001, 1/pitch_min, 1.3)
                # jitter_ppq5 = call(point_process, "Get jitter (ppq5)", start, stop, 0.0001, 1/pitch_min, 1.3)
                # jitter_ddp = call(point_process, "Get jitter (ddp)", start, stop, 0.0001, 1/pitch_min, 1.3)

                # # Shimmer
                # shimmer_local = call([sound,point_process], "Get shimmer (local)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                # shimmer_local_dB = call([sound,point_process], "Get shimmer (local_dB)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                # shimmer_apq3 = call([sound,point_process], "Get shimmer (apq3)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                # shimmer_apq5 = call([sound,point_process], "Get shimmer (apq5)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                # shimmer_apq11 = call([sound,point_process], "Get shimmer (apq11)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)
                # shimmer_dda = call([sound,point_process], "Get shimmer (dda)", start, stop, 0.0001, 1/pitch_min, 1.3, 1.6)

                # # HNR
                # harmonicity = call(sound, "To Harmonicity (cc)", 0.01, pitch_min, 0.1, 4.5)
                # hnr_mean = call(harmonicity, "Get mean", start, stop)   
                # hnr_std = call(harmonicity, "Get standard deviation", start, stop)
                # hnr_min = call(harmonicity, "Get minimum", start, stop, 'parabolic')
                # hnr_max = call(harmonicity, "Get maximum", start, stop, 'parabolic')

                # # Formants
                # formant = call(sound, "To Formant (burg)", 0.0, 5.0, formant_max, 0.025, 50)

                # f1_mean = call(formant, "Get mean", 1, start, stop, "Hertz")
                # f1_std = call(formant, "Get standard deviation", 1, start, stop, "Hertz")
                # f1_median = call(formant, "Get quantile", 1, start, stop, "Hertz", 0.50)
                # f1_iqr = call(formant, "Get quantile", 1, start, stop, "Hertz", 0.75) - call(formant, "Get quantile", 1, start, stop, "Hertz", 0.25)
                # f1_range = call(formant, "Get maximum", 1, start, stop, "Hertz", "parabolic") - call(formant, "Get minimum", 1, start, stop, "Hertz", "parabolic")
                # f1_max = call(formant, "Get maximum", 1, start, stop, "Hertz", "parabolic")

                # f2_mean = call(formant, "Get mean", 2, start, stop, "Hertz")
                # f2_std = call(formant, "Get standard deviation", 2, start, stop, "Hertz")
                # f2_median = call(formant, "Get quantile", 2, start, stop, "Hertz", 0.50)
                # f2_iqr = call(formant, "Get quantile", 2, start, stop, "Hertz", 0.75) - call(formant, "Get quantile", 2, start, stop, "Hertz", 0.25)
                # f2_range = call(formant, "Get maximum", 2, start, stop, "Hertz", "parabolic") - call(formant, "Get minimum", 2, start, stop, "Hertz", "parabolic")
                # f2_max = call(formant, "Get maximum", 2, start, stop, "Hertz", "parabolic")

                # f3_mean = call(formant, "Get mean", 3, start, stop, "Hertz")
                # f3_std = call(formant, "Get standard deviation", 3, start, stop, "Hertz")
                # f3_median = call(formant, "Get quantile", 3, start, stop, "Hertz", 0.50)
                # f3_iqr = call(formant, "Get quantile", 3, start, stop, "Hertz", 0.75) - call(formant, "Get quantile", 3, start, stop, "Hertz", 0.25)
                # f3_range = call(formant, "Get maximum", 3, start, stop, "Hertz", "parabolic") - call(formant, "Get minimum", 3, start, stop, "Hertz", "parabolic")
                # f3_max = call(formant, "Get maximum", 3, start, stop, "Hertz", "parabolic")

                # # CP
                snd_vowel = sound.extract_part(start, stop)
                # spectrum = call(snd_vowel, "To Spectrum")
                # cepstrum = call(spectrum, "To PowerCepstrum")
                # cpp = call(cepstrum, "Get peak prominence", pitch_min, pitch_max, "parabolic", 0.001, 0.05, "Exponential decay", "Robust slow")
                # cp = call(cepstrum, "Get peak", pitch_min, pitch_max, "parabolic")

                # MFCC features
                mfccs_sentence = call(snd_vowel, "To MFCC", 12, 0.015, 0.005, 100, 100, 0.0)
                mfcc_matrix = mfccs_sentence.to_array()
                mfcc_mean = np.mean(mfcc_matrix, axis=1)
                mfcc_std = np.std(mfcc_matrix, axis=1)
                mfcc_median = np.median(mfcc_matrix, axis=1)
                mfcc_iqr = iqr(mfcc_matrix, axis=1)            

                # Save all the results
                # features_vow_tot.append({
                    # "ID": subj[:5],
                    # "label": folder[-1],
                    # "duration": duration,
                    # "f0 mean": f0_mean,
                    # "f0 median": f0_median,
                    # "f0 std": f0_std,
                    # "f0 iqr": f0_iqr,
                    # "f0 range": f0_range,
                    # "f0 max": f0_max,
                    # "jitter local": jitter_local,
                    # "jitter local absolute": jitter_local_absolute,
                    # "jitter rap": jitter_rap,
                    # "jitter ppq5": jitter_ppq5,
                    # "jitter ddp": jitter_ddp,
                    # "shimmer local": shimmer_local,
                    # "shimmer local dB": shimmer_local_dB,
                    # "shimmer apq3": shimmer_apq3,
                    # "shimmer apq5": shimmer_apq5,
                    # "shimmer apq11": shimmer_apq11,
                    # "shimmer dda": shimmer_dda,
                    # "hnr mean": hnr_mean,
                    # "hnr std": hnr_std,
                    # "hnr min": hnr_min,
                    # "hnr max": hnr_max,
                    # "cp": cp,
                    # "cpp": cpp,
                    # "f1 mean": f1_mean,
                    # "f1 std": f1_std,
                    # "f1 median": f1_median,
                    # "f1 iqr": f1_iqr,
                    # "f1 range": f1_range,
                    # "f1 max": f1_max,
                    # "f2 mean": f2_mean,
                    # "f2 std": f2_std,
                    # "f2 median": f2_median,
                    # "f2 iqr": f2_iqr,
                    # "f2 range": f2_range,
                    # "f2 max": f2_max,
                    # "f3 mean": f3_mean,
                    # "f3 std": f3_std,
                    # "f3 median": f3_median,
                    # "f3 iqr": f3_iqr,
                    # "f3 range": f3_range,
                    # "f3 max": f3_max
                # })

                # Non posso usare update perchè ci sono più vocali per soggetto
                features_v = {
                    "ID": subj[:5],
                    "label": folder[-1],
                    # "duration": duration,
                }
                for i in range(len(mfcc_mean)):
                    features_v.update({
                        f"mfcc{i}_mean": mfcc_mean[i],
                        # f"mfcc{i}_std": mfcc_std[i],
                        # f"mfcc{i}_median": mfcc_median[i],
                        # f"mfcc{i}_iqr": mfcc_iqr[i],
                    })
                
                features_vow_tot.append(features_v)

                results_vow_df = pd.DataFrame(features_vow_tot)
                # results_vow_df.to_excel(os.path.join(general_path, "Features", "Features_vowels.xlsx"), index=False)


            completed_iterations += 1
            elapsed_time = time.time() - iter_start
            avg_iteration_time = elapsed_time / completed_iterations
            remaining_time = (total_iterations - completed_iterations) * avg_iteration_time
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"{(completed_iterations/total_iterations)*100:.2f}%")

# # Save results

def add_suffix(row):
    suffix = "_" + row["label"]
    renamed = row.drop(["ID", "label"]).add_suffix(suffix)
    renamed["ID"] = row["ID"]
    return renamed


syllabels = results_sy_df.apply(add_suffix, axis=1)
syllabels = syllabels.groupby("ID", as_index=False).first()
syllabels.to_excel(os.path.join(general_path, "Features", "Features_syllables_mfcc.xlsx"), index=False) 

vowels = results_vow_df.apply(add_suffix, axis=1)
vowels = vowels.groupby("ID", as_index=False).first()
vowels.to_excel(os.path.join(general_path, "Features", "Features_vowels_mfcc.xlsx"), index=False)


progress_bar.close()
