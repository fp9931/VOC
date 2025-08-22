import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro, iqr, ttest_ind, mannwhitneyu, pearsonr, spearmanr, f_oneway, kruskal
import scipy.stats as ss
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def statistical_analysis (df, y, score):

    global results

    count_normal = (y == 4).sum()
    count_impaired = (y < 4).sum()
    proportion_impaired = count_impaired / (count_normal + count_impaired) if (count_normal + count_impaired) > 0 else 0
    print(f"Score {score}: {count_normal} normal, {count_impaired} impaired, proportion impaired {proportion_impaired:.2f}")

    if score != 'Bulbar':
        y = np.where(y == 4, 0, 1) # 0 normal, 1 impaired
        id_0 = df[y == 0].index
        id_1 = df[y == 1].index
        
        if score == 'Speech':
            for col in df.columns:
                results["Feature"].append(col)

        for col in df.columns:
            values_Normal = df.iloc[id_0][col]
            values_Impaired = df.iloc[id_1][col]
            values_Normal = values_Normal[~np.isnan(values_Normal)]
            values_Impaired = values_Impaired[~np.isnan(values_Impaired)]
            _, p_value_Normal = shapiro(values_Normal)
            is_normal_Normal = p_value_Normal > 0.05
            _, p_value_Impaired = shapiro(values_Impaired)
            is_normal_Impaired = p_value_Impaired > 0.05

            is_normal = is_normal_Normal and is_normal_Impaired

            if is_normal:
                _, p_value = ttest_ind(values_Normal, values_Impaired, equal_var=False)
            else:
                _, p_value = mannwhitneyu(values_Normal, values_Impaired, alternative='two-sided')

            results[f"Normality {score} Normal"].append(True if is_normal_Normal else False)
            results[f"Mean {score} Normal"].append(np.mean(values_Normal))
            results[f"Std {score} Normal"].append(np.std(values_Normal))
            results[f"Median {score} Normal"].append(np.median(values_Normal))
            results[f"IQR {score} Normal"].append(iqr(values_Normal))
            results[f"Normality {score} Impaired"].append(True if is_normal_Impaired else False)
            results[f"Mean {score} Impaired"].append(np.mean(values_Impaired))
            results[f"Std {score} Impaired"].append(np.std(values_Impaired))
            results[f"Median {score} Impaired"].append(np.median(values_Impaired))
            results[f"IQR {score} Impaired"].append(iqr(values_Impaired))
            results[f"p-value {score}"].append(p_value)


    else:
        id_0 = df[y == 0].index
        id_1 = df[y == 1].index
        id_2 = df[y == 2].index
        id_3 = df[y == 3].index
        id_4 = df[y == 4].index

        for col in df.columns:

            values_0 = df.iloc[id_0][col]
            values_1 = df.iloc[id_1][col]
            values_2 = df.iloc[id_2][col]
            values_3 = df.iloc[id_3][col]
            values_4 = df.iloc[id_4][col]

            values_0 = values_0[~np.isnan(values_0)]
            values_1 = values_1[~np.isnan(values_1)]
            values_2 = values_2[~np.isnan(values_2)]
            values_3 = values_3[~np.isnan(values_3)]
            values_4 = values_4[~np.isnan(values_4)]

            _, p_value_0 = shapiro(values_0)
            is_normal_0 = p_value_0 > 0.05
            _, p_value_1 = shapiro(values_1)
            is_normal_1 = p_value_1 > 0.05
            _, p_value_2 = shapiro(values_2)
            is_normal_2 = p_value_2 > 0.05
            _, p_value_3 = shapiro(values_3)
            is_normal_3 = p_value_3 > 0.05
            _, p_value_4 = shapiro(values_4)
            is_normal_4 = p_value_4 > 0.05

            results[f"Normality {score} 0"].append(True if is_normal_0 else False)
            results[f"Mean {score} 0"].append(np.mean(values_0))
            results[f"Std {score} 0"].append(np.std(values_0))
            results[f"Median {score} 0"].append(np.median(values_0))
            results[f"IQR {score} 0"].append(iqr(values_0))
            results[f"Normality {score} 1"].append(True if is_normal_1 else False)
            results[f"Mean {score} 1"].append(np.mean(values_1))
            results[f"Std {score} 1"].append(np.std(values_1))
            results[f"Median {score} 1"].append(np.median(values_1))
            results[f"IQR {score} 1"].append(iqr(values_1))
            results[f"Normality {score} 2"].append(True if is_normal_2 else False)
            results[f"Mean {score} 2"].append(np.mean(values_2))
            results[f"Std {score} 2"].append(np.std(values_2))
            results[f"Median {score} 2"].append(np.median(values_2))
            results[f"IQR {score} 2"].append(iqr(values_2))
            results[f"Normality {score} 3"].append(True if is_normal_3 else False)
            results[f"Mean {score} 3"].append(np.mean(values_3))
            results[f"Std {score} 3"].append(np.std(values_3))
            results[f"Median {score} 3"].append(np.median(values_3))
            results[f"IQR {score} 3"].append(iqr(values_3))
            results[f"Normality {score} 4"].append(True if is_normal_4 else False)
            results[f"Mean {score} 4"].append(np.mean(values_4))
            results[f"Std {score} 4"].append(np.std(values_4))
            results[f"Median {score} 4"].append(np.median(values_4))
            results[f"IQR {score} 4"].append(iqr(values_4))


            # # ANOVA or non-parametrical over scores
            if all([is_normal_0, is_normal_1, is_normal_2, is_normal_3, is_normal_4]):
                f_stat, p_val = f_oneway(values_0, values_1, values_2, values_3, values_4)    
            else:
                f_stat, p_val = kruskal(values_0, values_1, values_2, values_3, values_4)

            results["F-stat"].append(f_stat)
            results["p-value"].append(p_val)
                
            # # Post-hoc analysis (if needed)
            if p_val < 0.05: 
                if all([is_normal_0, is_normal_1, is_normal_2, is_normal_3, is_normal_4]):
                    posthoc = pairwise_tukeyhsd(np.concatenate([values_0, values_1, values_2, values_3, values_4]),
                                                 np.concatenate([np.repeat(0, len(values_0)),
                                                                 np.repeat(1, len(values_1)),
                                                                 np.repeat(2, len(values_2)),
                                                                 np.repeat(3, len(values_3)),
                                                                 np.repeat(4, len(values_4))]))
                    results["p-value 0 vs. 1"].append(posthoc.pvalues[0])
                    results["p-value 0 vs. 2"].append(posthoc.pvalues[1])
                    results["p-value 0 vs. 3"].append(posthoc.pvalues[2])
                    results["p-value 0 vs. 4"].append(posthoc.pvalues[3])
                    results["p-value 1 vs. 2"].append(posthoc.pvalues[4])
                    results["p-value 1 vs. 3"].append(posthoc.pvalues[5])
                    results["p-value 1 vs. 4"].append(posthoc.pvalues[6])
                    results["p-value 2 vs. 3"].append(posthoc.pvalues[7])
                    results["p-value 2 vs. 4"].append(posthoc.pvalues[8])
                    results["p-value 3 vs. 4"].append(posthoc.pvalues[9])
                else:
                    # posthoc = sp.posthoc_dunn(np.concatenate([values_0, values_1, values_2, values_3, values_4]),
                    #                              np.concatenate([np.repeat(0, len(values_0)),
                    #                                              np.repeat(1, len(values_1)),
                    #                                              np.repeat(2, len(values_2)),
                    #                                              np.repeat(3, len(values_3)),
                    #                                              np.repeat(4, len(values_4))]))
                    results["p-value 0 vs. 1"].append(posthoc.pvalues[0])
                    results["p-value 0 vs. 2"].append(posthoc.pvalues[1])
                    results["p-value 0 vs. 3"].append(posthoc.pvalues[2])
                    results["p-value 0 vs. 4"].append(posthoc.pvalues[3])
                    results["p-value 1 vs. 2"].append(posthoc.pvalues[4])
                    results["p-value 1 vs. 3"].append(posthoc.pvalues[5])
                    results["p-value 1 vs. 4"].append(posthoc.pvalues[6])
                    results["p-value 2 vs. 3"].append(posthoc.pvalues[7])
                    results["p-value 2 vs. 4"].append(posthoc.pvalues[8])
                    results["p-value 3 vs. 4"].append(posthoc.pvalues[9])

            else:
                results["p-value 0 vs. 1"].append(None)
                results["p-value 0 vs. 2"].append(None)
                results["p-value 0 vs. 3"].append(None)
                results["p-value 0 vs. 4"].append(None)
                results["p-value 1 vs. 2"].append(None)
                results["p-value 1 vs. 3"].append(None)
                results["p-value 1 vs. 4"].append(None)
                results["p-value 2 vs. 3"].append(None)
                results["p-value 2 vs. 4"].append(None)
                results["p-value 3 vs. 4"].append(None)

general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(general_path, 'Features')
results_path = os.path.join(general_path, 'Results')

# Load the cleaned dataframes
df_complete = pd.read_excel(os.path.join(features_path, 'complete_clean.xlsx'))

columns_to_drop_complete = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore', 
                    'SML11_t', 'SML12_t', 'SML13_t', 'SML21_t', 'SML22_t', 'SML23_t', 'SML31_t', 'SML32_t', 'SML33_t', 'SML41_t', 'SML42_t', 'SML43_t', 'x2D_DCT1_t', 'x2D_DCT2_t', 'x2D_DCT3_t', 'x2D_DCT4_t', 'x2D_DCT5_t', 'x2D_DCT6_t', 'x2D_DCT7_t', 'x2D_DCT8_t', 'x2D_DCT9_t',
                    'SML11_k', 'SML12_k', 'SML13_k', 'SML21_k', 'SML22_k', 'SML23_k', 'SML31_k', 'SML32_k', 'SML33_k', 'SML41_k', 'SML42_k', 'SML43_k', 'x2D_DCT1_k', 'x2D_DCT2_k', 'x2D_DCT3_k', 'x2D_DCT4_k', 'x2D_DCT5_k', 'x2D_DCT6_k', 'x2D_DCT7_k', 'x2D_DCT8_k', 'x2D_DCT9_k',
                    'SML11_p', 'SML12_p', 'SML13_p', 'SML21_p', 'SML22_p', 'SML23_p', 'SML31_p', 'SML32_p', 'SML33_p', 'SML41_p', 'SML42_p', 'SML43_p', 'x2D_DCT1_p', 'x2D_DCT2_p', 'x2D_DCT3_p', 'x2D_DCT4_p', 'x2D_DCT5_p', 'x2D_DCT6_p', 'x2D_DCT7_p', 'x2D_DCT8_p', 'x2D_DCT9_p', 
                    # 'mfcc_0_t', 'mfcc_1_t', 'mfcc_2_t', 'mfcc_3_t', 'mfcc_4_t', 'mfcc_5_t', 'mfcc_6_t', 'mfcc_7_t', 'mfcc_8_t', 'mfcc_9_t', 'mfcc_10_t', 'mfcc_11_t',
                    # 'mfcc_0_k', 'mfcc_1_k', 'mfcc_2_k', 'mfcc_3_k', 'mfcc_4_k', 'mfcc_5_k', 'mfcc_6_k', 'mfcc_7_k', 'mfcc_8_k', 'mfcc_9_k', 'mfcc_10_k', 'mfcc_11_k',
                    # 'mfcc_0_p', 'mfcc_1_p', 'mfcc_2_p', 'mfcc_3_p', 'mfcc_4_p', 'mfcc_5_p', 'mfcc_6_p', 'mfcc_7_p', 'mfcc_8_p', 'mfcc_9_p', 'mfcc_10_p', 'mfcc_11_p'
                    ]

# Filter ALS patients and drop unnecessary columns
als_df_complete = df_complete[df_complete['category'] == 'ALS'].reset_index(drop=True)

y_speech = als_df_complete['ALSFRS-R_SpeechSubscore'].values
y_swallowing = als_df_complete['ALSFRS-R_SwallowingSubscore'].values
y_bulbar = als_df_complete['PUMNS_BulbarSubscore'].values

als_df_complete = als_df_complete.drop(columns=columns_to_drop_complete)

results = {
    "Feature": [],

    "Normality Speech Normal": [],
    "Mean Speech Normal": [],
    "Std Speech Normal": [],
    "Median Speech Normal": [],
    "IQR Speech Normal": [],

    "Normality Speech Impaired": [],
    "Mean Speech Impaired": [],
    "Std Speech Impaired": [],
    "Median Speech Impaired": [],
    "IQR Speech Impaired": [],

    "p-value Speech": [],

    "Normality Swallowing Normal": [],
    "Mean Swallowing Normal": [],
    "Std Swallowing Normal": [],
    "Median Swallowing Normal": [],
    "IQR Swallowing Normal": [],

    "Normality Swallowing Impaired": [],
    "Mean Swallowing Impaired": [],
    "Std Swallowing Impaired": [],
    "Median Swallowing Impaired": [],
    "IQR Swallowing Impaired": [],

    "p-value Swallowing": [],

    "Normality Bulbar 0": [],
    "Mean Bulbar 0": [],
    "Std Bulbar 0": [],
    "Median Bulbar 0": [],
    "IQR Bulbar 0": [],

    "Normality Bulbar 1": [],
    "Mean Bulbar 1": [],
    "Std Bulbar 1": [],
    "Median Bulbar 1": [],
    "IQR Bulbar 1": [],

    "Normality Bulbar 2": [],
    "Mean Bulbar 2": [],
    "Std Bulbar 2": [],
    "Median Bulbar 2": [],
    "IQR Bulbar 2": [],

    "Normality Bulbar 3": [],
    "Mean Bulbar 3": [],
    "Std Bulbar 3": [],
    "Median Bulbar 3": [],
    "IQR Bulbar 3": [],

    "Normality Bulbar 4": [],
    "Mean Bulbar 4": [],
    "Std Bulbar 4": [],
    "Median Bulbar 4": [],
    "IQR Bulbar 4": [],

    "F-stat": [],
    "p-value": [],

    "p-value 0 vs. 1": [],
    "p-value 0 vs. 2": [],
    "p-value 0 vs. 3": [],
    "p-value 0 vs. 4": [],
    "p-value 1 vs. 2": [],
    "p-value 1 vs. 3": [],
    "p-value 1 vs. 4": [],
    "p-value 2 vs. 3": [],
    "p-value 2 vs. 4": [],
    "p-value 3 vs. 4": []
}

statistical_analysis(als_df_complete, y_speech, 'Speech')
statistical_analysis(als_df_complete, y_swallowing, 'Swallowing')
statistical_analysis(als_df_complete, y_bulbar, 'Bulbar')

# Save
results_df = pd.DataFrame(results)
results_df.to_csv("als_statistical_analysis_results.csv", index=False)
