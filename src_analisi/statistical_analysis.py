import os
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import shapiro, iqr, ttest_ind, mannwhitneyu, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn

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
                    groups = np.concatenate([
                        np.repeat(0, len(values_0)),
                        np.repeat(1, len(values_1)),
                        np.repeat(2, len(values_2)),
                        np.repeat(3, len(values_3)),
                        np.repeat(4, len(values_4))
                    ])
                    data = np.concatenate([values_0, values_1, values_2, values_3, values_4])
                    df_data = pd.DataFrame({'value': data, 'group': groups})
                    posthoc = posthoc_dunn(df_data, val_col='value', group_col='group', p_adjust='bonferroni')
                    results["p-value 0 vs. 1"].append(posthoc.loc[0, 1])
                    results["p-value 0 vs. 2"].append(posthoc.loc[0, 2])
                    results["p-value 0 vs. 3"].append(posthoc.loc[0, 3])
                    results["p-value 0 vs. 4"].append(posthoc.loc[0, 4])
                    results["p-value 1 vs. 2"].append(posthoc.loc[1, 2])
                    results["p-value 1 vs. 3"].append(posthoc.loc[1, 3])
                    results["p-value 1 vs. 4"].append(posthoc.loc[1, 4])
                    results["p-value 2 vs. 3"].append(posthoc.loc[2, 3])
                    results["p-value 2 vs. 4"].append(posthoc.loc[2, 4])
                    results["p-value 3 vs. 4"].append(posthoc.loc[3, 4])

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
features_path = os.path.join(general_path, 'Features/New')
results_path = os.path.join(general_path, 'Results/Second test')

# Load the cleaned dataframes
df_complete = pd.read_excel(os.path.join(features_path, 'complete_clean.xlsx'))

columns_to_drop_complete = ['subjid', 'category', 'sex', 'ALSFRS-R_SpeechSubscore', 'ALSFRS-R_SwallowingSubscore', 'PUMNS_BulbarSubscore']

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
results_df.to_excel(os.path.join(results_path, "statistical_analysis.xlsx"), index=False)
