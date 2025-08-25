import os
import pandas as pd

general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(general_path, 'Features/New')


syllables_df = pd.read_excel(os.path.join(features_path, 'syllabels_clean.xlsx'))

syllables_df = syllables_df.drop(columns=['category', 'sex'])

syllables_df_k = syllables_df[syllables_df['task'] == 'K']
syllables_df_t = syllables_df[syllables_df['task'] == 'T']
syllables_df_p = syllables_df[syllables_df['task'] == 'P']

syllables_df_k = syllables_df_k.drop(columns=['task'])
syllables_df_t = syllables_df_t.drop(columns=['task'])
syllables_df_p = syllables_df_p.drop(columns=['task'])

# Gorup by name and calculate mean for each feature (except 'name' and 'category' and 'sex')
mean_k = syllables_df_k.groupby('name').mean().add_suffix('_k').reset_index()
mean_t = syllables_df_t.groupby('name').mean().add_suffix('_t').reset_index()
mean_p = syllables_df_p.groupby('name').mean().add_suffix('_p').reset_index()

# Save the mean dataframes to new Excel files
# mean_k.to_excel(os.path.join(features_path, 'syllables_k_mean.xlsx'), index=False)
# mean_t.to_excel(os.path.join(features_path, 'syllables_t_mean.xlsx'), index=False)
# mean_p.to_excel(os.path.join(features_path, 'syllables_p_mean.xlsx'), index=False)

# Copute the sum of the feature column 'duration' for each task
sum_k = syllables_df_k.groupby('name')['duration'].sum().reset_index().rename(columns={'duration': 'tot_articulation_k'})
sum_t = syllables_df_t.groupby('name')['duration'].sum().reset_index().rename(columns={'duration': 'tot_articulation_t'})
sum_p = syllables_df_p.groupby('name')['duration'].sum().reset_index().rename(columns={'duration': 'tot_articulation_p'})

# Copute the total number of repetition of the syllable for each task
count_k = syllables_df_k.groupby('name').size().reset_index(name='number_of_repetitions_k')
count_t = syllables_df_t.groupby('name').size().reset_index(name='number_of_repetitions_t')
count_p = syllables_df_p.groupby('name').size().reset_index(name='number_of_repetitions_p')

# Merge the sum and count dataframes with the mean dataframes
mean_k = mean_k.merge(sum_k, on='name').merge(count_k, on='name')
mean_t = mean_t.merge(sum_t, on='name').merge(count_t, on='name')
mean_p = mean_p.merge(sum_p, on='name').merge(count_p, on='name')

# # Save the updated mean dataframes to new Excel files
mean_k.to_excel(os.path.join(features_path, 'syllables_k_mean.xlsx'), index=False)
mean_t.to_excel(os.path.join(features_path, 'syllables_t_mean.xlsx'), index=False)
mean_p.to_excel(os.path.join(features_path, 'syllables_p_mean.xlsx'), index=False)