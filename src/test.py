import os
import pandas as pd

general_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(general_path, 'Features')


syllables_df = pd.read_excel(os.path.join(features_path, 'syllabels_complete.xlsx'))
# Calcolo la somma delle colonne repetition che hanno lo stesso ID e stessa tipologia di task
reptot = syllables_df.groupby(['subjid', 'task'])['repetition'].max().reset_index()
articulationtot = syllables_df.groupby(['subjid', 'task'])['duration'].sum().reset_index()

reptot.to_excel(os.path.join(features_path, 'repetition_totals.xlsx'), index=False)
articulationtot.to_excel(os.path.join(features_path, 'articulation_totals.xlsx'), index=False)

