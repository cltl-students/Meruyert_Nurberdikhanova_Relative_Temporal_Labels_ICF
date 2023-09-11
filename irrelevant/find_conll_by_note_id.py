import pandas as pd 
from pathlib import Path
# check from_inception_tsv file format
pd.set_option('display.max_columns', None)


for dfpath in Path('/mnt/data/A-Proof/data2/a-proof-zonmw/data/from_inception_tsv/').glob('annotated_df_week_*.pkl'):

week_14_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/from_inception_tsv/annotated_df_week_14.pkl'
week_14 = pd.read_pickle(week_14_path)
print(week_14)