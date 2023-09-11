import pandas as pd


def get_rel_time_subset(dfpath, data_type):
    df = pd.read_pickle(dfpath)
    df['labels'].astype('string')
    df_background = df.loc[df['labels'] == '[0, 0, 0, 0, 0, 0, 0, 0, 0]']
    df_background = df_background.loc[df_background['background_sent']== True]
    df_background.to_csv(f'../data/{data_type}_background.tsv',sep='\t',index=False)
    df_target = df.loc[df['labels'] == '[0, 0, 0, 0, 0, 0, 0, 0, 0]']
    df_target = df_target.loc[df_target['target_sent']== True]
    df_target.to_csv(f'../data/{data_type}_target.tsv',sep='\t',index=False)
    df_current = df.loc[df['labels'] != '[0, 0, 0, 0, 0, 0, 0, 0, 0]']
    df_current.to_csv(f'../data/{data_type}_current.tsv',sep='\t',index=False)
    print(f'{data_type} pickled data has {len(df_background)} background sentences,\
                                         {len(df_target)} target sentences,\
                                         {len(df_current)} current sentences.')

