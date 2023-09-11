import pandas as pd 
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
def get_duplicate_rows(df):
    # keep false to see how many times it happens
    df.sort_values(by='pad_sen_id',inplace=True)
    df_dupes = df[df.duplicated(subset=['pad_sen_id'],keep=False)]
    return df_dupes

def remove_older_duplicates(df):
    df['Notitiedatum'] = pd.to_datetime(df['Notitiedatum'],format='%Y/%m/%d')
    # sometimes duplicates have the same date => need to remove the shorter note or check if the sentence is in the note text
    df['index'] = df.index.values
    df_copy = df.copy()
    print(f'length of df before: {len(df_copy)}')
    pad_sen_id = df.groupby(by=['pad_sen_id','Notitiedatum'])
    group_pad = pad_sen_id.size()
    for i, group in group_pad.items():
        if group != 1: # select the ones with duplicates
            grouped = pad_sen_id.get_group(i) 
            # compare lengths of note
            grouped['len_note'] = grouped['all_text'].str.len()
            grouped['comparison'] = grouped.len_note.shift() # the first one will have NaN value
            indices = grouped['index'].to_list()
            drop_index_0 = grouped['len_note'][indices[1]] - grouped['comparison'][indices[1]]
            if drop_index_0 > 0:
                df_copy.drop(index=indices[0],inplace=True)
            else:
                df_copy.drop(index=indices[1],inplace=True)
    print(f'length of df after {len(df_copy)}')
    df_copy.sort_values(by='Notitiedatum',inplace=True)
    df_copy.drop_duplicates(subset=['pad_sen_id'],inplace=True,keep='last')
    df_copy.drop(columns=['index'],inplace=True)
    print(f'Length of df after keeping the latest {len(df_copy)}')
    return df_copy

def main():
    trainpath  = '../data/train_full_notes.tsv'
    train = pd.read_csv(trainpath, sep='\t')
    
    devpath = '../data/dev_full_notes.tsv'
    dev = pd.read_csv(devpath,sep='\t')

    train_no_dupes = remove_older_duplicates(train)
    dev_no_dupes = remove_older_duplicates(dev)
    train_no_dupes.to_csv('../data/train_full_notes_no_dupes.tsv',sep='\t',index=False)
    dev_no_dupes.to_csv('../data/dev_full_notes_no_dupes.tsv',sep='\t',index=False)

if __name__=='__main__':
    main()