import spacy, nl_core_news_sm
import pandas as pd 
pd.set_option('display.max_columns', None)
nlp = nl_core_news_sm.load()

def get_full_notes(rawpath,df_conll):
    ''''''
    df_raw = pd.read_pickle(rawpath)
    intersection = df_conll.merge(df_raw,on='NotitieID')
    intersection.drop(['institution_y','MDN_y','Typenotitie'],axis=1, inplace=True)
    intersection.rename(columns={'institution_x':'institution','MDN_x':'MDN'},inplace=True)
    intersection.drop_duplicates(inplace=True)
    print(f'{len(intersection)} rows in intersection, {len(intersection.pad_sen_id.unique())} unique in intersection')
    return intersection

def save_full_notes(df_conll,data_type):
    raw_2017='/mnt/data/A-Proof/data2/a-proof-zonmw/data/2017_raw/processed.pkl'
    raw_2018='/mnt/data/A-Proof/data2/a-proof-zonmw/data/2018_raw/processed.pkl'
    raw_2020='/mnt/data/A-Proof/data2/a-proof-zonmw/data/2020_raw/processed.pkl'
    intersection2017 = get_full_notes(raw_2017,df_conll)
    intersection2018 = get_full_notes(raw_2018,df_conll)
    intersection2020 = get_full_notes(raw_2020,df_conll)
    df = pd.concat([intersection2017,intersection2018,intersection2020])
    df.drop_duplicates(inplace=True)
    
    df = df.sample(frac=1)
    print(f'{len(df)} all sentences, {len(df.pad_sen_id.unique())} unique sentences')
    df.to_csv(f'../data/{data_type}_full_notes.tsv',sep='\t',index=False)

def get_note_length_from_raw(rawpath, df_conll, pipeline, year):
    '''
    Obtain note lengths via note IDs for your processed data from the raw data.

    :rawpath: str filepath to the raw file
    :df_conll: pandas dataframe of your data
    :pipeline: spacy dutch language pipeline to process the note text
    :year: str indicating year of raw notes file

    :return df_conll: modified dataframe with added note_lengths
    '''
    # raw_2017='/mnt/data/A-Proof/data2/a-proof-zonmw/data/2017_raw/processed.pkl'
    # raw_2018='/mnt/data/A-Proof/data2/a-proof-zonmw/data/2018_raw/processed.pkl'
    # raw_2020='/mnt/data/A-Proof/data2/a-proof-zonmw/data/2020_raw/processed.pkl'
    df = df_conll.loc[df_conll['year']== year]
    notes = df['note_id'].unique().tolist()
    note_texts, note_ids= get_relevant_raw_notes(rawpath,notes)
    note_lengths = []
    for text in note_texts:
        doc = pipeline(text)
        note_len = str(len(list(doc.sents)))
        note_lengths.append(note_len)
    note_dict = dict(zip(notes, note_lengths))
    df_conll['note_len'] = df_conll['note_id'].map(note_dict)
    return df_conll


# df_raw_2017 = pd.read_pickle('/mnt/data/A-Proof/data2/a-proof-zonmw/data/2017_raw/processed.pkl')
# intersection2017 = df_raw_2017.merge(train, on='NotitieID')
# # print(intersection2017)
# print(len(intersection2017['NotitieID'].unique()))
# df_raw_2018 = pd.read_pickle('/mnt/data/A-Proof/data2/a-proof-zonmw/data/2018_raw/processed.pkl')
# intersection2018 = df_raw_2018.merge(train, on='NotitieID')
# print(intersection2018)
# print(len(intersection2018['NotitieID'].unique()))
# df_raw_2020 = pd.read_pickle('/mnt/data/A-Proof/data2/a-proof-zonmw/data/2020_raw/processed.pkl')
# intersection2020 = train.merge(df_raw_2020, on='NotitieID')
# print(intersection2020)
# print(len(intersection2020['NotitieID'].unique()))

def main():
    train = pd.read_csv('../data/train.tsv', sep='\t',dtype='string')
    save_full_notes(train,'train')
    dev = pd.read_csv('../data/dev.tsv',sep='\t',dtype='string')
    save_full_notes(dev,'dev')

if __name__=='__main__':
    main()