import pandas as pd 
import numpy as np  
import spacy, nl_core_news_sm
pd.set_option('display.max_columns', None)

def get_note_length_from_raw(rawpath, df_conll, pipeline, year):
    '''
    Obtain note lengths via note IDs for your processed data from the raw data.

    :rawpath: str filepath to the raw file
    :df_conll: pandas dataframe of your data
    :pipeline: spacy dutch language pipeline to process the note text
    :year: str indicating year of raw notes file

    :return df_conll: modified dataframe with added note_lengths
    '''
    df = df_conll.loc[df_conll['year']== year]
    notes = df['note_id'].astype('string').unique().tolist()
    df_raw = pd.read_pickle(rawpath)
    intersection = df_raw.merge(df_conll, left_on = 'NotitieID',right_on='note_id')
    print(intersection)
    # note_texts, note_ids = get_relevant_raw_notes(rawpath,notes,year)
    note_lengths = []
    print(f"{year}: needed {len(notes)} notes, got {len(intersection)}")
    # notes_set = set(notes)
    # notes_of_year = [note for note in note_ids if note in notes_set]

    # for text in note_texts:
    #     doc = pipeline(text)
    #     note_len = str(len(list(doc.sents)))
    #     note_lengths.append(note_len)
    # note_dict = dict(zip(note_ids, note_lengths))
    # df_conll['note_len'] = df_conll['note_id'].map(note_dict)
    # return df_conll



def get_features(path,pipeline):
    df = pd.read_csv(path,sep='\t',dtype='string',header=0, encoding='utf-8', quoting = 3)
    # sentence number
    pad_list = df['pad_sent_id'].to_list()
    sent_ids = []
    for pad in df['pad_sent_id']:
        sent_id = pad.split('_')[1]
        sent_id = sent_id.lstrip('0')
        sent_ids.append(sent_id)
    df['sent_id'] = sent_ids
    # sentence position in a note
    # as a string
    sent_ids = df['sent_id'].to_list()
    note_len = df['note_len'].to_list()
    df['rel_position'] = [f'{sent}_{note}' for sent, note in zip(sent_ids,note_len)]
    # as a ratio => quartiles
    # quartiles = []
    # for sent, note in zip(sent_ids,note_len):
    #     sent = int(sent)
    #     note = int(note)
    #     rat = sent/note
    #     if rat < .25:
    #         quartile = 'Q1'
    #     elif .25 <=rat >= .5:
    #         quartile = 'Q2'
    #     elif .5 <= rat >= .75:
    #         quartile = 'Q3'
    #     else:
    #         quartile = 'Q4'
    #     quartiles.append(quartile)
    # df['quart_note'] = quartiles
    return df

def main():
    trainpath = '../data/train.conll'
    devpath = '../data/dev.conll'
    nlp =nl_core_news_sm.load()
    df_train = get_features(trainpath,nlp)
    df_dev = get_features(devpath,nlp)
    df_train.to_csv('../data/train_with_feat.conll',sep='\t',index=False)
    df_dev.to_csv('../data/dev_with_feat.conll',sep='\t',index=False)

if __name__=='__main__':
    main()