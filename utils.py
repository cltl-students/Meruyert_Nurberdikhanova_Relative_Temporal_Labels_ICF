import pandas as pd
import csv, spacy, nl_core_news_sm, os 
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
def get_rel_time_subset(dfpath:str, data_type:str) -> None:
    '''
    This function saves "background", "target", and "now" subsets used for the project from the original cleaned pickle files.

    :param dfpath: str path to the pickle file
    :param data_type: str specifying data split type

    :return None:
    '''
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

def merge_data(rel_time: str) -> None:
    '''
    This function merges and saves training, development, and test data of an original temporal label.
    
    :param rel_time: str specifying "background", "target", and "now" subsets

    :return None:
    '''
    train = pd.read_csv(f'../data/train_{rel_time}.tsv',sep='\t')
    dev = pd.read_csv(f'../data/dev_{rel_time}.tsv',sep='\t')
    test = pd.read_csv(f'../data/test_{rel_time}.tsv',sep='\t')
    merged_df = pd.concat([train,dev,test],ignore_index=True)
    merged_df.to_csv(f'../data/{rel_time}_merged.tsv', sep='\t', index=False)
    print(f'{rel_time} sets are merged! {len(merged_df)} sentences belong to {rel_time}.')

def split_subgroup(rel_time:str) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    '''
    This function shuffles the instances in a specified temporal subset and creates a new train, dev, test split.

    :param rel_time: str specifying "background", "target", and "now" subsets

    :return train: pandas DataFrame of training subset
    :return dev: pandas DataFrame of development subset
    :return test: pandas DataFrame of test subset
    '''
    df = pd.read_csv(f'../data/{rel_time}_merged.tsv',sep='\t')
    if rel_time == 'background':
        df['rel_time'] = 'past'
    elif rel_time == 'target':
        df['rel_time'] = 'future'
    elif rel_time == 'current':
        df['rel_time'] = 'now'
    else:
        raise ValueError('Please choose one of the existing rel_time options.')
    # sample 80% for train
    train = df.sample(frac=0.8,random_state=27) # random state for reproducibility
    # put aside the remaining 20%
    rest = df.drop(train.index)
    # sample 50% of remaining 20% aka 10% for dev
    dev = rest.sample(frac=0.5,random_state=27)
    # shuffle through the remainder for the test chunk
    test = rest.drop(dev.index).sample(frac=1)
    return train, dev, test

def merge_shuffle_split(df_p: pd.DataFrame, df_f: pd.DataFrame, df_n: pd.DataFrame, data_type: str) -> None:
    '''
    This function merges together and saves temporal subsets of specified data split.

    :param df_p: pandas DataFrame of past label
    :param df_f: pandas DataFrame of future label
    :param df_n: pandas DataFrame of now label
    :param data_type: str specyfing data split subset (train, dev, test)

    :return None:
    '''
    df = pd.concat([df_p,df_f, df_n])
    df = df.sample(frac=1)
    df.drop(['annotator','background_sent','target_sent','text_raw','original_labels','index','pred_domains_eb_ap_mod1','pred_domains_eb_ap_mod2'],axis=1, inplace=True)
    df.to_csv(f'../data/{data_type}.tsv',sep='\t',index=False)
    print(f'Created {data_type} set! {len(df)} sentences long.')

def make_df(split: list, table_type: str) -> None:
    '''
    This function saves a dataframe of specified descriptive statistics as a .TSV file and as a latex file.

    :param split: list of pandas DataFrames containing train, dev, and test subsets
    :param table_type: str specifying type of descriptive statistics table to generate

    :return None:
    '''
    df_list = []
    for data in split:
        types = data[table_type].unique()
        type_dict = {}
        for i in types:
            num = len(data.loc[data[table_type] == i])
            type_dict[i] = num
        df_list.append(type_dict)
    df = pd.DataFrame.from_dict(df_list,orient='columns')
    df['split'] = ['train','dev','test']
    df.set_index('split',inplace=True)
    df.index.names = [None]
    df.fillna(value=0,inplace=True)
    df = df.reindex(sorted(df.columns),axis=1,copy=False)
    if table_type == 'batch':
        df = df.T
    outfile = f'{table_type}.tsv'
    outdir = Path('./descriptives')
    outdir.mkdir(parents=True,exist_ok=True)
    df.to_csv(outdir/outfile,sep='\t')
    df.style.to_latex(f'./descriptives/{table_type}_latex.txt')

def get_descriptives(table_type: str) -> None:
    '''
    This function creates descriptive statistics of a specified type for train, dev, and test split and saves them to a descriptives directory.

    :param table_type: str specifying type of descriptive statistics to get
    '''
    outdir = Path('./descriptives')
    outdir.mkdir(parents=True,exist_ok=True)
    trainpath = '../data/train.tsv'
    devpath = '../data/dev.tsv'
    testpath = '../data/test.tsv'
    split = [pd.read_csv(trainpath,sep='\t'),pd.read_csv(devpath,sep='\t'),pd.read_csv(testpath,sep='\t')]
    if table_type == 'batch':
        make_df(split,table_type)
    elif table_type == 'year':
        make_df(split,table_type)
    elif table_type == 'institution':
        make_df(split,table_type)
    elif table_type == 'natural':
        trainpklpath = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/train.pkl'
        devpklpath = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/dev.pkl'
        testpklpath = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/test.pkl'
        pkl = [pd.read_pickle(trainpklpath),pd.read_pickle(devpklpath),pd.read_pickle(testpklpath)]
        df_list = []
        for df, df_pkl in zip(split,pkl):
            df_dict = {}
            df_dict['Jenia'] = len(df_pkl)
            past = len(df.loc[df['rel_time'] == 'past'])
            df_dict['past'] = past
            now = len(df.loc[df['rel_time'] == 'now'])
            df_dict['now'] = now
            future = len(df.loc[df['rel_time'] == 'future'])
            df_dict['future'] = future
            df_list.append(df_dict)
        df_natural = pd.DataFrame.from_dict(df_list,orient='columns')
        df_natural['split'] = ['train','dev','test']
        df_natural.set_index('split',inplace=True)
        df_natural.index.names = [None]
        df_natural['past_perc'] = df_natural['past'].div(df_natural['Jenia'])
        df_natural['past_perc'] = df_natural['past_perc'].mul(100).round(decimals=2)
        df_natural['now_perc'] = df_natural['now'].div(df_natural['Jenia'])
        df_natural['now_perc'] = df_natural['now_perc'].mul(100).round(decimals=2)
        df_natural['future_perc'] = df_natural['future'].div(df_natural['Jenia'])
        df_natural['future_perc'] = df_natural['future_perc'].mul(100).round(decimals=2)
        df_natural.insert(2,'past_perc',df_natural.pop('past_perc'))
        df_natural.insert(4,'now_perc',df_natural.pop('now_perc'))
        outfile = f'{table_type}.tsv'
        outlatex = f'{table_type}_latex.txt'
        df_natural.to_csv(outdir/outfile,sep='\t')
        df_natural.style.to_latex(outdir/outlatex)
    elif table_type == 'note_patient':
        df_list = []
        for data in split:
            df_dict = {}
            df_dict['notes'] = len(data.NotitieID.unique())
            df_dict['patients'] = len(data.MDN.unique())
            df_list.append(df_dict)
        df = pd.DataFrame.from_dict(df_list,orient='columns')
        df['split'] = ['train','dev','test']
        df.set_index('split',inplace=True)
        df.index.names = [None]
        outfile = f'{table_type}.tsv'
        outlatex = f'{table_type}_latex.txt'
        df.to_csv(outdir/outfile,sep='\t')
        df.style.to_latex(outdir/outlatex)
    else:
        raise ValueError('Invalid table_type input! \
                         Please choose one of the existing options: "batch", "year","institution","natural","note_patient".')
    
    print(f'{table_type} descriptives are saved!')

def get_full_notes(rawpath:str,df:pd.DataFrame) -> pd.DataFrame:
    '''
    This function links specified dataset with the full notes from the raw datafile.

    :param rawpath: str path to raw datafile
    :param df: pandas DataFrame containing the dataset

    :return intersection: pandas DataFrame of given dataset instances that have full notes from specified path
    '''
    df_raw = pd.read_pickle(rawpath)
    df = df.astype('string')
    df_raw = df_raw.astype('string')
    intersection = df.merge(df_raw,on='NotitieID')
    intersection.drop(['institution_y','MDN_y','Typenotitie'],axis=1, inplace=True)
    intersection.rename(columns={'institution_x':'institution','MDN_x':'MDN'},inplace=True)
    intersection.drop_duplicates(inplace=True)
    print(f'{len(intersection)} rows in intersection, {len(intersection.pad_sen_id.unique())} unique in intersection')
    return intersection

def remove_older_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    '''
    This function removes instances of the data that had several duplicate full notes. 
    This is a preprocessing step.

    :param df: pandas DataFrame that has duplicate instances

    :return df_copy: copy of the original pandas DataFrame without the duplicates
    '''
    df['Notitiedatum'] = pd.to_datetime(df['Notitiedatum'],format='mixed')
    # sometimes duplicates have the same date => need to remove the shorter note or check if the sentence is in the note text
    df['index'] = df.index.values
    df_copy = df.copy()
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
    df_copy.sort_values(by='Notitiedatum',inplace=True)
    df_copy.drop_duplicates(subset=['pad_sen_id'],inplace=True,keep='last')
    df_copy.drop(columns=['index'],inplace=True)
    df_copy= df_copy.sample(frac=1)
    return df_copy

def save_full_notes(df:pd.DataFrame,data_type:str) -> None:
    '''
    This function links full notes to specified dataset's instances from rawfiles of years 2017, 2018, and 2020.
    The linked dataset is then saved as a .TSV file.

    :param df: pandas DataFrame containing the dataset
    :param data_type: str specifying the data split (train, dev, test)

    :return None:
    '''
    raw_2017='/mnt/data/A-Proof/data2/a-proof-zonmw/data/2017_raw/processed_2017.pkl'
    raw_2018='/mnt/data/A-Proof/data2/a-proof-zonmw/data/2018_raw/processed_2018.pkl'
    raw_2020='/mnt/data/A-Proof/data2/a-proof-zonmw/data/2020_raw/processed_2020.pkl'
    intersection2017 = get_full_notes(raw_2017,df)
    intersection2018 = get_full_notes(raw_2018,df)
    intersection2020 = get_full_notes(raw_2020,df)
    df_notes = pd.concat([intersection2017,intersection2018,intersection2020])
    df_notes.drop_duplicates(inplace=True)
    print(f'{len(df_notes)} all sentences, {len(df_notes.pad_sen_id.unique())} unique sentences')
    df_no_dupes = remove_older_duplicates(df_notes)
    df_no_dupes.to_csv(f'../data/{data_type}_full_notes.tsv',sep='\t',index=False)
    print(f'Full notes linked to {data_type} set!')

def tsv_to_conll(inpath:str, outpath:str) -> None:
    '''
    This function converts a sentence based dataset into token level dataset and saves it as a CoNLL formatted file.

    :param inpath: str path to the dataset to convert
    :param outpath: str path to save the CoNLL dataset

    :return None:
    '''
    with open(inpath, 'r',encoding='utf-8') as infile:
        csv_reader = csv.reader(infile, delimiter='\t')
        next(csv_reader)
        with open(outpath, 'w', encoding='utf-8', newline='') as outfile:
            header = 'token\tpad_sent_id\tinstitution\tyear\tMDN\tNotitieID\tbatch\trel_time\tnote_len\tnote_date\tUPOS\tXPOS\thead:dep\n'
            outfile.write(header)
            for line in csv_reader:
                pad_sent_id = line[0]
                institution = line[1]
                year = line[2]
                mdn = line[3]
                note_id = line[4]
                batch = line[5]
                # icf_l_list = line[6]
                text = line[7]
                # len_text = line[8]
                rel_time = line[9]
                # tokenise text, get its upox, xpos, dependency label and head
                # get lenght of the note as well (in num of sentences)
                # add back the other columns
                nlp = nl_core_news_sm.load()
                doc = nlp(text)
                tab = '\t'
                note_date = line[10]
                full_note_text = line[11]
                doc_note = nlp(full_note_text)
                note_len = str(len(list(doc_note.sents)))
                tokens = [(token.text, token.pos_,token.tag_,token.dep_,token.head.text) for token in doc]
                rows = [f'{token[0]}{tab}{pad_sent_id}{tab}{institution}{tab}{year}{tab}{mdn}{tab}{note_id}{tab}{batch}{tab}{rel_time}{tab}{note_len}{tab}{note_date}{tab}{token[1]}{tab}{token[2]}{tab}{token[4]}:{token[3]}' for token in tokens]
                conll = '\n'.join(rows)
                conll = conll+'\n'
                outfile.write(conll)

def write_features(path:str, data_type:str, pipeline) -> None:
    '''
    This function adds features used for SVM model to the specified dataset and saves them as a new file.
    The added features are:
    [x] sentence position in a note 
    [x] note quartile (relative position)
    [x] temporal Named Entity
    [x] verb 
    [x] tense of the verb

    :param path: str path to the dataset
    :param data_type: str specifying the data split (train, dev, test)
    :param pipeline: spaCy Dutch language model

    :return None:
    '''
    df = pd.read_csv(path,sep='\t',encoding='utf-8',quoting=3,low_memory=False)
    # sentence number
    pad_list = df['pad_sent_id'].tolist()
    sent_ids = []
    for pad in pad_list:
        note_sent_id = pad.split('_')
        sent_id = note_sent_id[1]
        sent_id = sent_id.lstrip('0')
        sent_ids.append(sent_id)
    df['sent_id'] = sent_ids
    # sentence position in a note
    # as a string
    sent_ids = df['sent_id'].to_list()
    note_len = df['note_len'].to_list()
    df['rel_position'] = [f'{sent}_{note}' for sent, note in zip(sent_ids,note_len)]
    # as a ratio => quartiles
    quartiles = []
    for sent, note in zip(sent_ids,note_len):
        sent = float(sent)
        note = float(note)
        rat = sent/note
        if rat < .25:
            quartile = 'Q1'
        elif .25 <=rat <= .5:
            quartile = 'Q2'
        elif .5 <= rat <= .75:
            quartile = 'Q3'
        else:
            quartile = 'Q4'
        quartiles.append(quartile)
    df['quart_note'] = quartiles
    # try ner tagger for time and date
    # time => times smaller than a day
    # date => absolute or relative dates or periods
    # https://github.com/explosion/spaCy/issues/441
    sentences = df.groupby('pad_sent_id')
    ner_primary_list = []
    for _, sentence in sentences:
        tokens = sentence['token'].to_list()
        sent = ' '.join(tokens)
        doc = pipeline(sent)
        ner_list = []
        for i, tok in enumerate(tokens):
            ner = doc[i].ent_type_
            if ner in ('DATE','TIME'):
                ner_list.append(ner)
            else:
                ner_list.append('not_temp_ner')
        ner_primary_list.append(ner_list)
    temp_ner = [ner for ner_l in ner_primary_list for ner in ner_l]
    df['temp_ner'] = temp_ner    
    # tense of the verb
    morphology = df['XPOS'].to_list()
    verbs = []
    tense = []
    for tag in morphology:
        if 'WW' in tag:
            verbs.append(tag)
            if 'pv|tgw' in tag:
                tense.append('present_tense')
            elif 'pv|verl' in tag:
                tense.append('past_tense')
            elif 'inf|vrij' in tag:
                tense.append('infinitive')
            else:
                tense.append('-')
        else:
            verbs.append('-')
            tense.append('-')
    df['VERB'] = verbs
    df['tense'] = tense
    df.to_csv(f'../data/{data_type}_with_features.conll',sep='\t',index=False)

def make_feature_df(subset_l: list,column:str,data_type:str) -> None:
    '''
    This function creates a table with simple counts of instances in specified feature column of the dataset.
    The table is saved as a .TSV file and as a LaTeX table to a corpus directory.

    :param subset_l: list of pandas DataFrames of the temporal subsets
    :param column: str column name in the DataFrames
    :param data_type: str specifying the data split 

    :return None:
    '''
    df_list = []
    for df in subset_l:
        names = df[column].unique()
        name_dict = {}
        if column == 'quart_note':
            sentences = df.groupby(['pad_sent_id'])
            for name in names:
                count = 0
                for _, sentence in sentences:
                    item = sentence[column].tolist()[0]
                    if item == name:
                        count+=1
                name_dict[name] = count
        else:
            for name in names:
                count = len(df.loc[df[column] == name])
                name_dict[name] = count
        df_list.append(name_dict)
    feature = pd.DataFrame.from_dict(df_list,orient='columns')
    feature['time'] = ['past','now','future']
    feature.set_index('time',inplace=True)
    feature.index.names = [None]
    feature.fillna(value=0,inplace=True)
    feature = feature.reindex(sorted(feature.columns),axis=1,copy=False)
    feature_df = feature.T
    outfile = f'{data_type}_{column}.tsv'
    outdir = Path('./corpus')
    outdir.mkdir(parents=True,exist_ok=True)
    feature_df.to_csv(outdir/outfile,sep='\t')
    feature_df.style.to_latex(f'./corpus/{data_type}_{column}_latex.txt')

def stats_feature(subset_l: list,column:str,data_type:str) -> None:
    '''
    This function creates a table with simple distributional statistics of a specified numerical column in the dataset.
    The table is saved as a .TSV file and as a LaTeX table to a corpus directory.

    :param subset_l: list of pandas DataFrames of the temporal subsets
    :param column: str column name in the DataFrames
    :param data_type: str specifying the data split 

    :return None:
    '''
    df_list = []
    for df in subset_l:
        column_dict = {}
        df[column].astype('float')
        column_dict['mean'] = df[column].mean()
        column_dict['median'] = df[column].median()
        column_dict['std'] = df[column].std()
        column_dict['min'] = df[column].min()
        column_dict['max'] = df[column].max()
        df_list.append(column_dict)
    stats = pd.DataFrame.from_dict(df_list,orient='columns')
    stats['time'] = ['past','now','future']
    stats.set_index('time',inplace=True)
    stats.index.names = [None]
    stats_df = stats.T
    outfile = f'{data_type}_{column}.tsv'
    outdir = Path('./corpus')
    outdir.mkdir(parents=True,exist_ok=True)
    stats_df.to_csv(outdir/outfile,sep='\t')
    stats_df.style.to_latex(f'./corpus/{data_type}_{column}_latex.txt')

def corpus_analysis(inpath:str,data_type:str) -> None:
    '''
    This function gathers distributions for a corpus analysis of the SVM model features by their temporal labels.
    The features analysed are: verb,tense, note quartile, note length, and temporal Named Entity.

    :param inpath: str path to the dataset
    :param data_type: str specifying the data split 

    :return None:
    '''
    df = pd.read_csv(inpath,sep='\t')
    past = df.loc[df['rel_time'] == 'past']
    now = df.loc[df['rel_time'] == 'now']
    future = df.loc[df['rel_time'] == 'future']
    split = [past,now,future]
    ### check the distributions of features ###
    # 1 # verb
    make_feature_df(split,'VERB',data_type)
    # 2 # tense
    make_feature_df(split,'tense',data_type)
    # 3 # sentence position in the note (quartiles)
    make_feature_df(split,'quart_note',data_type)
    # 4 # sentence length
    stats_feature(split,'note_len',data_type)
    # 5 # temp NER
    # how often they occur in each subset
    make_feature_df(split,'temp_ner',data_type)

def convert_transformer_labels(df:pd.DataFrame) -> pd.DataFrame:
    '''
    This function converts numerical labels assigned by the transformer model back to the str temporal labels we are using.

    :param df: pandas DataFrame with transformer model predictions

    :return df: pandas DataFrame with human-readable transformer model predictions
    '''
    mapping = {0: 'past', 1: 'now', 2: 'future'}
    df['predictions'] = df['pred_medroberta'].map(mapping)
    return df

def get_evaluation_metrics(filepath:str,model_type:str,data_type:str,feat_num=None) -> None:
    '''
    This function creates a classification report on predictions of a specified model.
    The classification report is printed and saved into a reports directory.

    :param filepath: str path to the dataset with predictions
    :param model_type: str specifying model name used (medroberta, svm)
    :param data_type: str specifying data split (dev, test)
    :param feat_num: optional str specifying number of features used in the experiment

    :return None:
    '''
    df = pd.read_csv(filepath,sep='\t')
    if model_type == 'medroberta': # get human-readable prediction labels
        df = convert_transformer_labels(df)
    predictions = df['predictions']
    gold_labels = df['rel_time']
    label_names = ['past','now','future']
    print(classification_report(gold_labels,predictions,target_names=label_names))
    report_dict = classification_report(gold_labels,predictions,target_names=label_names,output_dict=True)
    report_df = pd.DataFrame(report_dict).T 
    outdir = Path('./reports')
    outdir.mkdir(parents=True,exist_ok=True)
    if feat_num:
        modelpath = f'./reports/classification_report_{model_type}_{feat_num}_{data_type}.tsv'
        latexpath = f'./reports/latex_classification_report_{model_type}_{feat_num}_{data_type}.txt'
    else:
        modelpath = f'./reports/classification_report_{model_type}_{data_type}.tsv'
        latexpath = f'./reports/latex_classification_report_{model_type}_{data_type}.txt'
    report_df.to_csv(modelpath,sep='\t')
    report_df.style.to_latex(latexpath)

def make_heatmap(inpath:str,model_type:str,exp:str,normal=False) -> None:
    '''
    This function creates a confusion matrix and a visualisation of it for specified predictions.

    :param inpath: str path to the predictions dataset
    :param model_type: str specifying model name used (medroberta, svm)
    :param exp: str specifying experiment type and data split (dev, test)
    :param normal: optional boolean specifying whether to use normalisation in the visualisation

    :return None:
    '''
    df = pd.read_csv(inpath,sep='\t')
    if model_type == 'medroberta': # get human readable prediction labels
        df = convert_transformer_labels(df)
    if normal:
        confusion = confusion_matrix(df['rel_time'],df['predictions'],normalize='true')
        confusion_df_n = pd.DataFrame(confusion, index=['true_past','true_now','true_future'],columns=['pred_past','pred_now','pred_future'])
        confusion_df_n.to_csv(f'norm_confusion_{model_type}_{exp}.tsv',sep='\t')
        heatmap = sns.heatmap(confusion_df_n, fmt='.2f',annot=True)
        fig = heatmap.get_figure()
        fig.savefig(f'norm_confusion_{model_type}_{exp}.jpg')
    else:
        confusion = confusion_matrix(df['rel_time'],df['predictions'])
        confusion_df = pd.DataFrame(confusion, index=['true_past','true_now','true_future'],columns=['pred_past','pred_now','pred_future'])
        confusion_df.to_csv(f'confusion_{model_type}_{exp}.tsv',sep='\t')
        heatmap = sns.heatmap(confusion_df, fmt='d',annot=True)
        fig = heatmap.get_figure()
        fig.savefig(f'confusion_{model_type}_{exp}.jpg')
    plt.clf()

def write_misclassified(inpath:str, model_type:str, exp:str) -> None:
    '''
    This function saves misclassified instances by temporal labels into new files.

    :param inpath: str path to the dataset
    :param model_type: str specifying model name used (medroberta, svm)
    :param exp: str specifying experiment type and data split (dev, test)

    :return None:
    '''
    df = pd.read_csv(inpath, sep='\t')
    if model_type == 'medroberta': # get human readable prediction labels
        df = convert_transformer_labels(df)
    future_past = df.loc[(df['rel_time'] == 'future') & (df['predictions'] == 'past')]
    future_now = df.loc[(df['rel_time'] == 'future') & (df['predictions'] == 'now')]
    future_past.to_csv(f'../data/future_past_{model_type}_{exp}.tsv',sep='\t')
    future_now.to_csv(f'../data/future_now_{model_type}_{exp}.tsv',sep='\t')
    past_future = df.loc[(df['rel_time'] == 'past') & (df['predictions'] == 'future')]
    past_now = df.loc[(df['rel_time'] == 'past') & (df['predictions'] == 'now')]
    past_future.to_csv(f'../data/past_future_{model_type}_{exp}.tsv',sep='\t')
    past_now.to_csv(f'../data/past_now_{model_type}_{exp}.tsv',sep='\t')
    now_future = df.loc[(df['rel_time'] == 'now') & (df['predictions'] == 'future')]
    now_past = df.loc[(df['rel_time'] == 'now') & (df['predictions'] == 'past')]
    now_future.to_csv(f'../data/now_future_{model_type}_{exp}.tsv',sep='\t')
    now_past.to_csv(f'../data/now_past_{model_type}_{exp}.tsv',sep='\t')

def get_error_conll(inpath:str) -> None:
    '''
    This function links dataset of misclassified instances with CoNLL formatted test set.
    The resulting CoNLL dataset of misclassifications is saved and used for error analysis.

    :param inpath: str path to the dataset

    :return None:
    '''
    df = pd.read_csv(inpath,sep='\t')
    df_conll = pd.read_csv('../data/test_with_features.conll',sep='\t')
    sent_ids = df['pad_sen_id'].tolist()
    errors = df_conll[df_conll['pad_sent_id'].isin(sent_ids)]
    stripped_path = inpath.removesuffix('.tsv')
    errors.to_csv(f'{stripped_path}.conll',sep='\t',index=False)