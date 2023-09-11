from utils import get_evaluation_metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn import svm
from scipy.sparse import hstack
import joblib
from pathlib import Path

def save_model_and_vec(model, vec:DictVectorizer,feat_num:str) -> None:
    '''
    This function saves specified trained model and feature vectoriser as .joblib files in svm directory.

    :param model: trained SVM model 
    :param vec: Dictionary Vectoriser used for training
    :param feat_num: str specifying number of features or experiment to identify the files by

    :return None:
    '''
    outdir = Path('./svm')
    outdir.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,f'{outdir}/svm_{feat_num}.joblib')
    joblib.dump(vec,f'{outdir}/vectorizer_{feat_num}.joblib')

def deps_feature(conll:pd.DataFrame) -> pd.DataFrame:
    '''
    This function simplifies dependency feature by removing token's head of dependency from a specified dataset.

    :param conll: pandas DataFrame containing the dataset

    :return conll: modified pandas DataFrame of the dataset
    '''
    headdeps = conll['head:dep'].tolist()
    deps = []
    for item in headdeps:
        dep = item.split(':')[1]
        deps.append(dep)
    conll['deps'] = deps
    return conll

def extract_feats(conll:pd.DataFrame, exp_num:int) -> list:
    '''
    This function extracts features for specified SVM experiment from the dataset to use for training.

    :param conll: pandas DataFrame containing the dataset
    :param exp_num: int specifying the experiment

    :return data: list of dictionaries of features
    '''
    sentences = conll.groupby(['pad_sent_id'])
    data = []
    for _, sentence in sentences:
        if exp_num == 1: # aka morphological
            featdict = {'UPOS': sentence['UPOS'], 'XPOS': sentence['XPOS'],'deps': sentence['deps']}
        elif exp_num == 2: # aka discourse
            note_len = sentence['note_len'].tolist()[0]
            quart_note = sentence['quart_note'].tolist()[0]
            featdict = {'quart_note': quart_note,'note_len': note_len}
        elif exp_num == 3: # aka temporal
            featdict = {'tense': sentence['tense'],'verb': sentence['VERB'],'temp_ner': sentence['temp_ner']}
        elif exp_num == 4: # aka modge-podge
            note_len = sentence['note_len'].tolist()[0]
            quart_note = sentence['quart_note'].tolist()[0]
            featdict = {'UPOS': sentence['UPOS'], 'XPOS': sentence['XPOS'],'deps': sentence['deps'],
                        'quart_note': quart_note,'note_len': note_len,
                        'tense': sentence['tense'],'verb': sentence['VERB'],'temp_ner': sentence['temp_ner']}
        else:
            raise ValueError('Wrong experiment number! Please provide an integer number between 1 and 4.')
        data.append(featdict)
    return data

def perform_svm_exp(exp_num:int, data_type: str) -> None:
    '''
    This function trains and predicts temporal labels using an SVM model of a specified predetermined experiment.
    Predictions are saved in a new .TSV file, as well as the model and feature vectoriser. 
    Classification report is printed and saved for the experiment.

    :param exp_num: int specifying the experiment
    :param data_type: str specifying the data split model is tested on

    :return None:
    '''
    print(f'Working on SVM experiment number {exp_num}...')
    conll = pd.read_csv('../data/train_with_features.conll',sep='\t')
    conll = deps_feature(conll)
    data = extract_feats(conll, exp_num)
    vec = DictVectorizer()
    train_vectors = vec.fit_transform(data)
    df = pd.read_csv('../data/train_full_notes.tsv',sep='\t')
    gold_labels = df['rel_time']
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['text'])
    train_feat = hstack([X,train_vectors],format='csr')
    print(f'shape of vectors for experiment {str(exp_num)}:')
    print(train_feat.shape)
    model = svm.SVC(kernel = 'linear')
    model.fit(train_feat,gold_labels)
    save_model_and_vec(model,vec,f'exp_{str(exp_num)}')
    dev_conll = pd.read_csv(f'../data/{data_type}_with_features.conll',sep='\t')
    dev_conll = deps_feature(dev_conll)
    dev_data = extract_feats(dev_conll,exp_num)
    dev_vectors = vec.transform(dev_data)
    dev_df = pd.read_csv('../data/test_full_notes.tsv',sep = '\t')
    dev_X = tfidf.transform(dev_df['text'])
    dev_feat = hstack([dev_X,dev_vectors],format='csr')
    predictions = model.predict(dev_feat)
    dev_df['predictions'] = predictions
    dev_df.to_csv(f'../data/{data_type}_svm_exp_{exp_num}_predictions.tsv',sep='\t',index=False)
    get_evaluation_metrics(f'../data/{data_type}_svm_exp_{exp_num}_predictions.tsv', 'svm', data_type, f'exp_{exp_num}')
