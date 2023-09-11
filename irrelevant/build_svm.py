import pandas as pd 
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm 
import joblib
from pathlib import Path
from scipy.sparse import hstack

def get_features(filepath:str,feat_num:str) -> tuple:
    df = pd.read_csv(filepath,sep='\t')
    sentences = df.groupby(['pad_sent_id'])
    data = []
    gold_labels = []
    for _, sentence in sentences:
        #0#
        tokens = sentence['token']
        #1#
        upos = sentence['UPOS']
        #2#
        quartiles = sentence['quart_note'].tolist()
        quartile = quartiles[0]
        #3#
        tense = sentence['tense']
        #4#
        temp_ner = sentence['temp_ner']
        #5#
        verbs = sentence['VERB']
        #6#
        deps = sentence['head:dep']
        #7#
        notelengths = sentence['note_len'].tolist()
        notelength = notelengths[0]
        #8#
        xpos = sentence['XPOS']
        # if feat_num == '0':
        #     feature_dict = {'tokens': tokens}
        # elif feat_num == '1':
        #     feature_dict = {'tokens': tokens,'upos': upos}
        # elif feat_num == '2':
        #     feature_dict = {'tokens': tokens,'upos': upos, 'quartiles': quartile}
        # elif feat_num == '3':
        #     feature_dict = {'tokens': tokens,'upos': upos, 'quartiles': quartile,
        #                     'tense': tense}
        # elif feat_num == '4':
        #     feature_dict = {'tokens': tokens,'upos': upos, 'quartiles': quartile, 
        #                     'tense': tense, 'temp_ner': temp_ner}
        # elif feat_num == '5':
        #     feature_dict = {'tokens': tokens,'upos': upos, 'quartiles': quartile, 
        #                     'tense': tense, 'temp_ner': temp_ner, 'verbs': verbs}
        # elif feat_num == '6':
        #     feature_dict = {'tokens': tokens,'upos': upos, 'quartiles': quartile, 
        #                     'tense': tense, 'temp_ner': temp_ner, 'verbs': verbs,
        #                     'note_len': notelength}
        # elif feat_num == '7':
        #     feature_dict = {'tokens': tokens,'upos': upos, 'quartiles': quartile, 
        #                     'tense': tense, 'temp_ner': temp_ner, 'verbs': verbs,
        #                     'note_len': notelength,'xpos': xpos}
        # elif feat_num == '8':
        #     feature_dict = {'tokens': tokens,'upos': upos, 'quartiles': quartile, 
        #                     'tense': tense, 'temp_ner': temp_ner, 'verbs': verbs,
        #                     'note_len': notelength, 'xpos': xpos,'deps': deps}
        # else:
        #     raise ValueError('Incorrect input! Please insert a value between "0" and "8".')
        if feat_num == '1':
            feature_dict = {'upos': upos}
        elif feat_num == '2':
            feature_dict = {'upos': upos, 'quartiles': quartile}
        elif feat_num == '3':
            feature_dict = {'upos': upos, 'quartiles': quartile,
                            'tense': tense}
        elif feat_num == '4':
            feature_dict = {'upos': upos, 'quartiles': quartile, 
                            'tense': tense, 'temp_ner': temp_ner}
        elif feat_num == '5':
            feature_dict = {'upos': upos, 'quartiles': quartile, 
                            'tense': tense, 'temp_ner': temp_ner, 'verbs': verbs}
        elif feat_num == '6':
            feature_dict = {'upos': upos, 'quartiles': quartile, 
                            'tense': tense, 'temp_ner': temp_ner, 'verbs': verbs,
                            'note_len': notelength}
        elif feat_num == '7':
            feature_dict = {'upos': upos, 'quartiles': quartile, 
                            'tense': tense, 'temp_ner': temp_ner, 'verbs': verbs,
                            'note_len': notelength,'xpos': xpos}
        elif feat_num == '8':
            feature_dict = {'upos': upos, 'quartiles': quartile, 
                            'tense': tense, 'temp_ner': temp_ner, 'verbs': verbs,
                            'note_len': notelength, 'xpos': xpos,'deps': deps}
        else:
            raise ValueError('Incorrect input! Please insert a value between "0" and "8".')
        data.append(feature_dict)
    return data, df

def get_gold_labels(filepath:str) -> list:
    df = pd.read_csv(filepath,sep='\t')
    sentences = df.groupby(['pad_sent_id'])
    gold_labels = []
    for _, sentence in sentences:
        head = sentence.head(1)
        gold_label = head['rel_time']
        gold_labels.append(gold_label)
    return gold_labels

def vectorise_features(data:list) -> tuple:
    vec = DictVectorizer()
    vectorised_features = vec.fit_transform(data)
    return vec, vectorised_features

def create_svm_clf(filepath:str, feat_num:str) -> tuple:
    data, conll = get_features(filepath, feat_num)
    # gold_labels = get_gold_labels(filepath)
    vec, vectorised_features = vectorise_features(data)
    # tfidf #
    df = pd.read_csv('../data/train_full_notes.tsv',sep='\t')
    # tfidf = TfidfVectorizer()
    # X = tfidf.fit_transform(df['text'])
    # joblib.dump(tfidf,'tfidf.joblib')
    # all_features = hstack([X,vectorised_features],format='csr')
    gold_labels = df['rel_time']
    # end #
    print(f'shape of vectors with {feat_num} features:')
    print(vectorised_features.shape)
    model = svm.SVC(kernel='linear')
    print('Creating SVM classifier...')
    model.fit(vectorised_features,gold_labels)
    save_model_and_vec(model,vec,feat_num)
    return model, vec

def save_model_and_vec(model, vec:DictVectorizer,feat_num:str) -> None:
    outdir = Path('./svm')
    outdir.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,f'{outdir}/svm_{feat_num}.joblib')
    joblib.dump(vec,f'{outdir}/vectorizer_{feat_num}.joblib')

def write_predictions(sentencepath:str,predictions:list) -> pd.DataFrame:
    sent_df = pd.read_csv(sentencepath,sep='\t')
    sent_df['predictions'] = predictions
    return sent_df

def classify_data(featurepath:str, feat_num:str, sentencepath:str, model, vec) -> None:
    # model = joblib.load(f'svm/svm_{feat_num}.joblib')
    # vec = joblib.load(f'svm/vectorizer_{feat_num}.joblib')
    data, df = get_features(featurepath,feat_num)
    vectorised_data = vec.transform(data)
    # tfidf #
    # tfidf = joblib.load('tfidf.joblib')
    # full_df = pd.read_csv(sentencepath,sep='\t')
    # X = tfidf.transform(full_df['text'])
    # all_features = hstack([X,vectorised_data],format='csr')
    # end #
    predictions = model.predict(vectorised_data)
    sent_df = write_predictions(sentencepath,predictions)
    stripped_path = sentencepath.removesuffix('.tsv')
    if 'dev' in stripped_path:
        stripped_path = stripped_path.removesuffix('_full_notes')
    outpath = f"{stripped_path}_svm_{feat_num}_predictions.tsv"
    sent_df.to_csv(outpath,sep='\t',index=False)
    print(f"A column with predictions was added.\nThe updated df is saved: {outpath}")

