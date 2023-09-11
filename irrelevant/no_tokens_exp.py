from build_svm import *
from utils import get_evaluation_metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
from sklearn import svm
from scipy.sparse import hstack

def deps_feature(conll):
    headdeps = conll['head:dep'].tolist()
    deps = []
    for item in headdeps:
        dep = item.split(':')[1]
        deps.append(dep)
    conll['deps'] = deps
    return conll

def select_1st_feat(feature:str):
    conll = pd.read_csv('../data/train_with_features.conll',sep='\t')
    conll = deps_feature(conll)
    sentences = conll.groupby(['pad_sent_id'])
    data = []
    for _, sentence in sentences:
        if feature == 'note_len':
            note_len = sentence['note_len'].tolist()[0]
            featdict = {'note_length': note_len}
        elif feature == 'quart_note':
            quart_note = sentence['quart_note'].tolist()[0]
            featdict = {'quart_note': quart_note}
        else:
            featdict = {feature: sentence[feature]}
        data.append(featdict)
    vec = DictVectorizer()
    train_vectors = vec.fit_transform(data)
    df = pd.read_csv('../data/train_full_notes.tsv',sep='\t')
    gold_labels = df['rel_time']
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['text'])
    train_feat = hstack([X,train_vectors],format='csr')
    print(f'shape of vectors with tfidf and {feature}:')
    print(train_feat.shape)
    model = svm.SVC(kernel = 'rbf')
    model.fit(train_feat,gold_labels)
    save_model_and_vec(model,vec,feature)
    dev_conll = pd.read_csv('../data/dev_with_features.conll',sep='\t')
    dev_conll = deps_feature(dev_conll)
    sentences = dev_conll.groupby(['pad_sent_id'])
    data = []
    for _, sentence in sentences:
        if feature == 'note_len':
            note_len = sentence['note_len'].tolist()[0]
            featdict = {'note_length': note_len}
        elif feature == 'quart_note':
            quart_note = sentence['quart_note'].tolist()[0]
            featdict = {'quart_note': quart_note}
        else:
            featdict = {feature: sentence[feature]}
        data.append(featdict)
    dev_vectors = vec.transform(data)
    dev_df = pd.read_csv('../data/dev_full_notes.tsv',sep = '\t')
    dev_X = tfidf.transform(dev_df['text'])
    dev_feat = hstack([dev_X,dev_vectors],format='csr')
    gold = dev_df['rel_time']
    predictions = model.predict(dev_feat)
    dev_df['predictions'] = predictions
    dev_df.to_csv(f'../data/dev_svm_{feature}_tfidf_predictions.tsv',sep='\t',index=False)
    get_evaluation_metrics(f'../data/dev_svm_{feature}_tfidf_predictions.tsv','svm','dev',f'{feature}_tfidf')

def select_2nd_feat(feature:str):
    conll = pd.read_csv('../data/train_with_features.conll',sep='\t')
    conll = deps_feature(conll)
    sentences = conll.groupby(['pad_sent_id'])
    data = []
    for _, sentence in sentences:
        if feature == 'note_len':
            note_len = sentence['note_len'].tolist()[0]
            featdict = {'deps':sentence['deps'],'note_length': note_len}
        elif feature == 'quart_note':
            quart_note = sentence['quart_note'].tolist()[0]
            featdict = {'deps':sentence['deps'], 'quart_note': quart_note}
        else:
            featdict = {'deps':sentence['deps'], feature: sentence[feature]}
        data.append(featdict)
    vec = DictVectorizer()
    train_vectors = vec.fit_transform(data)
    df = pd.read_csv('../data/train_full_notes.tsv',sep='\t')
    gold_labels = df['rel_time']
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['text'])
    train_feat = hstack([X,train_vectors],format='csr')
    print(f'shape of vectors with tfidf, deps, and {feature}:')
    print(train_feat.shape)
    model = svm.SVC(kernel = 'linear')
    model.fit(train_feat,gold_labels)
    save_model_and_vec(model,vec,feature)
    dev_conll = pd.read_csv('../data/dev_with_features.conll',sep='\t')
    dev_conll = deps_feature(dev_conll)
    sentences = dev_conll.groupby(['pad_sent_id'])
    data = []
    for _, sentence in sentences:
        if feature == 'note_len':
            note_len = sentence['note_len'].tolist()[0]
            featdict = {'note_length': note_len}
        elif feature == 'quart_note':
            quart_note = sentence['quart_note'].tolist()[0]
            featdict = {'deps':sentence['deps'], 'quart_note': quart_note}
        else:
            featdict = {feature: sentence[feature]}
        data.append(featdict)
    dev_vectors = vec.transform(data)
    dev_df = pd.read_csv('../data/dev_full_notes.tsv',sep = '\t')
    dev_X = tfidf.transform(dev_df['text'])
    dev_feat = hstack([dev_X,dev_vectors],format='csr')
    gold = dev_df['rel_time']
    predictions = model.predict(dev_feat)
    dev_df['predictions'] = predictions
    dev_df.to_csv(f'../data/dev_svm_{feature}_tfidf_predictions.tsv',sep='\t',index=False)
    get_evaluation_metrics(f'../data/dev_svm_{feature}_tfidf_predictions.tsv','svm','dev',f'{feature}_tfidf')

def select_3rd_feat(feature:str):
    conll = pd.read_csv('../data/train_with_features.conll',sep='\t')
    conll = deps_feature(conll)
    sentences = conll.groupby(['pad_sent_id'])
    data = []
    for _, sentence in sentences:
        quart_note = sentence['quart_note'].tolist()[0]
        if feature == 'note_len':
            note_len = sentence['note_len'].tolist()[0]
            featdict = {'deps':sentence['deps'],'quart_note':quart_note,'note_length': note_len}
        else:
            featdict = {'deps':sentence['deps'],'quart_note':quart_note, feature: sentence[feature]}
        data.append(featdict)
    vec = DictVectorizer()
    train_vectors = vec.fit_transform(data)
    df = pd.read_csv('../data/train_full_notes.tsv',sep='\t')
    gold_labels = df['rel_time']
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['text'])
    train_feat = hstack([X,train_vectors],format='csr')
    print(f'shape of vectors with tfidf, deps, quartile, and {feature}:')
    print(train_feat.shape)
    model = svm.SVC(kernel = 'linear')
    model.fit(train_feat,gold_labels)
    save_model_and_vec(model,vec,feature)
    dev_conll = pd.read_csv('../data/dev_with_features.conll',sep='\t')
    dev_conll = deps_feature(dev_conll)
    sentences = dev_conll.groupby(['pad_sent_id'])
    data = []
    for _, sentence in sentences:
        quart_note = sentence['quart_note'].tolist()[0]
        if feature == 'note_len':
            note_len = sentence['note_len'].tolist()[0]
            featdict = {'deps':sentence['deps'],'quart_note':quart_note,'note_length': note_len}
        else:
            featdict = {'deps':sentence['deps'],'quart_note':quart_note, feature: sentence[feature]}
        data.append(featdict)
    dev_vectors = vec.transform(data)
    dev_df = pd.read_csv('../data/dev_full_notes.tsv',sep = '\t')
    dev_X = tfidf.transform(dev_df['text'])
    dev_feat = hstack([dev_X,dev_vectors],format='csr')
    gold = dev_df['rel_time']
    predictions = model.predict(dev_feat)
    dev_df['predictions'] = predictions
    dev_df.to_csv(f'../data/dev_svm_{feature}_tfidf_predictions.tsv',sep='\t',index=False)
    get_evaluation_metrics(f'../data/dev_svm_{feature}_tfidf_predictions.tsv','svm','dev',f'{feature}_tfidf')

def select_4th_feat(feature:str):
    conll = pd.read_csv('../data/train_with_features.conll',sep='\t')
    conll = deps_feature(conll)
    sentences = conll.groupby(['pad_sent_id'])
    data = []
    for _, sentence in sentences:
        quart_note = sentence['quart_note'].tolist()[0]
        note_len = sentence['note_len'].tolist()[0]
        featdict = {'deps':sentence['deps'],'quart_note':quart_note,'note_length': note_len, feature: sentence[feature]}
        data.append(featdict)
    vec = DictVectorizer()
    train_vectors = vec.fit_transform(data)
    df = pd.read_csv('../data/train_full_notes.tsv',sep='\t')
    gold_labels = df['rel_time']
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['text'])
    train_feat = hstack([X,train_vectors],format='csr')
    print(f'shape of vectors with tfidf, deps, quartile, note length, and {feature}:')
    print(train_feat.shape)
    model = svm.SVC(kernel = 'linear')
    model.fit(train_feat,gold_labels)
    save_model_and_vec(model,vec,feature)
    dev_conll = pd.read_csv('../data/dev_with_features.conll',sep='\t')
    dev_conll = deps_feature(dev_conll)
    sentences = dev_conll.groupby(['pad_sent_id'])
    data = []
    for _, sentence in sentences:
        quart_note = sentence['quart_note'].tolist()[0]
        note_len = sentence['note_len'].tolist()[0]
        featdict = {'deps':sentence['deps'],'quart_note':quart_note,'note_length': note_len, feature: sentence[feature]}
        data.append(featdict)
    dev_vectors = vec.transform(data)
    dev_df = pd.read_csv('../data/dev_full_notes.tsv',sep = '\t')
    dev_X = tfidf.transform(dev_df['text'])
    dev_feat = hstack([dev_X,dev_vectors],format='csr')
    gold = dev_df['rel_time']
    predictions = model.predict(dev_feat)
    dev_df['predictions'] = predictions
    dev_df.to_csv(f'../data/dev_svm_{feature}_tfidf_predictions.tsv',sep='\t',index=False)
    get_evaluation_metrics(f'../data/dev_svm_{feature}_tfidf_predictions.tsv','svm','dev',f'{feature}_tfidf')


# first feature is deps

# for n in ['UPOS','XPOS','VERB','quart_note','note_len','tense','temp_ner']:
#     select_2nd_feat(n)

# second feature is quart_note

# for n in ['UPOS','XPOS','VERB','note_len','tense','temp_ner']:
#     select_3rd_feat(n)

# third feature is note_len

# for n in ['UPOS','XPOS','VERB','tense','temp_ner']:
#     select_4th_feat(n)

### rbf ###

# first feat

for n in ['deps','UPOS','XPOS','VERB','quart_note','note_len','tense','temp_ner']:
    select_1st_feat(n)

# 0 #
# df = pd.read_csv('../data/train_full_notes.tsv',sep='\t')
# vec = TfidfVectorizer()
# X = vec.fit_transform(df['text'])
# print(f'shape of vectors with 0 features:')
# print(X.shape)
# gold_labels = df['rel_time']
# model = svm.SVC(kernel = 'rbf')
# print('Creating SVM classifier...')
# model.fit(X,gold_labels)
# save_model_and_vec(model,vec,'0')
# dev_df = pd.read_csv('../data/dev_full_notes.tsv',sep = '\t')
# dev_feat = vec.transform(dev_df['text'])
# gold = dev_df['rel_time']
# predictions = model.predict(dev_feat)
# dev_df['predictions'] = predictions
# dev_df.to_csv('../data/dev_svm_0_predictions.tsv',sep='\t',index=False)
# get_evaluation_metrics('../data/dev_svm_0_predictions.tsv','svm','dev','0')

# for n in ['1','2','3','4','5','6','7','8']:
#     model, vec = create_svm_clf('../data/train_with_features.conll',n)
#     classify_data('../data/dev_with_features.conll',n,'../data/dev_full_notes.tsv', model,vec)
#     get_evaluation_metrics(f'../data/dev_svm_{n}_predictions.tsv','svm','dev',n)

