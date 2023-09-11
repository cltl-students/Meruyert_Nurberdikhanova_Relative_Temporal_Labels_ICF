from build_svm import *
import pandas as pd 
for n in ['0','1','2','3','4','5','6','7','8']:
    data_0, df = get_features('../data/train_with_features.conll',n)
    vec, vectors = vectorise_features(data_0)
    print(f'For {n} number of features:')
    print(vectors.shape)
    # sent_1 = vec.transform(data_0[0])
    # print(f'shape of first sentence in training: {sent_1.shape}}')
# print('feature names of first sentence:')
# print(sent_1.get_feature_names_out())