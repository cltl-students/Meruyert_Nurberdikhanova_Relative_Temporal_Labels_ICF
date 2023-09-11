import pandas as pd 

train = pd.read_csv('../data/train.tsv',sep='\t')
print(train)
dev = pd.read_csv('../data/dev.tsv',sep='\t')
print(dev)
test = pd.read_csv('../data/test.tsv',sep='\t')
print(test)