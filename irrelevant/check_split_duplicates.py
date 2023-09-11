import pandas as pd


train = pd.read_csv('../data/train.tsv',sep='\t')
train.reset_index(drop=True,inplace=True)
dev = pd.read_csv('../data/dev.tsv',sep='\t')
test = pd.read_csv('../data/test.tsv',sep='\t')

print(f'train: {len(train)} sentences, {len(train.pad_sen_id.unique())}  unique sentences')
print(f'dev: {len(dev)} sentences, {len(dev.pad_sen_id.unique())}  unique sentences')
print(f'test: {len(test)} sentences, {len(test.pad_sen_id.unique())}  unique sentences')


# maybe dive deeper
train_background_p='../data/train_background.tsv'
dev_background_p='../data/dev_background.tsv'
test_background_p='../data/test_background.tsv'
train_target_p='../data/train_target.tsv'
dev_target_p='../data/dev_target.tsv'
test_target_p='../data/test_target.tsv'
train_background = pd.read_csv(train_background_p,sep='\t')
dev_background = pd.read_csv(dev_background_p,sep='\t')
test_background = pd.read_csv(test_background_p,sep='\t')
train_target = pd.read_csv(train_target_p,sep='\t')
dev_target = pd.read_csv(dev_target_p,sep='\t')
test_target = pd.read_csv(test_target_p,sep='\t')
train_cur_p = '../data/train_current.tsv'
dev_cur_p = '../data/dev_current.tsv'
test_cur_p = '../data/test_current.tsv'
train_current = pd.read_csv(train_cur_p, sep='\t')
dev_current = pd.read_csv(dev_cur_p, sep='\t')
test_current = pd.read_csv(test_cur_p, sep='\t')
print(f'train back: {len(train_background)} sentences, {len(train_background.pad_sen_id.unique())}  unique sentences')
print(f'dev back: {len(dev_background)} sentences, {len(dev_background.pad_sen_id.unique())}  unique sentences')
print(f'test back: {len(test_background)} sentences, {len(test_background.pad_sen_id.unique())}  unique sentences')
print(f'train tar: {len(train_target)} sentences, {len(train_target.pad_sen_id.unique())}  unique sentences')
print(f'dev tar: {len(dev_target)} sentences, {len(dev_target.pad_sen_id.unique())}  unique sentences')
print(f'test tar: {len(test_target)} sentences, {len(test_target.pad_sen_id.unique())}  unique sentences')
print(f'train cur: {len(train_current)} sentences, {len(train_current.pad_sen_id.unique())}  unique sentences')
print(f'dev cur: {len(dev_current)} sentences, {len(dev_current.pad_sen_id.unique())}  unique sentences')
print(f'test cur: {len(test_current)} sentences, {len(test_current.pad_sen_id.unique())}  unique sentences')

background_p = '../data/background_merged.tsv'
target_p = '../data/target_merged.tsv'
current_p = '../data/current_merged.tsv'
background = pd.read_csv(background_p,sep='\t')
target = pd.read_csv(target_p, sep='\t')
current = pd.read_csv(current_p,sep='\t')
print(f'current: {len(current)} sentences, {len(current.pad_sen_id.unique())}  unique sentences')
print(f'background: {len(background)} sentences, {len(background.pad_sen_id.unique())}  unique sentences')
print(f'target: {len(target)} sentences, {len(target.pad_sen_id.unique())}  unique sentences')

train_b = pd.read_csv('../data/for_sampling/train_b.tsv',sep='\t')
dev_b = pd.read_csv('../data/for_sampling/dev_b.tsv',sep='\t')
test_b = pd.read_csv('../data/for_sampling/test_b.tsv',sep='\t')
train_t = pd.read_csv('../data/for_sampling/train_t.tsv',sep='\t')
dev_t = pd.read_csv('../data/for_sampling/dev_t.tsv',sep='\t')
test_t = pd.read_csv('../data/for_sampling/test_t.tsv',sep='\t')
train_c = pd.read_csv('../data/for_sampling/train_c.tsv',sep='\t')
dev_c = pd.read_csv('../data/for_sampling/dev_c.tsv',sep='\t')
test_c =pd.read_csv('../data/for_sampling/test_c.tsv',sep='\t')

print(f'train back: {len(train_b)} sentences, {len(train_b.pad_sen_id.unique())}  unique sentences')
print(f'dev back: {len(dev_b)} sentences, {len(dev_b.pad_sen_id.unique())}  unique sentences')
print(f'test back: {len(test_b)} sentences, {len(test_b.pad_sen_id.unique())}  unique sentences')
print(f'train tar: {len(train_t)} sentences, {len(train_t.pad_sen_id.unique())}  unique sentences')
print(f'dev tar: {len(dev_t)} sentences, {len(dev_t.pad_sen_id.unique())}  unique sentences')
print(f'test tar: {len(test_t)} sentences, {len(test_t.pad_sen_id.unique())}  unique sentences')
print(f'train cur: {len(train_c)} sentences, {len(train_c.pad_sen_id.unique())}  unique sentences')
print(f'dev cur: {len(dev_c)} sentences, {len(dev_c.pad_sen_id.unique())}  unique sentences')
print(f'test cur: {len(test_c)} sentences, {len(test_c.pad_sen_id.unique())}  unique sentences')
