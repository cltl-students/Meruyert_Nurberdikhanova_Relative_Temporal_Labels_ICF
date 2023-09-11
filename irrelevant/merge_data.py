import pandas as pd 

### merge all background and target data into single files to inspect
### then, split by 80/10/10 principles (documented in notion)
### pull 'current' subset of data from Jenia's training


def merge_data(rel_time):


    train = pd.read_csv(f'../data/train_{rel_time}.tsv',sep='\t')
    dev = pd.read_csv(f'../data/dev_{rel_time}.tsv',sep='\t')
    test = pd.read_csv(f'../data/test_{rel_time}.tsv',sep='\t')
    merged_df = pd.concat([train,dev,test],ignore_index=True)
    merged_df.to_csv(f'../data/{rel_time}_merged.tsv', sep='\t', index=False)
    print(f'{rel_time} sets are merged! {len(merged_df)} sentences belong to {rel_time}.')

# # read in paths
train_background = pd.read_csv(train_background_p,sep='\t')
dev_background = pd.read_csv(dev_background_p,sep='\t')
test_background = pd.read_csv(test_background_p,sep='\t')
train_target = pd.read_csv(train_target_p,sep='\t')
dev_target = pd.read_csv(dev_target_p,sep='\t')
test_target = pd.read_csv(test_target_p,sep='\t')

# # merge
background = pd.concat([train_background,dev_background,test_background],ignore_index=True)
target = pd.concat([train_target,dev_target,test_target],ignore_index=True)

# # save merged 
background.to_csv('../data/background_merged.tsv', sep='\t', index=False)
target.to_csv('../data/target_merged.tsv', sep='\t', index=False)

# same for current
train_cur_p = '../data/train_current.tsv'
dev_cur_p = '../data/dev_current.tsv'
test_cur_p = '../data/test_current.tsv'
train_current = pd.read_csv(train_cur_p, sep='\t')
dev_current = pd.read_csv(dev_cur_p, sep='\t')
test_current = pd.read_csv(test_cur_p, sep='\t')
current = pd.concat([train_current,dev_current,test_current], ignore_index=True)
current.to_csv('../data/current_merged.tsv',sep='\t',index=False)
