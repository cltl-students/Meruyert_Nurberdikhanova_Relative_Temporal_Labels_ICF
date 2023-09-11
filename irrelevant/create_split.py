import pandas as pd 

background_p = '../data/background_merged.tsv'
target_p = '../data/target_merged.tsv'
current_p = '../data/current_merged.tsv'

background = pd.read_csv(background_p,sep='\t')
target = pd.read_csv(target_p, sep='\t')
current = pd.read_csv(current_p,sep='\t')

# add a column for temporal label
background['rel_time'] = 'past'
target['rel_time'] = 'future'
current['rel_time'] = 'now'

# first extract 80/10/10 split for each subgroup
def split_subgroup(rel_time):
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

train_b, dev_b, test_b = split_subgroup(background)
train_t, dev_t, test_t = split_subgroup(target)
train_c, dev_c, test_c = split_subgroup(current)
# NOTE: recheck the indices (which is why saving this shit here)
train_b.to_csv('../data/for_sampling/train_b.tsv',sep='\t')
dev_b.to_csv('../data/for_sampling/dev_b.tsv',sep='\t')
test_b.to_csv('../data/for_sampling/test_b.tsv',sep='\t')
train_t.to_csv('../data/for_sampling/train_t.tsv',sep='\t')
dev_t.to_csv('../data/for_sampling/dev_t.tsv',sep='\t')
test_t.to_csv('../data/for_sampling/test_t.tsv',sep='\t')
train_c.to_csv('../data/for_sampling/train_c.tsv',sep='\t')
dev_c.to_csv('../data/for_sampling/dev_c.tsv',sep='\t')
test_c.to_csv('../data/for_sampling/test_c.tsv',sep='\t')

# third merge splits by subgroups and shuffle them again
def merge_shuffle_split(df_b, df_t, df_c, data_type):
    df = pd.concat([df_b,df_t, df_c])
    df = df.sample(frac=1)
    df.drop(['annotator','background_sent','target_sent','text_raw','original_labels','index','pred_domains_eb_ap_mod1','pred_domains_eb_ap_mod2'],axis=1, inplace=True)
    df.to_csv(f'../data/{data_type}.tsv',sep='\t',index=False)
    print(f'Created {data_type} set! {len(df)} sentences long.')

train = merge_shuffle_split(train_b,train_t,train_c)
dev = merge_shuffle_split(dev_b,dev_t,dev_c)
test = merge_shuffle_split(test_b,test_t,test_c)

# drop unnecessary columns
train.drop(['annotator','background_sent','target_sent','text_raw','original_labels','index','pred_domains_eb_ap_mod1','pred_domains_eb_ap_mod2'],axis=1, inplace=True)
dev.drop(['annotator','background_sent','target_sent','text_raw','original_labels','index','pred_domains_eb_ap_mod1','pred_domains_eb_ap_mod2'],axis=1, inplace=True)
test.drop(['annotator','background_sent','target_sent','text_raw','original_labels','index','pred_domains_eb_ap_mod1','pred_domains_eb_ap_mod2'],axis=1, inplace=True)

# save the splits
#check the indices again pls
train.to_csv('../data/train.tsv',sep='\t',index=False)
dev.to_csv('../data/dev.tsv',sep='\t', index=False)
test.to_csv('../data/test.tsv',sep='\t', index=False)

