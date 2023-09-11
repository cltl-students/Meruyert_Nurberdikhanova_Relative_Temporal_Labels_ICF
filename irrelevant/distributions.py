import pandas as pd
def make_df(split: list, table_type: str):
    df_list = []
    for data in split:
        types = data[table_type].unique()
        type_dict = {}
        for i in types:
            num = len(data.loc[data['batch'] == i])
            type_dict[i] = num
            df_list.append(type_dict)
    df = pd.DataFrame(df_list, index=['train','dev','test'])
    df.fillna(value=0,inplace=True)
    df = df.reindex(sorted(df.columns),axis=1,copy=False)
    df.to_csv(f'descriptives/{table_type}.tsv',sep='\t')
    df.style.to_latex(f'descriptives/{table_type}_latex.txt')

def get_descriptives(table_type: str):
    trainpath = '../data/train.tsv'
    devpath = '../data/dev.tsv'
    testpath = '../data/dev.tsv'
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
        df_natural = pd.DataFrame(df_list,index=['train','dev','test'])
        df_natural['past_%'] = df_natural['past']/df_natural['Jenia']
        df_natural['past_%'].mul(100).round(decimals=2)
        df_natural['now_%'] = df_natural['now']/df_natural['Jenia']
        df_natural['now_%'].mul(100).round(decimals=2)
        df_natural['future_%'] = df_natural['future']/df_natural['Jenia']
        df_natural['future_%'].mul(100).round(decimals=2)
        df_natural.insert(2,'past_%',df.pop('past_%'))
        df_natural.insert(4,'now_%',df.pop('now_%'))
        df_natural.to_csv(f'descriptives/{table_type}.tsv',sep='\t')
        df_natural.style.to_latex(f'descriptives/{table_type}_latex.txt')
    elif table_type == 'note_patient':
        df_list = []
        for data in split:
            df_dict = {}
            df_dict['notes'] = len(data.NotitieID.unique())
            df_dict['patients'] = len(data.MDN.unique())
            df_list.append(df_dict)
        df = pd.DataFrame(df_list,index=['train','dev','test'])
        df.to_csv(f'descriptives/{table_type}.tsv',sep='\t')
        df.style.to_latex(f'descriptives/{table_type}_latex.txt')
    else:
        raise ValueError('Invalid table_type! Please choose one of the existing options.')
    
    print(f'{table_type} descriptives are saved!')






## to check distributions in the main data split ##
train_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/train.pkl'
test_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/test.pkl'
dev_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/dev.pkl'
pd.set_option('display.max_columns', None)
train_sept = pd.read_pickle(train_path)
dev_sept = pd.read_pickle(dev_path)
test_sept = pd.read_pickle(test_path)
print('train september')
print(len(train_sept.NotitieID.unique()))
# check batches/weeks
train_batches = train_sept['batch'].unique()
for batch in train_batches:
    print(f"{batch} => {len(train_sept.loc[train_sept['batch'] == batch])}")
# check years
train_years = train_sept['year'].unique()
for year in train_years:
    print(f"{year} =>  {len(train_sept.loc[train_sept['year'] == year])}")
print(len(train_sept.MDN.unique()))
train_inst = train_sept.institution.unique()
for inst in train_inst:
    print(f"{inst} => {len(train_sept.loc[train_sept['institution']==inst])}")
print('dev september')
print(len(dev_sept.NotitieID.unique()))
print(len(train_sept.NotitieID.unique()))
dev_batches = dev_sept['batch'].unique()
for batch in dev_batches:
    print(f"{batch} => {len(dev_sept.loc[dev_sept['batch'] == batch])}")
print(len(dev_sept.MDN.unique()))
dev_inst = dev_sept.institution.unique()
for inst in dev_inst:
    print(f"{inst} => {len(dev_sept.loc[dev_sept['institution']==inst])}")
print('test september')
print(len(test_sept.NotitieID.unique()))
test_batches = test_sept['batch'].unique()
for batch in test_batches:
    print(f"{batch} => {len(test_sept.loc[test_sept['batch'] == batch])}")
test_years = test_sept['year'].unique()
for year in test_years:
    print(f"{year} =>  {len(test_sept.loc[test_sept['year'] == year])}")
print(len(test_sept.MDN.unique()))
test_inst = test_sept.institution.unique()
for inst in test_inst:
    print(f"{inst} => {len(test_sept.loc[test_sept['institution']==inst])}")

### check distributions for current subgroup ###
train_cur_p = '../data/train_current.tsv'
dev_cur_p = '../data/dev_current.tsv'
test_cur_p = '../data/test_current.tsv'
train_current = pd.read_csv(train_cur_p, sep='\t')
dev_current = pd.read_csv(dev_cur_p, sep='\t')
test_current = pd.read_csv(test_cur_p, sep='\t')

print('train current')
print(f'note IDs {len(train_current.NotitieID.unique())}')
train_batches = train_current['batch'].unique()
for batch in train_batches:
    print(f"{batch} => {len(train_current.loc[train_current['batch'] == batch])}")
train_b_years = train_current['year'].unique()
for year in train_b_years:
    print(f"{year} =>  {len(train_current.loc[train_current['year'] == year])}")
print(f'MDNs {len(train_current.MDN.unique())}')
train_b_inst = train_current.institution.unique()
for inst in train_b_inst:
    print(f"{inst} => {len(train_current.loc[train_current['institution']==inst])}")

print('dev current')
print(f'note IDs {len(dev_current.NotitieID.unique())}')
train_batches = dev_current['batch'].unique()
for batch in train_batches:
    print(f"{batch} => {len(dev_current.loc[dev_current['batch'] == batch])}")
train_b_years = dev_current['year'].unique()
for year in train_b_years:
    print(f"{year} =>  {len(dev_current.loc[dev_current['year'] == year])}")
print(f'MDNs {len(dev_current.MDN.unique())}')
train_b_inst = dev_current.institution.unique()
for inst in train_b_inst:
    print(f"{inst} => {len(dev_current.loc[dev_current['institution']==inst])}")

print('test current')
print(f'note IDs {len(test_current.NotitieID.unique())}')
train_batches = test_current['batch'].unique()
for batch in train_batches:
    print(f"{batch} => {len(test_current.loc[test_current['batch'] == batch])}")
train_b_years = test_current['year'].unique()
for year in train_b_years:
    print(f"{year} =>  {len(test_current.loc[test_current['year'] == year])}")
print(f'MDNs {len(test_current.MDN.unique())}')
train_b_inst = test_current.institution.unique()
for inst in train_b_inst:
    print(f"{inst} => {len(test_current.loc[test_current['institution']==inst])}")

## to check distributions in the background and target subgroups ##
train_background_p='../data/train_background.tsv'
dev_background_p='../data/dev_background.tsv'
test_background_p='../data/test_background.tsv'
train_target_p='../data/train_target.tsv'
dev_target_p='../data/dev_target.tsv'
test_target_p='../data/test_target.tsv'
# train_background = pd.read_csv(train_background_p,sep='\t')
# dev_background = pd.read_csv(dev_background_p,sep='\t')
# test_background = pd.read_csv(test_background_p,sep='\t')
# train_target = pd.read_csv(train_target_p,sep='\t')
# dev_target = pd.read_csv(dev_target_p,sep='\t')
# test_target = pd.read_csv(test_target_p,sep='\t')
# # check batches/weeks for background
# print('background train')
# print(len(train_background.NotitieID.unique()))
# train_batches = train_background['batch'].unique()
# for batch in train_batches:
#     print(f"{batch} => {len(train_background.loc[train_background['batch'] == batch])}")
# train_b_years = train_background['year'].unique()
# for year in train_b_years:
#     print(f"{year} =>  {len(train_background.loc[train_background['year'] == year])}")
# print(len(train_background.MDN.unique()))
# train_b_inst = train_background.institution.unique()
# for inst in train_b_inst:
#     print(f"{inst} => {len(train_background.loc[train_background['institution']==inst])}")

# print('background dev')
# print(len(dev_background.NotitieID.unique()))
# dev_batches = dev_background['batch'].unique()
# for batch in dev_batches:
#     print(f"{batch} => {len(dev_background.loc[dev_background['batch'] == batch])}")
# dev_b_years = dev_background['year'].unique()
# for year in dev_b_years:
#     print(f"{year} =>  {len(dev_background.loc[dev_background['year'] == year])}")
# print(len(dev_background.MDN.unique()))
# dev_b_inst = dev_background.institution.unique()
# for inst in dev_b_inst:
#     print(f"{inst} => {len(dev_background.loc[dev_background['institution']==inst])}")

# print('background test')
# print(len(test_background.NotitieID.unique()))
# test_batches = test_background['batch'].unique()
# for batch in test_batches:
#     print(f"{batch} => {len(test_background.loc[test_background['batch'] == batch])}")
# test_b_years = test_background['year'].unique()
# for year in test_b_years:
#     print(f"{year} =>  {len(test_background.loc[test_background['year'] == year])}")
# print(len(test_background.MDN.unique()))
# test_b_inst = test_background.institution.unique()
# for inst in test_b_inst:
#     print(f"{inst} => {len(test_background.loc[test_background['institution']==inst])}")

# # # check batches/weeks for target
# print('target train')
# print(len(train_target.NotitieID.unique()))
# train_batches = train_target['batch'].unique()
# for batch in train_batches:
#     print(f"{batch} => {len(train_target.loc[train_target['batch'] == batch])}")
# train_t_years = train_target['year'].unique()
# for year in train_t_years:
#     print(f"{year} =>  {len(train_target.loc[train_target['year'] == year])}")
# print(len(train_target.MDN.unique()))
# train_t_inst = train_target.institution.unique()
# for inst in train_t_inst:
#     print(f"{inst} => {len(train_target.loc[train_target['institution']==inst])}")

# print('target dev')
# print(len(dev_target.NotitieID.unique()))
# dev_batches = dev_target['batch'].unique()
# for batch in dev_batches:
#     print(f"{batch} => {len(dev_target.loc[dev_target['batch'] == batch])}")
# dev_t_years = dev_target['year'].unique()
# for year in dev_t_years:
#     print(f"{year} =>  {len(dev_target.loc[dev_target['year'] == year])}")
# print(len(dev_target.MDN.unique()))
# dev_t_inst = dev_target.institution.unique()
# for inst in dev_t_inst:
#     print(f"{inst} => {len(dev_target.loc[dev_target['institution']==inst])}")

# print('target test')
# print(len(test_target.NotitieID.unique()))
# test_batches = test_target['batch'].unique()
# for batch in test_batches:
#     print(f"{batch} => {len(test_target.loc[test_target['batch'] == batch])}")
# test_t_years = test_target['year'].unique()
# for year in test_t_years:
#     print(f"{year} =>  {len(test_target.loc[test_target['year'] == year])}")
# print(len(test_target.MDN.unique()))
# test_t_inst = test_target.institution.unique()
# for inst in test_t_inst:
#     print(f"{inst} => {len(test_target.loc[test_target['institution']==inst])}")
