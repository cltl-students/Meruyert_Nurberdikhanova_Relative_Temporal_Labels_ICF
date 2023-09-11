import pandas as pd
# checking data from classifier for categories only
train_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/train.pkl'
test_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/test.pkl'
dev_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/dev.pkl'
pd.set_option('display.max_columns', None)
# train_sept = pd.read_pickle(train_path)
# print('train september')
# print(train_sept)
# dev_sept = pd.read_pickle(dev_path)
# print('dev september')
# print(dev_sept)
# test_sept = pd.read_pickle(test_path)
# print('test september')
# print(test_sept)
# checking pilot data - covid and non covid
pilot_covid_trainpath = '/mnt/data/A-Proof/data2/Covid_data_11nov/a-proof-exp/traindata_covidbatch.pkl'
pilot_covid_testpath =  '/mnt/data/A-Proof/data2/Covid_data_11nov/a-proof-exp/testdata_covidbatch.pkl'
pilot_non_trainpath = '/mnt/data/A-Proof/data2/Non_covid_data_15oct/from_inception_tsv/annotated_df_Batch1_pilot_parsed.pkl'
# pandas == 1.5.3
#pilot_covid_train = pd.read_pickle(pilot_covid_trainpath)
#print('pilot cov train 11nov')
#print(pilot_covid_train)
# pilot_covid_test = pd.read_pickle(pilot_covid_testpath)
# print('pilot cov test 11nov')
# print(pilot_covid_test)
# pilot_non_train = pd.read_pickle((pilot_non_trainpath))
# print('pilot non cov train 15oct')
# print(pilot_non_train)

# checking pilot in september folder
pilot_covid_sept = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/pilot_cov_parsed.pkl'
pilot_non_sept = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/pilot_noncov_parsed.pkl'
pilot_covid = pd.read_pickle(pilot_covid_sept)
print('pilot covid september')
print(pilot_covid)
pilot_non = pd.read_pickle(pilot_non_sept)
print('pilot noncov september')
print(pilot_non)