import pandas as pd

# inspect test_original.pkl in expr_sept folder
# what differs from test.pkl

test_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/test.pkl'
test_original_path = '/mnt/data/A-Proof/data2/a-proof-zonmw/data/expr_sept/clf_domains/test_original.pkl'
test = pd.read_pickle(test_path)
test_original = pd.read_pickle(test_original_path)
pd.set_option('display.max_columns', None)
print('test pkl:')
print(test)
print('test original pkl')
print(test_original)

### conclusion ###
# same lengths 22082 lines
# difference in columns: test.pkl has one more column 'original_labels'
# must be reannotation
