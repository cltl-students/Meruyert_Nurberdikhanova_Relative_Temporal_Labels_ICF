import pandas as pd
from simpletransformers.classification import ClassificationModel
# model = ClassificationModel(
#     "roberta",
#     "outputs/final_model",
#     use_cuda=False,
#     num_labels=3
# )

# predictions, _ = model.predict(['afbouwen indien mogelijk'])
# print(predictions)

dev_df = pd.read_csv('../data/dev.tsv',sep='\t')
print(dev_df.dtypes)
print(dev_df['text'])
txt = dev_df['text'].tolist()
