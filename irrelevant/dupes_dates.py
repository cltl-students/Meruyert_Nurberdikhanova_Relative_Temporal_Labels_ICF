import pandas as pd

train_dupes = pd.read_csv('train_dupes.tsv',sep='\t')
train_dupes['Notitiedatum'] = pd.to_datetime(train_dupes['Notitiedatum'],format='%Y/%m/%d')
print(f"train duplicates were written between: {train_dupes['Notitiedatum'].dt.date.min()} and {train_dupes['Notitiedatum'].dt.date.max()}")

dev_dupes = pd.read_csv('dev_dupes.tsv',sep='\t')
dev_dupes['Notitiedatum'] = pd.to_datetime(dev_dupes['Notitiedatum'],format='%Y/%m/%d')
print(f"dev duplicates were written between: {dev_dupes['Notitiedatum'].dt.date.min()} and {dev_dupes['Notitiedatum'].dt.date.max()}")

notes = train_dupes['NotitieID'].unique()
print(f'train duplicates have {len(notes)} unique notes')
notes_d = dev_dupes['NotitieID'].unique()
print(f'dev duplicates have {len(notes_d)} unique notes')
