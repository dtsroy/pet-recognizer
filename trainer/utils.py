import json
import pandas as pd


def get_classes(fp='../data/data.parquet', jp='../data/classes.json', dp='../data/pro.parquet'):
    df = pd.read_parquet(fp)
    labels = list(set(list(df['label'])))
    labels = dict(zip(labels, range(len(labels))))
    # print(labels)

    new_df = pd.DataFrame()

    new_df['image'] = df['image']
    new_df.insert(1, 'label', -1)
    # print(new_df)
    for i in range(len(df['label'])):
        new_df.loc[i, 'label'] = labels[df.loc[i, 'label']]

    with open(jp, 'w+') as f:
        json.dump(labels, f, indent=True)
    new_df.to_parquet(dp)

