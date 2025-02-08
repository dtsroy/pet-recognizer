import json
import pandas as pd
import model
import torch
import onnx


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


def export_as_onnx(pth_p, out_p, using_model='M4'):
    if using_model == 'M4':
        m = model.M4()
    elif using_model == 'M3':
        m = model.M3()
    else:
        m = model.PetRecognizer()
    m.load_state_dict(torch.load(pth_p, map_location='cpu'))
    torch.onnx.export(
        model=m,
        args=(torch.zeros(1, 3, 224, 224),),
        f=out_p,
        input_names=['input0'],
        output_names=['output0'],
        opset_version=16,
    )


def get_image_bytes(_id, fp, outp):
    df = pd.read_parquet(fp)
    b = df.loc[_id, 'image']['bytes']
    with open(outp, 'wb+') as f:
        f.write(b)


get_image_bytes(1, '../data/pro.parquet', '0.bin')


