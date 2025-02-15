import json
import pandas as pd
import model
import torch
import onnx
import io
from PIL import Image


def get_classes(fp='../data/data.parquet', jp='../data/names.json', dp='../data/pro.parquet'):
    df = pd.read_parquet(fp)
    # return
    labels = list(set(list(df['label'])))
    labels.sort()
    print(labels[10])
    with open(jp, 'w+') as f:
        json.dump(labels, f, indent=True)
    # labels = dict(zip(labels, range(len(labels))))
    # print(labels)

    new_df = pd.DataFrame()

    new_df['image'] = df['image']
    new_df.insert(1, 'label', -1)
    # print(new_df)
    for i in range(len(df['label'])):
        new_df.loc[i, 'label'] = labels.index(df.loc[i, 'label'])


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
    print(df.loc[_id, 'label'])
    with open(outp, 'wb+') as f:
        f.write(b)


def get_image(_id, fp, outp):
    df = pd.read_parquet(fp)
    b = df.loc[_id, 'image']['bytes']
    print(df.loc[_id, 'label'])
    bio = io.BytesIO(b)
    image = Image.open(bio)
    image.save(outp)


# export_as_onnx('../m4/m_ep12.pth', '../models/PetRecognizerM4.onnx')
# get_image_bytes(10, '../data/pro.parquet', 'img0.bin')
# get_image_bytes(10, '../data/data.parquet', 'img0.bin_')
# get_classes()
get_image(10, '../data/pro.parquet', 'a1.png')
get_image(10, '../data/data.parquet', 'a2.png')
