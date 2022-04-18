# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os
import os.path as op
import json
import cv2
import base64
import yaml
from tqdm import tqdm

from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
from maskrcnn_benchmark.structures.tsv_file_ops import generate_linelist_file
from maskrcnn_benchmark.structures.tsv_file_ops import generate_hw_file
from maskrcnn_benchmark.structures.tsv_file import TSVFile
from maskrcnn_benchmark.data.datasets.utils.image_ops import img_from_base64

DATA_DIR = '/home/ubuntu/s3-drive/data'
data_path = op.join(DATA_DIR, 'wiki_split.json')
img_dir = op.join(DATA_DIR, 'wikicommons', 'resized')
output_dir = op.join(DATA_DIR, 'vinvl_data')

with open(data_path, 'r') as j:
    data = json.load(j)['images']

splits = ['train', 'val', 'test']

rows_img = {split: [] for split in splits}
rows_hw = {split: [] for split in splits}
# rows_label = []

for example in tqdm(data):
    example_split = example['split']

    img_p = example['filename']
    img_key = img_p.split('.')[0]
    img_path = op.join(img_dir, img_p)
    img = cv2.imread(img_path)
    img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])

    row = [img_key, img_encoded_str]
    rows_img[example_split].append(row)

    height = img.shape[0]
    width = img.shape[1]
    row_hw = [img_key, json.dumps([{"height":height, "width":width}])]
    rows_hw[example_split].append(row_hw)

    # # Here is just a toy example of labels.
    # # The real labels can be generated from the annotation files
    # # given by each dataset. The label is a list of dictionary 
    # # where each box with at least "rect" (xyxy mode) and "class"
    # # fields. It can have any other fields given by the dataset.
    # labels = []
    # labels.append({"rect": [1, 1, 30, 40], "class": "Dog"})
    # labels.append({"rect": [2, 3, 100, 100], "class": "Cat"})

    # row_label = [img_key, json.dumps(labels)]
    # rows_label.append(row_label)

# Create .yaml file for connecting .tsv files
yaml_dicts = {split: {attr: f'{split}.{attr}.tsv' for attr in ['img', 'hw']} # ['img', 'hw', 'label', 'linelist]
                for split in splits}

for split in splits:
    img_file = op.join(output_dir, yaml_dicts[split]['img'])
    hw_file = op.join(output_dir, yaml_dicts[split]['hw'])
    tsv_writer(rows_img[split], img_file)
    tsv_writer(rows_hw[split], hw_file)
    # label_file = "tools/mini_tsv/data/train.label.tsv"
    # linelist_file = "tools/mini_tsv/data/train.linelist.tsv"
    # tsv_writer(rows_label, label_file)
    # generate linelist file
    # generate_linelist_file(label_file, save_file=linelist_file)
    with open(op.join(output_dir, f'{split}.yaml'), 'w') as file:
            yaml.dump(yaml_dicts[split], file, default_flow_style=False)


# # To access a tsv file:
# # 1) Use tsv_reader to read dataset in given order
# rows = tsv_reader("tools/mini_tsv/data/train.tsv")
# rows_label = tsv_reader("tools/mini_tsv/data/train.label.tsv")
# for row, row_label in zip(rows, rows_label):
#     img_key = row[0]
#     labels = json.loads(row_label[1])
#     img = img_from_base64(row[1])

# # 2) use TSVFile to access dataset at any given row.
# tsv = TSVFile("tools/mini_tsv/data/train.tsv")
# row = tsv.seek(1) # to access the second row 
# img_key = row[0]
# img = img_from_base64(row[1])



