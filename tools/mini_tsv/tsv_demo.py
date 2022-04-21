# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os
import argparse
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

def main():
    output_dir = op.join(DATA_DIR, args.output_dir)
    with open(data_path, 'r') as j:
        data = json.load(j)['images']

    splits = ['train', 'val', 'test']

    rows_img = {split: [] for split in splits}
    rows_hw = {split: [] for split in splits}
    tgt_seqs = {split: [] for split in splits}
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

        example_tgt_seqs = {
            tgt_type: example[tgt_type]['raw'] for tgt_type in ['description', 'caption', 'context']
        }
        tgt_seqs[example_split].append({'image_id': img_key, **example_tgt_seqs})

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
    yaml_dicts = {split: {attr: f'{split}.{attr}.tsv' for attr in ['img', 'hw', 'linelist']} # ['img', 'hw', 'label', 'linelist]
                    for split in splits}
    for split in splits:
        yaml_dicts[split]['caption'] = f'{split}_caption.json'
        yaml_dicts[split]['labelmap'] = 'VG-SGG-dicts-vgoi6-clipped.json'

    for split in splits:
        img_file = op.join(output_dir, yaml_dicts[split]['img'])
        hw_file = op.join(output_dir, yaml_dicts[split]['hw'])
        tsv_writer(rows_img[split], img_file)
        tsv_writer(rows_hw[split], hw_file)
        with open(op.join(output_dir, f'{split}_caption.json'), 'w') as f:
            json.dump(tgt_seqs[split], f, indent=4)
        # label_file = "tools/mini_tsv/data/train.label.tsv"
        # tsv_writer(rows_label, label_file)

        # generate linelist file
        linelist_file = op.join(output_dir, yaml_dicts[split]['linelist'])
        generate_linelist_file(hw_file, save_file=linelist_file)
        with open(op.join(output_dir, f'{split}.yaml'), 'w') as file:
            yaml.dump(yaml_dicts[split], file, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    main()

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



