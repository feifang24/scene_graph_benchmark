import json
import numpy as np
import pandas as pd
import base64
import os.path as op
import argparse
import ast
import yaml
from maskrcnn_benchmark.data.datasets.utils.load_files import load_from_yaml_file

from maskrcnn_benchmark.structures.tsv_file_ops import tsv_writer

DATA_DIR = '/home/ubuntu/s3-drive/data'

def generate_features_for_split(data_dir: str, split: str):
    # Load height and width of every image
    hw_df = pd.read_csv(op.join(data_dir, f'{split}.hw.tsv'), sep='\t', header=None, converters={1:ast.literal_eval}, index_col=0)

    # Helper functions
    def generate_positional_features(rect,h,w):
        mask = np.array([w,h,w,h],dtype=np.float32)
        rect = np.clip(rect/mask,0,1)
        res = np.hstack((rect,[rect[3]-rect[1], rect[2]-rect[0]]))
        return res.astype(np.float32)

    def generate_features(x):
        #image_id, object data list of dictionary, number of detected objects
        idx, data,num_boxes = x[0],x[1],len(x[1])
        # read image height, width, and initialize array of features
        h,w,features_arr = hw_df.loc[idx,1][0]['height'],hw_df.loc[idx,1][0]['width'],[]

        # for every detected object in img
        for i in range(num_boxes):
            # read image region feature vector
            features = np.frombuffer(base64.b64decode(data[i].pop('feature')),np.float32)
            # add 6 additional dimensions
            pos_feat = generate_positional_features(data[i]['rect'],h,w)
            # stack feature vector with 6 additional dimensions
            x = np.hstack((features,pos_feat))
            features_arr.append(x.astype(np.float32))
            
        features = np.vstack(tuple(features_arr))
        features = base64.b64encode(features).decode("utf-8")
        return {"num_boxes": num_boxes, "features": features}

    def generate_labels(x):
        data = x[1]
        res = [{"class":el.pop('class'), "conf":el.pop('conf'), "rect": el['rect']} for el in data]
        return res

    # Directory of out predictions.tsv (bbox_id, class, conf, feature, rect)
    sg_tsv = op.join(data_dir, 'inference', split if len(args.splits) > 1 else '', 'vinvl_vg_x152c4/predictions.tsv')
    df = pd.read_csv(sg_tsv, sep='\t', header = None, converters={1:json.loads}) #converters={1:ast.literal_eval})
    df[1] = df[1].apply(lambda x: x['objects'])

    # Generate features from predictions.tsv
    df['feature'] = df.apply(generate_features,axis=1)
    df['feature'] = df['feature'].apply(json.dumps)

    df['label'] = df.apply(generate_labels,axis=1)
    df['label'] = df['label'].apply(json.dumps)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    args.splits = ['train', 'val', 'test'] if not args.debug else ['train']

    data_dir = op.join(DATA_DIR, args.data_dir)
    for split in args.splits:
        print(f'Generating labels and features for {split} split ...')
        yaml_dict = op.join(data_dir, f'{split}.yaml')
        config = load_from_yaml_file(yaml_dict)
        
        labels_and_features = generate_features_for_split(data_dir, split)

        for attr in ['label', 'feature']:
            config[attr] = f'{split}.{attr}.tsv'
            output_file = op.join(data_dir, config[attr])
            tsv_writer(labels_and_features[[0, attr]].values.tolist(),output_file)
        
        with open(yaml_dict, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
