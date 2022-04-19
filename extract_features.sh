export DATA_ROOT_DIR=/home/ubuntu/s3-drive/data
export VINVL_DATA_DIR=vinvl_data

# python tools/mini_tsv/tsv_demo.py $VINVL_DATA_DIR

python tools/test_sg_net.py \
  --config-file sgg_configs/vgattr/vinvl_x152c4.yaml \
  TEST.IMS_PER_BATCH 1 \
  MODEL.WEIGHT pretrained_model/vinvl_vg_x152c4.pth \
  MODEL.ROI_HEADS.NMS_FILTER 1 \
  MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
  TEST.OUTPUT_FEATURE True \
  OUTPUT_DIR $DATA_ROOT_DIR/$VINVL_DATA_DIR \
  DATA_DIR $DATA_ROOT_DIR/$VINVL_DATA_DIR \
  TEST.IGNORE_BOX_REGRESSION True \
  MODEL.ATTRIBUTE_ON True \
  DATASETS.TEST "('train.yaml', 'val.yaml', 'test.yaml')" \
  DATASETS.FACTORY_TEST "('VGTSVDataset', 'VGTSVDataset', 'VGTSVDataset')"

python postprocess_features.py $VINVL_DATA_DIR

unset DATA_ROOT_DIR
unset VINVL_DATA_DIR