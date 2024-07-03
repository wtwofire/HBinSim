#!/bin/sh


# Root path for the model
MODEL_PATH= /home/wangyan/experiment/HBinSim/checkpoints

# Path to the sqlite db with diassembled functions
DB_PATH=../data/esh-dataset-1523.db

Embeddings_Table= "embeddings"

python ExperimentUtil.py -db $DB_PATH  -mod $MODEL_PATH -e

#/home/ubuntu/wy/binary_code_similarity/binary_similarity

