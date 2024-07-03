#!/bin/sh

# Type of the network to use

#NETWORK_TYPE="Attention_Mean"
#NETWORK_TYPE="Arith_Mean"
#NETWORK_TYPE="RNN"
#NETWORK_TYPE="Annotations"
NETWORK_TYPE="GCN"
# Root path for the experiment
MODEL_PATH=/home/wangyan/experiment/HBinSim/
#MODEL_PATH = H:/papers_experiment_code/GraphEmbedding/Unsupervised-Features-Learning-For-Binary-Similarity-master/

# Path to the sqlite db with diassembled functions
DB_PATH=../data/binary_obf_HBinSim_5-100_none_fla.db

# Path to embedding matrix
EMBEDDING_MATRIX=../data/i2v/token_embedding_matrix_final.npy

# Path to instruction2id dictionary
INS2ID=../data/i2v/word2vec.json

BLOCKINFO=../data/i2v/block_info.json
# Add this argument to train.py to use random instructions embeddings
RANDOM_EMBEDDINGS="-r"

# Add this argument to train.py to use trainable instructions embeddings
TRAINABLE_EMBEDDINGS="-te"

python train.py --o $MODEL_PATH -n $DB_PATH -nn $NETWORK_TYPE -e $EMBEDDING_MATRIX -j $INS2ID -b $BLOCKINFO

#/home/ubuntu/wy/binary_code_similarity/binary_similarity

