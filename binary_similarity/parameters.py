# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

import argparse
import time
import os
import logging

def getLogger(logfile):
    logger = logging.getLogger(__name__)
    hdlr = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger, hdlr

class Flags:

    def __init__(self):
        parser = argparse.ArgumentParser(description=' cryptoarb.')

        parser.add_argument("-o", "--output", dest="output_file", help="output directory for logging and models", required=False)
        parser.add_argument("-e", "--embedding_matrix", dest="embedding_matrix", help="file with the embedding matrix for the instructions",required=False)
        parser.add_argument("-j", "--json_asm2id", dest="json_asm2id",help="file with the dictionary of instructions ids", required=False)
        parser.add_argument("-b", "--block_info_dir", dest="block_info_dir",help="file with the dictionary of block information", required=False)
        parser.add_argument("-n", "--dbName", dest="db_name", help="Name of the database", required=False)
        parser.add_argument("-ld","--load_dir", dest="load_dir", help="Load the model from directory load_dir", required=False)
        parser.add_argument("-nn","--network_type", help="network type: Arith_Mean, Weighted_Mean, RNN, CCS", required=False, dest="network_type")
        parser.add_argument("-r", "--random", help="if present the network use random embedder", default=False, action="store_true", dest="random_embedding", required=False)
        parser.add_argument("-te","--trainable_embedding", help="if present the network consider the embedding as trainable", action="store_true", dest="trainable_embeddings", default=False)
        parser.add_argument("-cv","--cross_val", help="if present the training is done with cross validiation", default=False, action="store_true", dest="cross_val")

        args = parser.parse_args()
        self.network_type = args.network_type
        #self.network_type = "GCN"
        if self.network_type == "Annotations":
            self.feature_type = 'acfg'
        elif self.network_type in ["Arith_Mean", "Attention_Mean", "RNN","BLSTM","GCN"]:
            self.feature_type = 'lstm_cfg'
        else:
            print("ERROR NETWORK NOT FOUND")
            exit(0)

        self.batch_size = 150           # minibatch size (-1 = whole dataset)250
        self.num_epochs = 30            # number of epochs
        self.embedding_size = 128        #100# dimension of latent layers
        #self.embedding_size = 100
        self.embedding_size_stru2vec = 100
        self.learning_rate = 0.001      # init learning_rate
        self.max_lv = 2                 # embedd depth
        #self.T_iterations= 2            # max rounds of message passing
        self.T_iterations= 2            #1# max rounds of message passing
        self.l2_reg_lambda = 0   # 0 #0.0002#regularization coefficient
        self.num_checkpoints = 1        # max number of checkpoints
        self.out_dir = args.output_file # directory for logging
        #self.out_dir = r"H:\papers_experiment_code\GraphEmbedding\Unsupervised-Features-Learning-For-Binary-Similarity-master\experiment\out"
        self.db_name = args.db_name
        #self.db_name = r"H:\papers_experiment_code\GraphEmbedding\Unsupervised-Features-Learning-For-Binary-Similarity-master\data\OpenSSL_dataset.db"
        self.load_dir=str(args.load_dir)
        self.random_embedding = args.random_embedding
        self.trainable_embeddings = args.trainable_embeddings
        self.cross_val = args.cross_val
        self.cross_val_fold = 5
        self.rnn_state_size = 50  # dimesion of the rnn state
        #self.rnn_depth = 2              # depth of the rnn
        self.rnn_depth = 2  # depth of the rnn
        
        self.rnn_kind = 0               #kind of rnn cell 0: lstm cell 1: GRU cell

        ## ATTENTION PARAMETERS
        self.attention_hops = 10
        self.attention_detph = 250

        # RNN SINGLE PARAMETER
        self.dense_layer_size = 2000
        self.seed = 2                   # random seed
        self.output_dim = 64    #struc2vec need
        self.file_embedding_matrix = args.embedding_matrix
        #self.file_embedding_matrix = r"H:\papers_experiment_code\GraphEmbedding\Unsupervised-Features-Learning-For-Binary-Similarity-master\data\i2v\embedding_matrix.npy"
        self.json_asm2id = args.json_asm2id
        #self.json_asm2id = r"H:\papers_experiment_code\GraphEmbedding\Unsupervised-Features-Learning-For-Binary-Similarity-master\data\i2v\word2id.json"
        self.block_info_dir = args.block_info_dir


        self.max_instructions = 100     # number of instructions150
        self.MAX_NUM_VERTICES = 100    
        self.MIN_NUM_VERTICES = 1
        self.dropout = 0.3     
        self.reset_logdir()
        
    def reset_logdir(self):
        # create logdir
        timestamp = str(int(time.time()))
        #retrain
        #timestamp = "1594996651"
        self.logdir = os.path.abspath(os.path.join(self.out_dir, "runs", timestamp))   
        os.makedirs(self.logdir, exist_ok=True)   

        # create logger
        self.log_file = str(self.logdir)+'/console.log'
        self.logger, self.hdlr = getLogger(self.log_file)

        # create symlink for last_run
        sym_path_logdir = str(self.out_dir)+"/last_run"
        try:
            os.unlink(sym_path_logdir)   
        except:
            pass
        try:            
            os.symlink(self.logdir, sym_path_logdir)
        except:
            print("\nfailed to create symlink!\n")

    def close_log(self):
        self.hdlr.close()
        self.logger.removeHandler(self.hdlr)
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def __str__(self):
        msg = ""
        msg +="\n  Parameters:\n"
        msg +="\tNetwork_Type: {}\n".format(self.network_type)
        msg +="\tRandom embedding: {}\n".format(self.random_embedding)
        msg +="\tTrainable embedding: {}\n".format(self.trainable_embeddings)
        msg +="\tFeature Type: {}\n".format(self.feature_type)
        msg +="\tlogdir: {}\n".format(self.logdir)
        msg +="\tbatch_size: {}\n".format(self.batch_size)
        msg +="\tnum_epochs: {}\n".format(self.num_epochs)
        msg +="\tembedding_size: {}\n".format(self.embedding_size)
        msg +="\tlearning_rate: {}\n".format(self.learning_rate)
        msg +="\tmax_lv: {}\n".format(self.max_lv)
        msg +="\tT_iterations: {}\n".format(self.T_iterations)
        msg +="\tl2_reg_lambda: {}\n".format(self.l2_reg_lambda)
        msg +="\tnum_checkpoints: {}\n".format(self.num_checkpoints)
        msg +="\tseed: {}\n".format(self.seed)
        msg +="\tMAX_NUM_VERTICES: {}\n".format(self.MAX_NUM_VERTICES)
        msg += "\tMax Instructions per cfg node: {}\n".format(self.max_instructions)
        if self.network_type == "RNN" or self.network_type=="Attention":
            msg += "\tRNN type (0, lstm; 1, GRU): {}\n".format(self.rnn_kind)
            msg += "\tRNN Depth: {}\n".format(self.rnn_depth)
        if self.network_type== "Attention":
            msg += "\tAttention hops:{}\n".format(self.attention_hops)
            msg += "\tAttention depth:{}\n".format(self.attention_detph)
        if self.network_type=="RNN_SINGLE":
            msg += "\tAttention hops:{}\n".format(self.attention_hops)
            msg += "\tAttention depth:{}\n".format(self.attention_detph)
            msg += "\tDense Layer Size:{}\n".format(self.dense_layer_size)
        if self.network_type=="BLSTM":
            msg += "\tAttention hops:{}\n".format(self.attention_hops)
            msg += "\tAttention depth:{}\n".format(self.attention_detph)
            msg += "\tDense Layer Size:{}\n".format(self.dense_layer_size)
        return msg
