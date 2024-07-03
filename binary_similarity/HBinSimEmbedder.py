import tensorflow as tf
# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/
#from s2v_trainer import S2VTrainer
import numpy as np
import random
import os
import sys
class HBinSimEmbedder:

    def __init__(self, model_file):
        self.model_file_dir = model_file
        self.session = None
        self.x_1 = None
        self.adj_1 = None
        self.len_1 = None
        self.x_1_acfg = None
        self.emb = None
        self.seed = 2
        random.seed(self.seed)
        np.random.seed(self.seed)

    def loadmodel(self):
        tf.reset_default_graph()
        with tf.Graph().as_default() as g:
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
            )

            sess = tf.Session(config=session_conf)
            tf.set_random_seed(self.seed)

            saver = tf.train.import_meta_graph(os.path.join(self.model_file_dir, "model.meta"))
            #checkpoint_dir = os.path.abspath(os.path.join(self.model_file_dir, "checkpoints"))
            saver.restore(sess, os.path.join(self.model_file_dir, "model"))
            self.session=sess
        #with tf.Session() as sess:
        #    saver = tf.train.import_meta_graph(os.path.join(self.model_file_dir, "model.meta"))
        #    saver.restore(sess, os.path.join(self.model_file_dir, "model"))
        #    self.session=sess
        return sess

    def get_tensor(self):
        self.x_1 = self.session.graph.get_tensor_by_name("x_1:0")
        self.adj_1 = self.session.graph.get_tensor_by_name("adj_1:0")
        self.len_1 = self.session.graph.get_tensor_by_name("lenghts_1:0")
        self.x_1_acfg = self.session.graph.get_tensor_by_name("x_acfg_1:0")
        self.emb = tf.nn.l2_normalize(self.session.graph.get_tensor_by_name('graph_embedding_1/embedding1:0'), axis=1)

    def embedd(self, input_x, input_adj, input_acfg, input_lengths):

        out_embedding= self.session.run(self.emb, feed_dict = {
                                                    self.x_1: input_x,
                                                    self.adj_1: input_adj,
                                                    self.x_1_acfg: input_acfg,
                                                    self.len_1: input_lengths,})

        return out_embedding
