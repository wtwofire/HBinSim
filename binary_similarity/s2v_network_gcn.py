# SAFE TEAM
# attention+(struc2vec+att-b_output)+l2_normalize + mmd
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#concat acfg and lstm, and add attention

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import scipy.sparse as sp
import numpy as np
#from tensorflow.models.research.domain_adaptation.domain_separation import losses
#from tensorflow.models.research.domain_adaptation.domain_separation import utils
#from functools import partial
class NetworkGCN:

    def __init__(self,
                 features_size,
                 rnn_depth,
                 learning_rate,  # Learning rate
                 l2_reg_lambda,
                 rnn_state_size,  # Dimension of the RNN State
                 batch_size,
                 max_instructions,
                 embedding_matrix,  # Matrix containg the embeddings for each asm instruction
                 trainable_embeddings,
                 # if this value is True, the embeddings of the asm instruction are modified by the training.
                 attention_hops,  # attention hops parameter r of [1]
                 attention_detph,  # attention detph parameter d_a of [1]
                 dense_layer_size,  # parameter e of [1]
                 embedding_size,  # size of the final function embedding, in our test this is twice the rnn_state_size
                 max_lv,
                 T_iterations,
                 embedding_size_stru2vec,
                 output_dim,
                 max_basic_block,
                 dropout
                 ):
        print("Features size"+str(features_size))
        self.features_size = features_size
        self.rnn_depth = rnn_depth  # if this value is modified then the RNN becames a multilayer network. In our tests we fix it to 1 feel free to be adventurous.
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.rnn_state_size = rnn_state_size
        self.batch_size = batch_size
        self.max_instructions = max_instructions
        self.embedding_matrix = embedding_matrix
        self.trainable_embeddings = trainable_embeddings
        self.attention_hops = attention_hops
        self.attention_detph = attention_detph
        self.dense_layer_size = dense_layer_size
        self.embedding_size = embedding_size
        self.max_lv=max_lv
        self.T_iterations=T_iterations
        self.embedding_size_stru2vec = embedding_size_stru2vec
        self.output_dim = output_dim
        self.max_basic_block = max_basic_block
        self.dropout = dropout
        self.generateGraphClassificationNetwork()

    
    def Attention_Features(self, input_x, lengths):

        flattened_inputs = tf.reshape(input_x, [-1, tf.shape(input_x)[2]], name="Flattening")
        flattened_embedded = tf.nn.embedding_lookup(self.instruction_embeddings_t, flattened_inputs)
        self.W0 = tf.Variable(tf.constant(1.0 / self.max_instructions, shape=[1, self.max_instructions]), name="W0")
        w0_tiled = tf.tile(tf.expand_dims(self.W0, 0), [tf.shape(flattened_inputs)[0], 1, 1], name="W0_tiled")
        last_outputs = tf.squeeze(tf.nn.l2_normalize(tf.matmul(w0_tiled, flattened_embedded,
                                                               name='features_weighted_mean')), axis=1)
        gather_output2 = tf.reshape(last_outputs, [-1, tf.shape(input_x)[1], self.features_size], name="Deflattening")
        output = tf.identity(gather_output2, name="LSTMOutput")
        output=tf.nn.l2_normalize(output)#[batch_size*MAX_NUM_VERTICES,embedding_matrix]
        return output         


    def Attention_Features_HetGNN_inst(self, input_x, lengths):

        flattened_inputs = tf.reshape(input_x, [-1, tf.shape(input_x)[2], tf.shape(input_x)[3]], name="Flattening")
        #flattened_embedded = tf.nn.embedding_lookup(self.instruction_embeddings_t, flattened_inputs)#[batch*block,max_instruction,features_size]

        # We do a tile to account for training batches
        ws1_tiled = tf.tile(tf.expand_dims(self.WS1, 0), [tf.shape(flattened_inputs)[0], 1, 1], name="WS1_tiled")
        #ws2_tile = tf.tile(tf.expand_dims(self.WS2, 0), [tf.shape(H)[0], 1, 1], name="WS2_tiled")
        # we compute the matrix A
        #tf.nn.leaky_relu()
        self.A = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(ws1_tiled, tf.transpose(flattened_inputs, perm=[0, 2, 1]))),
                               name="IAttention_Matrix")#[batch*block,1,max_instruction]
        # embedding matrix M
        M = tf.squeeze(tf.matmul(self.A, flattened_inputs), axis=1)#[batch*block,features_size]
        #last = tf.reduce_sum(M, axis=1, name="inst_reduce_sum") #[batch*block,features_size]
        gather_output =  tf.reshape(M, [-1, tf.shape(input_x)[1], self.features_size], name="Deflattening")
        output = tf.identity(gather_output, name="InstAtteOutput")
        return output

    def Attention_Features_HetGNN_block(self, input_x, Wa):
        # We do a tile to account for training batches
        #input_x:[batch,block,embedding]
        wsa_tiled = tf.tile(tf.expand_dims(Wa, 0), [tf.shape(input_x)[0], 1, 1], name="Wa_tiled")
        #wsb_tiled = tf.tile(tf.expand_dims(Wb, 0), [tf.shape(H)[0], 1, 1], name="Wb_tiled")

        # we compute the matrix B
        #tf.nn.leaky_relu()
        self.B = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(wsa_tiled, tf.transpose(input_x, perm=[0, 2, 1]))),
                               name="Block_Attention_Matrix")#[batch, 1, block]
        # embedding matrix F
        
        #output = tf.reduce_sum(M, axis=1, name="Block_reduce_sum") #[batch*block,features_size]
        output = tf.identity(self.B, name="BATTE")#[batch, 1, block]
        return output

    def GCNField(self, input_x, input_adj, CONV_PARAMS, Wa, Wb, Wc, name):
        
        W1_tiled = tf.tile(tf.expand_dims(Wa, 0), [tf.shape(input_x)[0], 1, 1], name=name + "_W1_tiled")
        W2_tiled = tf.tile(tf.expand_dims(Wb, 0), [tf.shape(input_x)[0], 1, 1], name=name + "_W2_tiled")

        CONV_PARAMS_tiled = []
        for lv in range(self.max_lv):
            CONV_PARAMS_tiled.append(tf.tile(tf.expand_dims(CONV_PARAMS[lv], 0), [tf.shape(input_x)[0], 1, 1],
                                             name=name + "_CONV_PARAMS_tiled_" + str(lv)))

        w1xv = tf.matmul(input_x, W1_tiled, name=name + "_w1xv")
        l = tf.matmul(input_adj, w1xv, name=name + '_l_iteration' + str(1))
        out = w1xv
        for i in range(self.T_iterations - 1):
            ol = l
            lv = self.max_lv - 1
            while lv >= 0:
                with tf.name_scope('cell_' + str(lv)) as scope:
                    node_linear = tf.matmul(ol, CONV_PARAMS_tiled[lv], name=name + '_conv_params_' + str(lv))
                    if lv > 0:
                        ol = tf.nn.relu(node_linear, name=name + '_relu_' + str(lv))
                    else:
                        ol = node_linear
                lv -= 1

            out = tf.nn.tanh(w1xv + ol, name=name + "_mu_iteration" + str(i + 2))
            l = tf.matmul(input_adj, out, name=name + '_l_iteration' + str(i + 2))
        
        out_atte = self.Attention_Features_HetGNN_block(out,Wc) 
        #fi = tf.expand_dims(out_atte, axis=1, name=name + "_y_potential_expand_dims")
        graph_embedding = tf.squeeze(tf.matmul(out_atte, out, name=name + '_graph_embedding'),axis=1)
      
        return graph_embedding



    def generateGraphClassificationNetwork(self):
        print("Features size:"+str(self.features_size))

        self.instruction_embeddings_t = tf.Variable(initial_value=tf.constant(self.embedding_matrix),
                                                    trainable=self.trainable_embeddings,
                                                    name="instruction_embedding", dtype=tf.float32)

        self.x_1 = tf.placeholder(tf.float32, [None, None, None,None], name="x_1")
        self.adj_1 = tf.placeholder(tf.float32, [None, None, None], name="adj_1")
        self.lenghts_1 = tf.placeholder(tf.int32, [None,None], name='lenghts_1')
        self.x_2 = tf.placeholder(tf.float32, [None, None, None,None], name="x_2")
        self.adj_2 = tf.placeholder(tf.float32, [None, None, None], name="adj_2")
        self.lenghts_2 = tf.placeholder(tf.int32, [None,None], name='lenghts_2')
        self.y = tf.placeholder(tf.float32, [None], name='y_')

        self.x_1_acfg = tf.placeholder(tf.float32, [None, None, 8], name="x_acfg_1")
        self.x_2_acfg = tf.placeholder(tf.float32, [None, None, 8], name="x_acfg_2")

        self.norms = []

        l2_loss = tf.constant(0.0)
        
        #2020-2-28
        
        with tf.name_scope('parameters_MeanField'):

            self.W1_LSTM = tf.Variable(tf.truncated_normal([self.features_size, self.embedding_size], stddev=0.1), 
                                  name="W1_LSTM")
            self.W2_LSTM = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1),
                                  name="W2_LSTM")
            self.W1_ACFG = tf.Variable(tf.truncated_normal([8, self.embedding_size], stddev=0.1), 
                                  name="W1_ACFG")
            self.W2_ACFG = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1),
                                  name="W2_ACFG")
            #self.norms.append(tf.norm(self.W1))

            self.CONV_PARAMS_LSTM = []
            for lv in range(self.max_lv):
                v = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1),
                                name="CONV_PARAMS_LSTM_" + str(lv))
                self.CONV_PARAMS_LSTM.append(v)
                self.norms.append(tf.norm(v))


            #ACFG
            self.CONV_PARAMS_ACFG = []
            #layer1 = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1),
            #                    name="CONV_PARAMS_ACFG_layer1")
            #self.CONV_PARAMS_ACFG.append(layer1)
            for lv in range(self.max_lv):
                v = tf.Variable(tf.truncated_normal([self.embedding_size, self.embedding_size], stddev=0.1),
                                name="CONV_PARAMS_ACFG_" + str(lv))
                self.CONV_PARAMS_ACFG.append(v)
                self.norms.append(tf.norm(v))

            
            #self.norms.append(tf.norm(self.W2))

            ##attention para
            #self.W_omega = tf.Variable(tf.truncated_normal([self.max_basic_block, self.max_basic_block], stddev=0.1))
            #self.B_omega = tf.Variable(tf.truncated_normal([self.max_basic_block], stddev=0.1))
            #self.U_omega = tf.Variable(tf.truncated_normal([self.max_basic_block], stddev=0.1))

            self.WS1 = tf.Variable(tf.truncated_normal([1, self.features_size], stddev=0.1),
                                   name="WS1")
            #self.WS2 = tf.Variable(tf.truncated_normal([self.features_size, self.features_size*2], stddev=0.1),
            #                       name="WS2")
            #attention lstm
            self.WS1_B_LSTM = tf.Variable(tf.truncated_normal([1, self.embedding_size], stddev=0.1),
                                   name="WS1_B_LSTM")
            #self.WS2_B_LSTM = tf.Variable(tf.truncated_normal([self.features_size, self.features_size*2], stddev=0.1),
            #                       name="WS2_B_LSTM")
            #attention ACFG
            self.WS1_B_ACFG = tf.Variable(tf.truncated_normal([1, self.embedding_size], stddev=0.1),
                                   name="WS1_B_ACFG")
            #self.WS2_B_ACFG = tf.Variable(tf.truncated_normal([8, 8*2], stddev=0.1),
            #                       name="WS2_B_ACFG")
            #attention diff feature
            self.WS1_diff_feature = tf.Variable(tf.truncated_normal([1, self.embedding_size*2], stddev=0.1),
                                   name="WS1_B_ACFG")
            self.WS2_diff_feature = tf.Variable(tf.truncated_normal([1, self.embedding_size*2], stddev=0.1),
                                   name="WS2_B_ACFG")

        with tf.name_scope('GCNExtraction1'):
            with tf.variable_scope('blstm1'):
                self.x_1_after_blstm = self.Attention_Features_HetGNN_inst(self.x_1, self.lenghts_1)
        with tf.name_scope('GCNExtraction2'):
            with tf.variable_scope('blstm2'):
                self.x_2_after_blstm = self.Attention_Features_HetGNN_inst(self.x_2, self.lenghts_2)
        
        with tf.name_scope('LSTM_GCNField1'):
            self.lstm_graph_embedding_1 = tf.nn.l2_normalize(self.GCNField(self.x_1_after_blstm, self.adj_1, self.CONV_PARAMS_LSTM, self.W1_LSTM, self.W2_LSTM, self.WS1_B_LSTM, "LSTM_GCNField1"), axis=1,
                name="lstm_graph_embedding_1")

        with tf.name_scope('LSTM_GCNField2'):
            self.lstm_graph_embedding_2 = tf.nn.l2_normalize(self.GCNField(self.x_2_after_blstm, self.adj_2, self.CONV_PARAMS_LSTM, self.W1_LSTM, self.W2_LSTM,self.WS1_B_LSTM, "LSTM_GCNField2"), axis=1,
                name="lstm_graph_embedding_2")

        with tf.name_scope('ACFG_GCNField1'):
            self.acfg_graph_embedding_1 = tf.nn.l2_normalize(self.GCNField(self.x_1_acfg, self.adj_1, self.CONV_PARAMS_ACFG, self.W1_ACFG, self.W2_ACFG, self.WS1_B_ACFG, "ACFG_GCNField1"), axis=1,
                name="acfg_graph_embedding_1")

        with tf.name_scope('ACFG_GCNField2'):
            self.acfg_graph_embedding_2 = tf.nn.l2_normalize(self.GCNField(self.x_2_acfg, self.adj_2, self.CONV_PARAMS_ACFG, self.W1_ACFG, self.W2_ACFG, self.WS1_B_ACFG, "ACFG_GCNField2"), axis=1,
                name="acfg_graph_embedding_2")
        with tf.name_scope('graph_embedding_1'):
            #lstm and acfg agg by attention
            self.a_graph_embedding_1 = tf.concat([self.lstm_graph_embedding_1, self.acfg_graph_embedding_1], 1, 
            name="a_graph_embedding_1")#[batch,feature*2]
            self.b_graph_embedding_1 = tf.concat([self.lstm_graph_embedding_1, self.lstm_graph_embedding_1], 1, 
            name="b_graph_embedding_1")#[batch,feature*2]
            #print("embedding1",tf.shape(self.a_graph_embedding_1))
            #print("embedding2",tf.shape(self.b_graph_embedding_1))

            self.concat_embedding_1 = tf.reshape(tf.concat([self.a_graph_embedding_1, self.b_graph_embedding_1], 1), [-1, 2, self.embedding_size*2],
                name="concat_embedding_1")
            self.concat_att = self.Attention_Features_HetGNN_block(self.concat_embedding_1, self.WS1_diff_feature)#[batch,1,2]
            
            self.graph_embedding_1 = tf.nn.l2_normalize(tf.squeeze(tf.matmul(self.concat_att, tf.reshape(self.a_graph_embedding_1, [-1, 2, self.embedding_size])),axis=1)
                                             ,axis=1,name="embedding1")
        with tf.name_scope('graph_embedding_2'):
            self.a_graph_embedding_2 = tf.concat([self.lstm_graph_embedding_2, self.acfg_graph_embedding_2], 1, 
            name="a_graph_embedding_2")#[batch,feature*2]
            self.b_graph_embedding_2 = tf.concat([self.lstm_graph_embedding_2, self.lstm_graph_embedding_2], 1, 
            name="b_graph_embedding_2")#[batch,feature*2]

            self.concat_embedding_2 = tf.reshape(tf.concat([self.a_graph_embedding_2, self.b_graph_embedding_2], 1), [-1, 2, self.embedding_size*2],
                name="concat_embedding_2")
            self.concat_att = self.Attention_Features_HetGNN_block(self.concat_embedding_2, self.WS1_diff_feature)#[batch,1,2]
            
            self.graph_embedding_2 = tf.nn.l2_normalize(tf.squeeze(tf.matmul(self.concat_att, tf.reshape(self.a_graph_embedding_2, [-1, 2, self.embedding_size])),axis=1)
                                                 ,axis=1, name="embedding2")

        with tf.name_scope('Siamese'):
            self.cos_similarity = tf.reduce_sum(tf.multiply(self.graph_embedding_1, self.graph_embedding_2), axis=1,
                                                name="cosSimilarity")
            #self.cos_similarity = tf.reduce_sum(self.graph_embedding_1*self.graph_embedding_2, 1) / tf.sqrt(tf.reduce_sum(self.graph_embedding_1**2, 1) * tf.reduce_sum(self.graph_embedding_2**2, 1) + 1e-10)
        
        # Regularization
        with tf.name_scope("Regularization"):
            l2_loss += tf.nn.l2_loss(self.WS1)

            #l2_loss += tf.nn.l2_loss(self.W1_LSTM)
            #l2_loss += tf.nn.l2_loss(self.W1_ACFG)

            l2_loss += tf.nn.l2_loss(self.WS1_B_LSTM)
            #l2_loss += tf.nn.l2_loss(self.WS2_B_LSTM)
            l2_loss += tf.nn.l2_loss(self.WS1_B_ACFG)
            #l2_loss += tf.nn.l2_loss(self.WS2_B_ACFG)
            l2_loss += tf.nn.l2_loss(self.WS1_diff_feature)
            l2_loss += tf.nn.l2_loss(self.WS2_diff_feature)
            for lv in range(self.max_lv):
                l2_loss += tf.nn.l2_loss(self.CONV_PARAMS_LSTM[lv])
                l2_loss += tf.nn.l2_loss(self.CONV_PARAMS_ACFG[lv])
            #l2_loss += tf.nn.l2_loss(self.W2)

        # CalculateMean cross-entropy loss
        with tf.name_scope("Loss"):

            self.loss = tf.reduce_mean(tf.squared_difference(self.cos_similarity, self.y), name="loss")

            self.regularized_loss =  self.loss + self.l2_reg_lambda*l2_loss

        # Train step
        with tf.name_scope("Train_Step"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.regularized_loss)
