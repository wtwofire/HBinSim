# SAFE TEAM
# Hamming_similairyt
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#

import tensorflow as tf

class NetworkBLSTM:

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
                 max_basic_block
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
        self.generateGraphClassificationNetwork()


    def blstmFeatures(self, input_x, lengths):
        #lengths
        #input_x:[batch_size,MAX_NUM_VERTICES,max_instructions]
        #input_adj:[batch_size,MAX_NUM_VERTICES,MAX_NUM_VERTICES]
        #lengths:[batch_size,MAX_NUM_VERTICES]
        flattened_inputs=tf.reshape(input_x,[-1,tf.shape(input_x)[2]],name="Flattening")
        #flattened_inputs[batch_size*MAX_NUM_VERTICES,max_instructions]
        #flattened_lenghts[batch_size*MAX_NUM_VERTICES]
        #max = tf.reduce_max(flattened_lenghts)
        flattened_lenghts = tf.reshape(lengths, [-1])
         
        #flattened_inputs=flattened_inputs[:,:self.max_instructions]
        flattened_embedded = tf.nn.embedding_lookup(self.instruction_embeddings_t, flattened_inputs)
        #flattened_embedded[batch_size*MAX_NUM_VERTICES,max_instructions,word2vec_dim]
               
        
        zeros = tf.zeros(tf.shape(flattened_lenghts)[0], dtype=tf.int32)
        mask = tf.not_equal(flattened_lenghts, zeros)
        int_mask = tf.cast(mask, tf.int32)#数据类型转换
        fake_output = tf.zeros([self.features_size], dtype=tf.float32)

        partitions = tf.dynamic_partition(flattened_embedded, int_mask, 2)#划分根据int_mask标志矩阵flattened_embedded，输出2个矩阵
        real_nodes=partitions[1]
        real_lenghts=tf.boolean_mask(flattened_lenghts,mask)
        fake_zero = tf.tile([fake_output], [tf.shape(flattened_embedded)[0] - tf.shape(partitions[1])[0], 1])

        
        # We create the GRU RNN
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, real_nodes,
                                                                    sequence_length=real_lenghts, dtype=tf.float32, time_major=False)
        # We create the matrix H
        H = tf.concat([output_fw, output_bw], axis=2)
        #H[batch_size*MAX_NUM_VERTICES,max_instructions,2*rnn_state_size]
        # We do a tile to account for training batches
        
        condition_indices = tf.dynamic_partition(
            tf.range(tf.shape(flattened_embedded)[0]), int_mask, 2)
        
        last_outputs = tf.dynamic_stitch(condition_indices, [fake_zero, H])
        H = tf.reshape(last_outputs,[-1,tf.shape(input_x)[1],self.rnn_state_size*2], name="Deflattening")
 
        ws1_tiled = tf.tile(tf.expand_dims(self.WS1, 0), [tf.shape(H)[0], 1, 1], name="WS1_tiled")
        #ws1[batch_size*MAX_NUM_VERTICES,self.attention_detph, 2 * self.rnn_state_size]
        ws2_tile = tf.tile(tf.expand_dims(self.WS2, 0), [tf.shape(H)[0], 1, 1], name="WS2_tiled")
        #ws2[batch_size*MAX_NUM_VERTICES,self.attention_hops, self.attention_detph]
        # we compute the attention matrix A
        self.A = tf.nn.softmax(tf.matmul(ws2_tile, tf.nn.tanh(tf.matmul(ws1_tiled, tf.transpose(H, perm=[0, 2, 1])))),
                               name="Attention_Matrix")
        #transpose H[batch_size*MAX_NUM_VERTICES,2*rnn_state_size,max_instructions]
        #A[batch_size*MAX_NUM_VERTICES,attention_detph,MAX_NUM_VERTICES]
        # embedding matrix M
        M = tf.identity(tf.matmul(self.A, H), name="Attention_Embedding")
        #M[batch_size*MAX_NUM_VERTICES,attention_hops,2 * self.rnn_state_size]
        # we create the flattened version of M
        #flattened_M = tf.reshape(M, [input_x[0],input_x[1],-1])
        flattened_M = tf.reshape(M, [tf.shape(M)[0], self.attention_hops * self.rnn_state_size * 2])

        #flattened_M[batch_size*MAX_NUM_VERTICES,self.attention_hops * self.rnn_state_size * 2]
        return flattened_M
    
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
    
    #structure2vec
    def meanField(self, input_x, input_adj, max_basic_block, T_iterations, embedding_size_stru2vec,Wnode, Wembed, W_output, b_output):
        #input_x[batch_size*MAX_NUM_VERTICES,embedding_size]
        #input_adj:[batch_size,MAX_NUM_VERTICES,MAX_NUM_VERTICES]
        node_val = tf.reshape(tf.matmul(tf.reshape(input_x,[-1,self.features_size]) , Wnode),[-1, max_basic_block, embedding_size_stru2vec])#node_val

        #I = tf.reshape(tf.eye(tf.shape(input_adj)[1],batch_shape=[tf.shape(input_adj)[0]]),[-1,tf.shape(input_adj)[2]])
        #input_adj_flattened = tf.reshape(input_adj, [-1,tf.shape(input_adj)[2]])
        #input_adj_I = tf.reshape(tf.add(input_adj_flattened,I),[-1,tf.shape(input_adj)[1],tf.shape(input_adj)[2]])
        cur_msg = tf.nn.relu(node_val)   #[batch, node_num, embed_dim]
        for t in range(T_iterations):
        #Message convey
            Li_t = tf.matmul(input_adj, cur_msg)  #[batch, node_num, embed_dim]
            #Complex Function
            cur_info = tf.reshape(Li_t, [-1, embedding_size_stru2vec])
            for Wi in Wembed:
                if (Wi == Wembed[-1]):
                    cur_info = tf.matmul(cur_info, Wi)
                else:
                    cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
            neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
            #Adding
            tot_val_t = node_val + neigh_val_t
            #Nonlinearity
            tot_msg_t = tf.nn.tanh(tot_val_t)
            cur_msg = tot_msg_t   #[batch, node_num, embed_dim]
            
        self.W1 = tf.Variable(tf.constant(1.0 / max_basic_block, shape=[1, max_basic_block]), name="W1")
        w1_tiled = tf.tile(tf.expand_dims(self.W1, 0), [tf.shape(cur_msg)[0], 1, 1], name="W1_tiled")
        last_outputs = tf.squeeze(tf.nn.l2_normalize(tf.matmul(w1_tiled, cur_msg,
                                                               name='features_weighted_mean')), axis=1)
        #gather_output2 = tf.reshape(last_outputs, [-1, tf.shape(input_x)[1], self.features_size], name="Deflattening")
        #g_embed = tf.reduce_sum(cur_msg, 1)   #[batch, embed_dim]
        #output = tf.matmul(g_embed, W_output) + b_output  
        output = tf.matmul(last_outputs, W_output)  
        return output

    def generateGraphClassificationNetwork(self):
        print("Features size:"+str(self.features_size))

        self.instruction_embeddings_t = tf.Variable(initial_value=tf.constant(self.embedding_matrix),
                                                    trainable=self.trainable_embeddings,
                                                    name="instruction_embedding", dtype=tf.float32)

        self.x_1 = tf.placeholder(tf.int32, [None, None, None], name="x_1")
        self.adj_1 = tf.placeholder(tf.float32, [None, None, None], name="adj_1")
        self.lenghts_1 = tf.placeholder(tf.int32, [None,None], name='lenghts_1')
        self.x_2 = tf.placeholder(tf.int32, [None, None, None], name="x_2")
        self.adj_2 = tf.placeholder(tf.float32, [None, None, None], name="adj_2")
        self.lenghts_2 = tf.placeholder(tf.int32, [None,None], name='lenghts_2')
        self.y = tf.placeholder(tf.float32, [None], name='y_')

        self.norms = []

        l2_loss = tf.constant(0.0)

        # 1. parameters for MeanField
        with tf.name_scope('parameters_MeanField'):

            self.WS1 = tf.Variable(tf.truncated_normal([self.attention_detph, 2 * self.rnn_state_size], stddev=0.1),
                                   name="WS1")
            self.WS2 = tf.Variable(tf.truncated_normal([self.attention_hops, self.attention_detph], stddev=0.1),
                                   name="WS2")
            
            #size
            rnn_layers_fw = [tf.nn.rnn_cell.GRUCell(size) for size in ([self.rnn_state_size] * self.rnn_depth)]
            rnn_layers_bw = [tf.nn.rnn_cell.GRUCell(size) for size in ([self.rnn_state_size] * self.rnn_depth)]
            
            self.cell_fw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_fw)
            self.cell_bw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_bw)                       

            self.Wnode = tf.Variable(tf.truncated_normal(
                shape = [self.features_size, self.embedding_size_stru2vec], stddev = 0.1, dtype = tf.float32))
            self.Wembed = []
            for i in range(self.max_lv):
                self.Wembed.append(tf.Variable(tf.truncated_normal(
                    shape = [self.embedding_size_stru2vec, self.embedding_size_stru2vec], stddev = 0.1, dtype = tf.float32)))

            self.W_output = tf.Variable(tf.truncated_normal(
                shape = [self.embedding_size_stru2vec, self.output_dim], stddev = 0.1, dtype = tf.float32))
            self.b_output = tf.Variable(tf.constant(0, shape = [self.output_dim], dtype = tf.float32))
            
        with tf.name_scope('BLSTMExtraction1'):
            with tf.variable_scope('blstm1'):
                self.x_1_after_blstm = self.Attention_Features(self.x_1, self.lenghts_1)
        with tf.name_scope('BLSTMExtraction2'):
            with tf.variable_scope('blstm2'):
                self.x_2_after_blstm = self.Attention_Features(self.x_2, self.lenghts_2)
        
        #tf.layers.dense
        #self.dense_1 = tf.nn.relu(tf.layers.dense(self.x_1_after_blstm, self.dense_layer_size))
        #self.dense_2 = tf.nn.relu(tf.layers.dense(self.x_2_after_blstm, self.dense_layer_size))

        #with tf.name_scope('Embedding1'):
        #    self.basic_embedding_1 = tf.layers.dense(self.dense_1, self.embedding_size)
        #with tf.name_scope('Embedding2'):
        #    self.basic_embedding_2 = tf.layers.dense(self.dense_2, self.embedding_size)
        #basic_embedding_1 basic_embedding_2[batch_size*MAX_NUM_VERTICES,embedding_size]
        with tf.name_scope('MeanField1'):
             self.graph_embedding_1 = tf.nn.l2_normalize(self.meanField(self.x_1_after_blstm, self.adj_1, self.max_basic_block, self.T_iterations,
                                                    self.embedding_size_stru2vec, self.Wnode, self.Wembed, self.W_output, self.b_output), axis=1,
                name="embedding1")

        with tf.name_scope('MeanField2'):
            self.graph_embedding_2 = tf.nn.l2_normalize(self.meanField(self.x_2_after_blstm, self.adj_2, self.max_basic_block, self.T_iterations,
                                                    self.embedding_size_stru2vec, self.Wnode, self.Wembed, self.W_output, self.b_output), axis=1,
                name="embedding2")

        with tf.name_scope('Siamese'):
            self.cos_similarity = tf.reduce_sum(tf.multiply(self.graph_embedding_1, self.graph_embedding_2), axis=1,
                                                name="cosSimilarity")
        with tf.name_scope("Hamming_similarity"):
            self.hamming_similarity = tf.reduce_sum(tf.multiply(tf.tanh(self.graph_embedding_1),tf.tanh(self.graph_embedding_2)), axis =1)   
        # Regularization
        with tf.name_scope("Regularization"):
            l2_loss += tf.nn.l2_loss(self.WS1)
            for lv in range(self.max_lv):
                l2_loss += tf.nn.l2_loss(self.Wembed[lv])
            l2_loss += tf.nn.l2_loss(self.WS2)

        # CalculateMean cross-entropy loss
        with tf.name_scope("Loss"):
            #A_square = tf.matmul(self.A, tf.transpose(self.A, perm=[0, 2, 1]))

            #I = tf.eye(tf.shape(A_square)[1])
            #I_tiled = tf.tile(tf.expand_dims(I, 0), [tf.shape(A_square)[0], 1, 1], name="I_tiled")
            #self.A_pen = tf.norm(A_square - I_tiled)

            self.loss = tf.reduce_mean(tf.squared_difference(self.cos_similarity, self.y), name="loss")
            #self.cross_entropy2=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.cos_similarity, labels=self.y))
            self.regularized_loss = self.loss + self.l2_reg_lambda * l2_loss

        # Train step
        with tf.name_scope("Train_Step"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.regularized_loss)
