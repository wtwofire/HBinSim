# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
import tensorflow as tf


# structure2vec
# DE-MF : discriminative embedding using Mean Field


class NetworkLSTM:

    def __init__(self,
                 features_size,
                 embedding_size,
                 max_lv,
                 T_iterations,
                 learning_rate,
                 l2_reg_lambda,
                 batch_size,
                 max_instructions,
                 max_nodes,
                 rnn_depth,
                 rnn_kind,
                 embedding_matrix,
                 trainable_embeddings
                 ):
        print("Features size"+str(features_size))
        self.features_size = features_size
        self.embedding_size = embedding_size
        self.max_lv = max_lv
        self.T_iterations = T_iterations
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.RRN_HIDDEN = features_size
        self.batch_size = batch_size
        self.max_instructions = max_instructions
        self.max_nodes = max_nodes
        self.rnn_depth = rnn_depth
        self.rnn_kind=rnn_kind
        self.embedding_matrix = embedding_matrix
        self.trainable_embeddings = trainable_embeddings
        self.generateGraphClassificationNetwork()

    def extract_axis_1(self, data, ind):
        """
        Get specified elements along the first axis of tensor.
        :param data: Tensorflow tensor that will be subsetted.
        :param ind: Indices to take (one for each element along axis 0 of data).
        :return: Subsetted tensor.
        """
        ind=tf.nn.relu(ind-1)
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)

        return res

    def create_flattening_array(self, max_nodes, batch_size):
        shape_array = []
        for p in range(0, batch_size):
            for i in range(0, max_nodes):
                shape_array.append([p, i])
        return shape_array

    def create_gather_array(self, max_nodes, batch_size):
        shape_array = []
        for p in range(0, batch_size):
            x = []
            for i in range(0, max_nodes):
                x.append([0, i + p * max_nodes])
            shape_array.append(x)
        return shape_array

    def lstmFeatures(self, input_x, lengths):
        flattened_embedded = tf.reshape(input_x, [-1, tf.shape(input_x)[2], tf.shape(input_x)[3]], name="Flattening")
        #flattened_embedded = tf.nn.embedding_lookup(self.instruction_embeddings_t, flattened_inputs)
        last_outputs = tf.squeeze(tf.nn.l2_normalize(tf.reduce_mean(flattened_embedded, name='arith_mean', axis=1), axis=1))

        gather_output2 = tf.reshape(last_outputs, [-1, tf.shape(input_x)[1], self.features_size], name="Deflattening")
        output = tf.identity(gather_output2, name="LSTMOutput")
        output = tf.nn.l2_normalize(output)
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
        
        fi = tf.expand_dims(tf.reduce_sum(out, axis=1, name=name + "_y_potential_reduce_sum"), axis=1,
                            name=name + "_y_potential_expand_dims")
        graph_embedding = tf.matmul(fi, W2_tiled, name=name + '_graph_embedding')
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
                self.x_1_after_blstm = self.lstmFeatures(self.x_1, self.lenghts_1)
        with tf.name_scope('GCNExtraction2'):
            with tf.variable_scope('blstm2'):
                self.x_2_after_blstm = self.lstmFeatures(self.x_2, self.lenghts_2)
        
        with tf.name_scope('LSTM_GCNField1'):
            self.graph_embedding_1 = tf.nn.l2_normalize(tf.squeeze(self.GCNField(self.x_1_after_blstm, self.adj_1, self.CONV_PARAMS_LSTM, self.W1_LSTM, self.W2_LSTM, self.WS1_B_LSTM, "LSTM_GCNField1"), axis=1), axis=1,
                name="graph_embedding_1")

        with tf.name_scope('LSTM_GCNField2'):
            self.graph_embedding_2 = tf.nn.l2_normalize(tf.squeeze(self.GCNField(self.x_2_after_blstm, self.adj_2, self.CONV_PARAMS_LSTM, self.W1_LSTM, self.W2_LSTM,self.WS1_B_LSTM, "LSTM_GCNField2"), axis=1), axis=1,
                name="graph_embedding_2")

        #with tf.name_scope('ACFG_GCNField1'):
        #    self.acfg_graph_embedding_1 = tf.nn.l2_normalize(tf.squeeze(self.GCNField(self.x_1_acfg, self.adj_1, self.CONV_PARAMS_ACFG, self.W1_ACFG, self.W2_ACFG, self.WS1_B_ACFG, "ACFG_GCNField1"), axis=1), axis=1,
        #        name="acfg_graph_embedding_1")

        #with tf.name_scope('ACFG_GCNField2'):
        #    self.acfg_graph_embedding_2 = tf.nn.l2_normalize(tf.squeeze(self.GCNField(self.x_2_acfg, self.adj_2, self.CONV_PARAMS_ACFG, self.W1_ACFG, self.W2_ACFG, self.WS1_B_ACFG, "ACFG_GCNField2"), axis=1), axis=1,
        #        name="acfg_graph_embedding_2")

        #with tf.name_scope('graph_embedding_1'):
        #    self.graph_embedding_1 = tf.nn.l2_normalize(tf.add(self.lstm_graph_embedding_1, self.acfg_graph_embedding_1), axis=1, name="embedding1")
        #with tf.name_scope('graph_embedding_2'):
        #    self.graph_embedding_2 = tf.nn.l2_normalize(tf.add(self.lstm_graph_embedding_2, self.acfg_graph_embedding_2), axis=1, name="embedding2")

        with tf.name_scope('Siamese'):
            #self.cos_similarity = tf.reduce_sum(tf.multiply(self.graph_embedding_1, self.graph_embedding_2), axis=1,
            #                                    name="cosSimilarity")
            self.cos_similarity = tf.reduce_sum(self.graph_embedding_1*self.graph_embedding_2, 1) / tf.sqrt(tf.reduce_sum(self.graph_embedding_1**2, 1) * tf.reduce_sum(self.graph_embedding_2**2, 1) + 1e-10)
        
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
