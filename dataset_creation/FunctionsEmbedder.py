# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/

import json

import numpy as np
import networkx as nx
from networkx import json_graph
from scipy import sparse
import sqlite3
from tqdm import tqdm
import traceback
import sys
sys.path.append("..")
from asm_embedding.FunctionNormalizer import FunctionNormalizer
from binary_similarity.HBinSimEmbedder import HBinSimEmbedder
import time

class FunctionsEmbedder:

    def __init__(self,  model, batch_size, max_instruction, max_basic_block):
        self.batch_size = batch_size
        self.max_basic_block = max_basic_block
        self.normalizer = FunctionNormalizer(max_instruction)
        self.HBinSim = HBinSimEmbedder(model)
        self.HBinSim.loadmodel()
        self.HBinSim.get_tensor()

    def compute_embeddings(self, functions, lenghts):

        input_adj, input_x, input_acfg = zip(*functions)

        embeddings = self.HBinSim.embedd(input_x, input_adj, input_acfg, lenghts)
        return embeddings

    @staticmethod
    def create_table(db_name, table_name):
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY, {}  TEXT)".format(table_name, table_name))
        conn.commit()
        conn.close()


    def padAndFilterLSTM_Embedding(self, input_x, input_len):

        output_g = []
        output_len=[]

        for g, lens in zip(input_x, input_len):

            try:
                # graph 1
                adj1 = g[0]
                nodes1 = g[1]
                nodes1_acfg = g[2]


                if (len(nodes1) <= self.max_basic_block):

                    pad_lenght1 = self.max_basic_block - len(nodes1)
                    new_node1 = np.pad(nodes1, [(0, pad_lenght1), (0, 0), (0, 0)], mode='constant')
                    nodes1_acfg_normaliz = np.pad(nodes1_acfg, [(0, pad_lenght1), (0, 0)], mode='constant')

                    pad_lenght1 = self.max_basic_block - adj1.shape[0]
                    adj1_dense = np.pad(adj1.todense(), [(0, pad_lenght1), (0, pad_lenght1)], mode='constant')
                    #adj1_normalized = preprocess_adj_undirect(adj1_dense)
                    g1 = (adj1_dense, new_node1, nodes1_acfg_normaliz)

                    output_g.append(g1)
                    new_lens_0 = lens+[0]*(self.max_basic_block-len(lens))
                    output_len.append(new_lens_0)
                else:
                    new_node1 = np.asarray(nodes1[0:self.max_basic_block])
                    nodes1_acfg_normaliz = np.asarray(nodes1_acfg[0:self.max_basic_block])

                    adj1_todense = adj1.todense()
                    adj1_dense = np.asarray(adj1_todense[0:self.max_basic_block,0:self.max_basic_block])

                    new_lens_0 = np.asarray(lens[0:self.max_basic_block])
                    g1 = (adj1_dense, new_node1, nodes1_acfg_normaliz)
                    output_g.append(g1)
                    output_len.append(new_lens_0)
                
            except:
                print("########:")
                traceback.print_exc()
                pass

        return output_g, output_len

    def get_data_from_cfg(self, cfg):
        adj = sparse.csr_matrix([1,1])
        lenghts = []
        node_matrix = []
        try:
            adj = nx.adjacency_matrix(cfg)
            nodes = cfg.nodes(data=True)
            for i, n in enumerate(nodes):
                node_filtered,length_filtered = self.normalizer.normalize(n[1]['embeddings'])
                lenghts.append(length_filtered)
                node_matrix.append(node_filtered)
        except:
            pass
        return adj, node_matrix, lenghts

    def remove_bad_acfg_node(self, g):
        nodeToRemove = []
        for n in g.nodes(data=True):
            f = n[1]['features']
            if len(f.keys()) == 0:
                nodeToRemove.append(n[0])
        for n in nodeToRemove:
            g.remove_node(n)
        return g

    def get_node_matrix(self, nodes):
        num_node = len(nodes)
        node_matrix = np.zeros([num_node, 8])
        for i, n in enumerate(nodes):
            f = n[1]['features']
            if isinstance(f['constant'], list):
                node_matrix[i, 0] = len(f['constant'])
            else:
                node_matrix[i, 0] = f['constant']
            if isinstance(f['string'], list):
                node_matrix[i, 1] = len(f['string'])
            else:
                node_matrix[i, 1] = f['string']
            node_matrix[i, 2] = f['transfer']
            node_matrix[i, 3] = f['call']
            node_matrix[i, 4] = f['instruction']
            node_matrix[i, 5] = f['arith']
            node_matrix[i, 6] = f['offspring']
            node_matrix[i, 7] = f['betweenness']
        return node_matrix

    def get_data_from_acfg(self, g):
        g = self.remove_bad_acfg_node(g)
        if len(g.nodes) > 0:
            adj = nx.adjacency_matrix(g)
            node_matrix = self.get_node_matrix(g.nodes(data=True))
        else:
            adj = sparse.bsr_matrix(np.zeros([1, 1]))
            node_matrix = np.zeros([1, 8])
        lenght = 8
        return adj, node_matrix, lenght

    def compute_and_save_embeddings_from_db(self, db_name, table_name):
        FunctionsEmbedder.create_table(db_name, table_name)
        conn = sqlite3.connect(db_name)
        cur = conn.cursor()
        q = cur.execute("SELECT id FROM functions WHERE id not in (SELECT id from {})".format(table_name))
        ids = q.fetchall()

        for i in tqdm(range(0, len(ids), self.batch_size)):
            batch_ids = ids[i:i+self.batch_size]
            functions = []
            lengths = []
            for my_id in batch_ids:
                #q = cur.execute("SELECT instructions_list FROM filtered_functions where id=?", my_id)
                q0_lstm = cur.execute("SELECT lstm_cfg FROM lstm_cfg WHERE id=?", my_id)
                adj0, node0, lengths0 = self.get_data_from_cfg(json_graph.adjacency_graph(json.loads(q0_lstm.fetchone()[0])))
                
                q0_acfg = cur.execute("SELECT acfg FROM acfg WHERE id=?", my_id)
                adj0_acfg, node0_acfg, lengths0_acfg = self.get_data_from_acfg(json_graph.adjacency_graph(json.loads(q0_acfg.fetchone()[0])))

                functions.append((adj0, node0, node0_acfg))
                lengths.append(lengths0)
            functions, output_lengths = self.padAndFilterLSTM_Embedding(functions, lengths)
            starttime = time.time()
            embeddings = self.compute_embeddings(functions, output_lengths)
            endtime = time.time()
            avg_time = (endtime -starttime) / float(len(batch_ids))
            print("time: ",avg_time)
            for l, id in enumerate(batch_ids):
                cur.execute("INSERT INTO {} VALUES (?,?)".format(table_name), (id[0], np.array2string(embeddings[l])))
            conn.commit()

