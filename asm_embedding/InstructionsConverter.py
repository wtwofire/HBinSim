# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
import json
import numpy as np
import math

class InstructionsConverter:

    def __init__(self, json_i2id, block_info_dir,tokenembedding):

        with open(json_i2id, 'r') as f:
            line = f.readline()
            self.i2id = json.loads(line)

        with open(block_info_dir, 'r') as b:
            line = b.readline()
            self.block_info = json.loads(line)

        self.tokenEmbeddings = tokenembedding

    def convert_to_ids(self, instructions_list):
        ret_array = []
        # For each instruction we add +1 to its ID because the first
        # element of the embedding matrix is zero
        for x in instructions_list:
            if x in self.i2id:
                ret_array.append(self.i2id[x] + 1)
            #elif 'X_' in x:
                # print(str(x) + " is not a known x86 instruction")
            #    ret_array.append(self.i2id['X_UNK'] + 1)
            #elif 'A_' in x:
                # print(str(x) + " is not a known arm instruction")
            #    ret_array.append(self.i2id['A_UNK'] + 1)

            else:
                # print("There is a problem " + str(x) + " does not appear to be an asm or arm instruction")
                ret_array.append(self.i2id["UNK"] + 1)
        return ret_array


    def inst_convert(self,instruction_list, numInsns, opcodeCounts):
        instEmbeddings = []
        opcodeEmbeddings = []
        operandEmbeddings = []
        opcode_idx_list = self.block_info["opcode_idx_list"]
        totalBlockNum = self.block_info["totalblocknum"]
        insToBlockCounts = self.block_info["insToBlockCounts"]
        temp = 0
        embedding_size = 64
        #rev_i2id = {v : k for k, v in self.i2id.items()}
        if len(instruction_list) != 0:
            for token in instruction_list:
                if token in self.i2id:
                    tokenid = self.i2id[token] + 1
                else:
                    tokenid = self.i2id["UNK"] + 1
                #if tokenid != 0:
                tokenEmbedding = self.tokenEmbeddings[tokenid]
                if tokenid-1 in opcode_idx_list:
                    # the first opcode not execute
                    if temp != 0:
                        # if no operand, give zeros
                        if len(operandEmbeddings) == 0:
                            operandEmbedding_mean = np.zeros(embedding_size)
                        else:
                            operandEmbedding_mean = np.mean(operandEmbeddings,axis=0)
                        inst = np.concatenate((opcodeEmbeddings, operandEmbedding_mean), axis=0)
                        instEmbeddings.append(inst)
                        operandEmbeddings.clear()
                    # here we multiple the embedding with its TF-IDF weight if the token is an opcode
                    if token in insToBlockCounts and token in opcodeCounts:

                        tf_weight = opcodeCounts[token] / numInsns
                        x = totalBlockNum / insToBlockCounts[token]
                        idf_weight = math.log(x)
                        tf_idf_weight = tf_weight * idf_weight
                    # print("tf-idf: ", token, opcodeCounts[token], opcodeNum, totalBlockNum, insToBlockCounts[token], tf_weight, idf_weight)
                    else:
                        tf_idf_weight = 1
                    opcodeEmbeddings = np.array(tokenEmbedding * tf_idf_weight)
                    temp = 1
                else:
                    operandEmbeddings.append(tokenEmbedding)
                #else:
                #    opcodeEmbeddings= np.zeros(embedding_size)
                #    operandEmbeddings.append(np.zeros(embedding_size))
            # the last inst
            if len(operandEmbeddings) == 0:
                operandEmbedding_mean = np.zeros(embedding_size)
            else:
                operandEmbedding_mean = np.mean(operandEmbeddings,axis=0)
            inst = np.concatenate((opcodeEmbeddings, operandEmbedding_mean), axis=0)
            instEmbeddings.append(inst)

        else:
            instEmbeddings = np.zeros(embedding_size*2)

        return instEmbeddings






