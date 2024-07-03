# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
import json
import r2pipe
import networkx as nx
#from dataset_creation.BlockFeaturesExtractor import BlockFeaturesExtractor
from BlockFeaturesExtractor import BlockFeaturesExtractor
import traceback
import numpy as np
import math


# register list
register_list_8_byte = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp', 'rip', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']

register_list_4_byte = ['eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp', 'eip', 'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d']

register_list_2_byte = ['ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp', 'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w']

register_list_1_byte = ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl', 'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']


class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """

    # ----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

class RadareFunctionAnalyzer:

    def __init__(self, filename, use_symbol, block_info):
        self.r2 = r2pipe.open(filename, flags=['-2'])
        self.filename = filename
        self.arch, _ = self.get_arch()
        self.use_symbol = use_symbol

        with open(block_info + "/block_info.json", 'r') as b:
            line = b.readline()
            self.block_info = json.loads(line)

        with open(block_info + "/word2vec.json", 'r') as b:
            line = b.readline()
            self.i2id = json.loads(line)

        emb_dir = block_info + "/token_embedding_matrix_final.npy"
        self.tokenEmbeddings = np.float32(np.load(emb_dir))

    def __enter__(self):
        return self

    @staticmethod
    def filter_reg(op):
        if op["value"] in register_list_1_byte:
            ret = 'reg1'
        elif op["value"]  in register_list_2_byte:
            ret = 'reg2'
        elif op["value"]  in register_list_4_byte:
            ret = 'reg4'
        elif op["value"]  in register_list_8_byte:
            ret = 'reg8'
        else:
            ret = "reg"
        return ret

    @staticmethod
    def filter_imm(op, string_addr):
        if op['value'] in string_addr:
            ret = str('STR')
        else:
            ret = str('HIMM')

        return ret
    
    @staticmethod
    def filter_mem_reg(value):
        if value in register_list_1_byte:
            ret = 'reg1'
        elif value  in register_list_2_byte:
            ret = 'reg2'
        elif value  in register_list_4_byte:
            ret = 'reg4'
        elif value  in register_list_8_byte:
            ret = 'reg8'
        return ret

    @staticmethod
    def filter_mem(op):
        if "base" not in op:
            op["base"] = 0
        
        if op["base"] == 0:
            ret = "[" + "MEM" + "]"

        elif op["base"] != 0 and "index" not in op:
            reg_base = RadareFunctionAnalyzer.filter_mem_reg(op["base"])
            disp = str('HIMM')
            ret = '[' + reg_base + "+" + disp + ']'      

        else:
            reg_base = RadareFunctionAnalyzer.filter_mem_reg(op["base"])
            index = RadareFunctionAnalyzer.filter_mem_reg(op["index"])
            disp = str('HIMM')
            scale = str('HIMM')
            ret = '[' + reg_base + "+" + index + "*" + scale + "+" + disp + ']'
        return ret

    @staticmethod
    def filter_memory_references(op, string_addr):

        if op["type"] == 'reg':
            inst = RadareFunctionAnalyzer.filter_reg(op)
        elif op["type"] == 'imm':
            inst = RadareFunctionAnalyzer.filter_imm(op, string_addr)
        elif op["type"] == 'mem':
            inst = RadareFunctionAnalyzer.filter_mem(op)
        else:
            inst = "UNK"
        return inst

    @staticmethod
    def get_callref(my_function, depth):
        calls = {}
        if 'callrefs' in my_function and depth > 0:
            for cc in my_function['callrefs']:
                if cc["type"] == "C":
                    calls[cc['at']] = cc['addr']
        return calls


    def inst_convert(self, instruction_list, numInsns, opcodeCounts):
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
                        #print("opcode: ", token)
                        #print("inst: ", type(inst.tolist()))
                        instEmbeddings.append(inst.tolist())
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
                    #print("operand: ", token)
                    operandEmbeddings.append(tokenEmbedding)
            # the last inst
            if len(operandEmbeddings) == 0:
                operandEmbedding_mean = np.zeros(embedding_size)
            else:
                operandEmbedding_mean = np.mean(operandEmbeddings,axis=0)
            inst = np.concatenate((opcodeEmbeddings, operandEmbedding_mean), axis=0)
            instEmbeddings.append(inst.tolist())

        else:
            instEmbeddings = np.zeros(embedding_size*2).tolist()
        #print(instEmbeddings)
        return instEmbeddings


    def process_instructions(self, instructions, string_addr):
        
        tokens = []
        numInsns = 0
        opcodeCounts = {}
        #filtered_instructions = []
        embeddings = []
        for insn in instructions:
            numInsns = numInsns + 1
            #operands = []
            if 'opex' not in insn:
                continue

            if insn["mnemonic"] not in opcodeCounts:
                opcodeCounts[insn["mnemonic"]] = 1
            else:
                opcodeCounts[insn["mnemonic"]] = opcodeCounts[insn["mnemonic"]] + 1

            tokens.append(insn["mnemonic"])
            for op in insn["opex"]["operands"]:
                optoken = RadareFunctionAnalyzer.filter_memory_references(op, string_addr)
                if optoken != '':
                    tokens.append(optoken)


        embeddings = self.inst_convert(tokens, numInsns, opcodeCounts)

        return tokens, numInsns, opcodeCounts, embeddings

    def process_block(self, block):
        bytes = ""
        disasm = []
        for op in block['ops']:
            if 'disasm' in op:
                disasm.append(op['disasm'])
                bytes += str(op['bytes'])

        self.r2.cmd("s " + str(block['offset']))
        instructions = json.loads(self.r2.cmd("aoj " + str(len(block['ops']))))
        string_addresses = [s['vaddr'] for s in json.loads(self.r2.cmd("izzj"))]
        bfe = BlockFeaturesExtractor(self.arch, instructions, block['ops'], string_addresses)
        annotations = bfe.getFeatures()
        tokens, numInsns, opcodeCounts, embeddings = self.process_instructions(instructions, string_addresses)

        return disasm, bytes, annotations, tokens, numInsns, opcodeCounts, embeddings

    def function_to_cfg(self, func):
        if self.use_symbol:
            s = 'vaddr'
        else:
            s = 'offset'

        self.r2.cmd('s ' + str(func[s]))
        try:
            cfg = json.loads(self.r2.cmd('agfj ' + str(func[s])))
        except:
            cfg = []

        my_cfg = nx.DiGraph()
        acfg = nx.DiGraph()
        lstm_cfg = nx.DiGraph()

        if len(cfg) == 0:
            return my_cfg, acfg, lstm_cfg
        else:
            cfg = cfg[0]

        for block in cfg['blocks']:
            disasm, block_bytes, annotations, tokens, numInsns, opcodeCounts, embeddings = self.process_block(block)
            my_cfg.add_node(block['offset'], asm=block_bytes, label=disasm)
            acfg.add_node(block['offset'], features=annotations)
            #print(type(embeddings))
            lstm_cfg.add_node(block['offset'], features=tokens, numInsns=numInsns, opcodeCounts=opcodeCounts, embeddings=embeddings)

        for block in cfg['blocks']:
            if 'jump' in block:
                if block['jump'] in my_cfg.nodes:
                    my_cfg.add_edge(block['offset'],block['jump'])
                    acfg.add_edge(block['offset'], block['jump'])
                    lstm_cfg.add_edge(block['offset'], block['jump'])
            if 'fail' in block:
                if block['fail'] in my_cfg.nodes:
                    my_cfg.add_edge(block['offset'],block['fail'])
                    acfg.add_edge(block['offset'], block['fail'])
                    lstm_cfg.add_edge(block['offset'], block['fail'])

        between = nx.betweenness_centrality(acfg)
        for n in acfg.nodes(data=True):
            d = n[1]['features']
            d['offspring'] = len(nx.descendants(acfg, n[0]))
            d['betweenness'] = between[n[0]]
            n[1]['features'] = d

        return my_cfg, acfg, lstm_cfg

    def get_arch(self):
        arch = None
        bits = None
        try:
            info = json.loads(self.r2.cmd('ij'))
            if 'bin' in info:
                arch = info['bin']['arch']
                bits = info['bin']['bits']
        except:
            print("Error loading file")
            arch = None
            bits = None
        return arch, bits

    def find_functions(self):
        self.r2.cmd('aaa')
        try:
            function_list = json.loads(self.r2.cmd('aflj'))
        except:
            function_list = []
        return function_list

    def find_functions_by_symbols(self):
        self.r2.cmd('aa')
        try:
            symbols = json.loads(self.r2.cmd('isj'))
            fcn_symb = [s for s in symbols if s['type'] == 'FUNC']
        except:
            fcn_symb = []
        return fcn_symb

    def analyze(self):
        if self.use_symbol:
            function_list = self.find_functions_by_symbols()
        else:
            function_list = self.find_functions()

        result = {}
        for my_function in function_list:
            if self.use_symbol:
                address = my_function['vaddr']
            else:
                address = my_function['offset']

            try:
                cfg, acfg, lstm_cfg = self.function_to_cfg(my_function)
                result[my_function['name']] = {'cfg': cfg, "acfg": acfg, "lstm_cfg": lstm_cfg, "address": address}
            except:
                print("Error in functions: {} from {}".format(my_function['name'], self.filename))
                traceback.print_exc()
                pass
        return result

    def close(self):
        self.r2.quit()

    def __exit__(self, exc_type, exc_value, traceback):
        self.r2.quit()



