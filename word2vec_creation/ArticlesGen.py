# SAFE TEAM
#
#
# distributed under license: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt) #
#
#import FunctionAnalyzerRadare
#from FunctionAnalyzerRadare import RadareFunctionAnalyzer
import json
#import multiprocessing
#from multiprocessing import Pool
#from multiprocessing.dummy import Pool as ThreadPool
import os
import random
import signal
#import sqlite3
from tqdm import tqdm
from networkx.readwrite import json_graph
from deepwalk import deepwalk
import traceback

import r2pipe
import networkx as nx
from BlockFeaturesExtractor import BlockFeaturesExtractor


# register list
register_list_8_byte = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp', 'rip', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']

register_list_4_byte = ['eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp', 'eip', 'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d']

register_list_2_byte = ['ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp', 'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w']

register_list_1_byte = ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl', 'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']


# this list contains all the indices of the opcode in opcode_list
opcode_idx_list = []
dictionary = {}
reversed_dictionary = {}
# this list contains all the opcode in the two binaries
opcode_list = []
#IDF
insToBlockCounts = {}

totalblocknum = 0





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

    def __init__(self, filename, use_symbol, outputDir):
        self.r2 = r2pipe.open(filename, flags=['-2'])
        self.filename = filename
        self.arch, _ = self.get_arch()
        self.use_symbol = use_symbol
        self.outputDir = outputDir

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


    def process_instructions(self, instructions, string_addr):
        
        tokens = []
        numInsns = 0
        opcodeCounts = {}
        countedInsns = []
        global opcode_list
        global insToBlockCounts
        #filtered_instructions = []
        for insn in instructions:
            numInsns = numInsns + 1
            #operands = []
            if 'opex' not in insn:
                continue
            if insn["mnemonic"] not in opcode_list:
                opcode_list.append(insn["mnemonic"])
            
            if insn["mnemonic"] not in countedInsns:
                if insn["mnemonic"] not in insToBlockCounts.keys():
                    insToBlockCounts[insn["mnemonic"]] = 1
                else:
                    insToBlockCounts[insn["mnemonic"]] = insToBlockCounts[insn["mnemonic"]] + 1
                countedInsns.append(insn["mnemonic"])

            if insn["mnemonic"] not in opcodeCounts:
                opcodeCounts[insn["mnemonic"]] = 1
            else:
                opcodeCounts[insn["mnemonic"]] = opcodeCounts[insn["mnemonic"]] + 1

            tokens.append(insn["mnemonic"])
            for op in insn["opex"]["operands"]:
                optoken = RadareFunctionAnalyzer.filter_memory_references(op, string_addr)
                if optoken != '':
                    tokens.append(optoken)
        return tokens, numInsns, opcodeCounts

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
        tokens, numInsns, opcodeCounts = self.process_instructions(instructions, string_addresses)


        return disasm, bytes, annotations, tokens, numInsns, opcodeCounts

    def function_to_cfg(self, func):
        global totalblocknum
        if self.use_symbol:
            s = 'vaddr'
        else:
            s = 'offset'

        self.r2.cmd('s ' + str(func[s]))
        try:
            cfg = json.loads(self.r2.cmd('agfj ' + str(func[s])))
        except:
            cfg = []
            traceback.print_exc()

        my_cfg = nx.DiGraph()
        acfg = nx.DiGraph()
        lstm_cfg = nx.DiGraph()

        if len(cfg) == 0:
            return my_cfg, acfg, lstm_cfg
        else:
            cfg = cfg[0]
        #cfg["nargs"]: the number of arg
        #cfg["nlocals"]: the number of loacl var
        blockIdxToTokens = {}
        blockIdxToOpcodeCounts = {}
        blockIdxToOpcodeNum = {}
        for block in cfg['blocks']:
            totalblocknum += len(cfg['blocks'])
            try:
                disasm, block_bytes, annotations, tokens, numInsns, opcodeCounts = self.process_block(block)
            except:
                traceback.print_exc()
            my_cfg.add_node(block['offset'], asm=block_bytes, label=disasm)
            #acfg.add_node(block['offset'], features=annotations)
            #lstm_cfg.add_node(block['offset'], features= tokens)
            
            blockIdxToTokens[block['offset']] = tokens
            blockIdxToOpcodeCounts[block['offset']] = opcodeCounts
            blockIdxToOpcodeNum[block['offset']] = numInsns

        for block in cfg['blocks']:
            if 'jump' in block:
                if block['jump'] in my_cfg.nodes:
                    my_cfg.add_edge(block['offset'],block['jump'])
                    #acfg.add_edge(block['offset'], block['jump'])
                    #lstm_cfg.add_edge(block['offset'], block['jump'])
            if 'fail' in block:
                if block['fail'] in my_cfg.nodes:
                    my_cfg.add_edge(block['offset'],block['fail'])
                    #acfg.add_edge(block['offset'], block['fail'])
                    #lstm_cfg.add_edge(block['offset'], block['fail'])

        #between = nx.betweenness_centrality(acfg)
        '''for n in acfg.nodes(data=True):
            d = n[1]['features']
            d['offspring'] = len(nx.descendants(acfg, n[0]))
            d['betweenness'] = between[n[0]]
            n[1]['features'] = d'''

        return my_cfg, acfg, lstm_cfg, blockIdxToTokens, blockIdxToOpcodeCounts, blockIdxToOpcodeNum

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
            traceback.print_exc()
        return arch, bits

    def find_functions(self):
        self.r2.cmd('aaa')
        try:
            function_list = json.loads(self.r2.cmd('aflj'))
        except:
            function_list = []
            traceback.print_exc()
        return function_list

    def find_functions_by_symbols(self):
        self.r2.cmd('aa')
        try:
            symbols = json.loads(self.r2.cmd('isj'))
            fcn_symb = [s for s in symbols if s['type'] == 'FUNC']
        except:
            fcn_symb = []
            traceback.print_exc()
        return fcn_symb

    #root function
    def analyze(self):
        if self.use_symbol:
            function_list = self.find_functions_by_symbols()
        else:
            function_list = self.find_functions()

        result = {}
        binary_info = {}

        path = self.filename.split("/")

        if not os.path.isdir(os.path.join(self.outputDir, "edgelist", path[-2])):
            os.makedirs(os.path.join(self.outputDir, "edgelist", path[-2]))

        edgelistdir = os.path.join(self.outputDir, 'edgelist',path[-2], path[-1])
        with open(edgelistdir, 'w') as edgelistFile:
            for my_function in function_list:
                function_info = {}
                if self.use_symbol:
                    address = my_function['vaddr']
                else:
                    address = my_function['offset']

                try:
                    cfg, acfg, lstm_cfg, blockIdxToTokens, blockIdxToOpcodeCounts, blockIdxToOpcodeNum = self.function_to_cfg(my_function)
                    #result[my_function['name']] = {'cfg': cfg, "acfg": acfg, "lstm_cfg": lstm_cfg, "address": address}
                    
                    function_info["blockIdxToTokens"] = blockIdxToTokens
                    function_info["blockIdxToOpcodeCounts"] = blockIdxToOpcodeCounts
                    function_info["blockIdxToOpcodeNum"] = blockIdxToOpcodeNum
                    #function_info["block_num"] = len(blockIdxToOpcodeCounts)
                    binary_info[my_function['name']] = function_info

                    edgelist = list(cfg.edges)
                    for(src, tgt) in edgelist:
                        edgelistFile.write(str(src) + " " + str(tgt) + "\n")
                except Exception as e:
                    print("Error in functions: {} from {}".format(my_function['name'], self.filename))
                    traceback.print_exc()
                    pass

        
        #binary_article_info = {}
        #binary_article_info["opcode_list"] = p_opcode_list
        #binary_article_info["insToBlockCounts"] = insToBlockCounts
        #binary_article_info["extract_time"] = extract_time
        #with open(self.outputDir + "binary_info.json", 'w') as f:
        #    f.write(json.dumps(binary_article_info))
        
        #return result, binary_info
        return binary_info

    def close(self):
        self.r2.quit()

    def __exit__(self, exc_type, exc_value, traceback):
        self.r2.quit()





class BinaryArticleGeneration:

    def __init__(self, rootdir, outputDir):
        self.root_path = rootdir
        self.outputDir = outputDir

    #@staticmethod
    #def worker(item):
    #    BinaryArticleGeneration.analyze_file(item)
    #    return 0

    @staticmethod
    def extract_function(graph_analyzer):
        return graph_analyzer.extractAll()

    @staticmethod
    def to_jsongraph(graph):
        return json.dumps(json_graph.adjacency_data(graph))


    @staticmethod
    # return dictionary: index to token, reversed_dictionary: token to index
    def vocBuild(blockIdxToTokens):

        global opcode_idx_list
        global opcode_list
        global dictionary
        #vocabulary = []
        global reversed_dictionary
        #count = [['UNK'], -1]
        reversed_dictionary["UNK"] = 0
        index = len(reversed_dictionary)
        for idx in blockIdxToTokens:
            for token in blockIdxToTokens[idx]:
                #vocabulary.append(token)
                if token not in reversed_dictionary.keys():
                    reversed_dictionary[token] = index
                    if token in opcode_list and index not in opcode_idx_list:
                         opcode_idx_list.append(index)
                        # print("token:", token, " has idx: ", str(index))
                    index = index + 1
                    
        dictionary = dict(zip(reversed_dictionary.values(), reversed_dictionary.keys()))
        #ount.extend(collections.Counter(vocabulary).most_common(1000 - 1))
        #print('20 most common tokens: ', count[:20])

        #del vocabulary

        return 


    # generate article for word2vec. put all random walks together into one article.
    # we put a tag between blocks
    def articlesGen(walks, blockIdxToTokens, reversed_dictionary, outputDir, edgedir):
        # stores all the articles, each article itself is a list
        article = []
        articleinfo = {}
        # stores all the block boundary indice. blockBoundaryIndices[i] is a list to store indices for articles[i].
        # each item stores the index for the last token in the block
        blockBoundaryIdx = []
        for walk in walks:
            # one random walk is served as one article
            for idx in walk:
                if idx in blockIdxToTokens:
                    tokens = blockIdxToTokens[idx]
                    for token in tokens:
                        article.append(reversed_dictionary[token])
                blockBoundaryIdx.append(len(article) - 1)
                # aritcle.append(boundaryIdx)
        
        insnStartingIndices = []
        indexToCurrentInsnsStart = {}
        # blockEnd + 1 so that we can traverse to blockEnd
        # go through the current block to retrive instruction starting indices
        for i in range(0, len(article)): 
            if article[i] in opcode_idx_list:
                insnStartingIndices.append(i)
            indexToCurrentInsnsStart[i] = len(insnStartingIndices) - 1

        articleinfo["article"] = article
        articleinfo["blockBoundaryIdx"] = blockBoundaryIdx
        articleinfo["insnStartingIndices"] = insnStartingIndices
        articleinfo["indexToCurrentInsnsStart"] = indexToCurrentInsnsStart
        #articleinfo["totalBlockNum"] = totalBlockNum

        #print(articleinfo["totalBlockNum"])
        path = edgedir.split('/')
        articledir = os.path.join(outputDir, "article", path[-2], path[-1])
        if not os.path.isdir(os.path.join(outputDir, "article", path[-2])):
            os.makedirs(os.path.join(outputDir, "article", path[-2]))
        with open(articledir, 'w') as f:
            f.write(json.dumps(articleinfo))

        return 


    @staticmethod
    def analyze_file(outputDir, filename,use_symbol):
        #global pool_sem
        os.setpgrp()

        #filename = item[0]
        #outputDir = item[1]
        #use_symbol = item[2]
        global opcode_idx_list
        global dictionary
        global reversed_dictionary
        # this list contains all the opcode in the two binaries
        global opcode_list
        #IDF
        global insToBlockCounts
        global totalblocknum

        path = filename.split("/")
        print("#1: {0} --> {1} CFGs Start Generating!".format(path[-2], path[-1]))
        analyzer = RadareFunctionAnalyzer(filename, use_symbol, outputDir)
        #p = ThreadPool(1)
        #res = p.apply_async(analyzer.analyze, args = (opcode_list, insToBlockCounts))
        result = {}
        try:
            result = analyzer.analyze()
        except Exception as e:
                print("Aborting due to error:" + str(filename))
                traceback.print_exc()
        
        print("#2: vocabulary buildup")
        blockIdxToTokensA = {}
        #totalBlockNum = {}
        for index in result:
            func = result[index]
            #totalBlockNum[index] = func["block_num"]
            for block in func["blockIdxToTokens"]:
                blockIdxToTokensA[block] =  func["blockIdxToTokens"][block]

        BinaryArticleGeneration.vocBuild(blockIdxToTokensA)

        print("#3: enerate random walks")
        
        EDGELIST_FILE = os.path.join(outputDir, "edgelist", path[-2], path[-1])
        walks = deepwalk.randomWalksGen(EDGELIST_FILE)
        #print(walks)
        #print(blockIdxToTokensA)
     
        # step 4: generate articles based on random walks
        print("#4: generate articles based on random walks")
        BinaryArticleGeneration.articlesGen(walks, blockIdxToTokensA, reversed_dictionary, outputDir, EDGELIST_FILE)

        with open(outputDir + "word2vec.json",'w') as f:
            f.write(json.dumps(reversed_dictionary.copy()))
        block_info = {}
        block_info["opcode_idx_list"] = opcode_idx_list
        block_info["opcode_list"] = opcode_list
        block_info["dictionary"] = dictionary
        block_info["reversed_dictionary"] = reversed_dictionary
        block_info["insToBlockCounts"] = insToBlockCounts
        block_info["totalblocknum"] = totalblocknum
        
        with open(outputDir + "block_info.json","w") as f:
            f.write(json.dumps(block_info))
        #print("opcode_idx_list: ", opcode_idx_list)
        #print("dictionary: ", dictionary)
        #print("reversed_dictionary: ", reversed_dictionary)
        print("#5: {0} --> {1} articles generatied".format(path[-2], path[-1]))
        print("----------------")

        #analyzer.close()
        return 0


    # Scan the root directory to find all the file to analyze,
    # query also the db for already analyzed files.
    def scan_for_file(self, start):
        file_list = []
        # Scan recursively all the subdirectory
        directories = os.listdir(start)
        for item in directories:
            item = os.path.join(start,item)
            if os.path.isdir(item):
                file_list.extend(self.scan_for_file(item + os.sep))
            elif os.path.isfile(item):
                file_list.append(item)
        return file_list


    # if statu is ture, the value of variable is init from file
    def global_var_init(self, statu = False):
        
        global opcode_idx_list
        global dictionary
        global reversed_dictionary
        # this list contains all the opcode in the two binaries
        global opcode_list
        #IDF
        global insToBlockCounts
        global totalblocknum
        if statu:
            with open(self.outputDir + "block_info.json", 'r') as f:
                line = f.readline()
                block_info = json.loads(line)

                opcode_idx_list = block_info["opcode_idx_list"]
                dictionary  = block_info["dictionary"]
                reversed_dictionary = block_info["reversed_dictionary"]
                opcode_list = block_info["opcode_list"]
                insToBlockCounts = block_info["insToBlockCounts"]
                totalblocknum = block_info["totalblocknum"]
        else:
            pass
    @staticmethod
    def remove_repetition_binary(outputDir, filelist):
        binary_list = []
        for filename in filelist:
            path = filename.split("/")
            articledir = os.path.join(outputDir, "article", path[-2], path[-1])
            if os.path.isfile(articledir):
                print("{0} -> {1} haved Generated".format(path[-2], path[-1]))
            else:
                binary_list.append(filename)
        return binary_list

    # root function to create the db
    def generate_article(self, use_symbol):
        #global pool_sem

        #pool_sem = multiprocessing.BoundedSemaphore(value=1)

        #self.create_db()
        file_list = self.scan_for_file(self.root_path)

        print('Found ' + str(len(file_list)) + ' during the scan')
        #file_list = self.remove_override(file_list)
        print('Find ' + str(len(file_list)) + ' files to analyze')

        remove_generated_file_list = self.remove_repetition_binary(self.outputDir, file_list)

        print('Have ' + str(len(remove_generated_file_list)) + ' files need to analyze')
        random.shuffle(remove_generated_file_list)
        self.global_var_init(True)
        #t_args = [(f, self.outputDir, use_symbol, opcode_idx_list, dictionary, reversed_dictionary, opcode_list, insToBlockCounts) for f in file_list]

        # Start a parallel pool to analyze files
        #p = Pool(processes=None, maxtasksperchild=10)
        for file in tqdm(remove_generated_file_list, total=len(remove_generated_file_list)):
            print(file)
            result = BinaryArticleGeneration.analyze_file(self.outputDir, file, use_symbol)

        return 0
        #p.close()
        #p.join()





