import click
import json
import networkx as nx
import numpy as np
import os
import re

from collections import Counter
from collections import defaultdict
from scipy.sparse import coo_matrix
from tqdm import tqdm

reg64 = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rsp', 'rbp']
reg2idx = {'al': 0,  'bl': 1,  'cl': 2,  'dl': 3,  'sil': 4,  'dil': 5,  'spl': 6,  'bpl': 7, 
           'ah': 0,  'bh': 1,  'ch': 2,  'dh': 3,  'sih': 4,  'dih': 5,  'sph': 6,  'bph': 7,
           'ax': 0,  'bx': 1,  'cx': 2,  'dx': 3,  'si': 4,  'di': 5,  'sp': 6,  'bp': 7,
           'eax': 0,  'ebx': 1,  'ecx': 2,  'edx': 3,  'esi': 4,  'edi': 5,  'esp': 6,  'ebp': 7}

def process_defuse(reg_list):
    # ['ebx.4', 'rbp.8,rdi.8,ds.2', 'rbp.8', 'rdi.8', 'ebx.4,rdi.8,ds.2']
    # ['sp+2C8.C', 'sp+68.8']
    # ['', '', 'sss']
    res = []
    for s in reg_list:
        inst_reg = []
        if s == '':
            res.append(list())
            continue
        regs = s.split(',')

        for reg in regs:
            reg = re.sub(r'\.[0-9A-Fa-f]+', '', reg)
            if '+' in reg:
                regtmp = re.sub(r'\+.*', '', reg)
            else:
                regtmp = reg
            if regtmp in reg2idx:
                regafter = reg64[reg2idx[regtmp]]
                reg = reg.replace(regtmp, regafter)
            inst_reg.append(reg)

        res.append(inst_reg)
    return res

def process_insts(insts_list):
    ret = []
    #['add    rbx7.8, #1.8, rbx7.8{6}', 'jg     [ds.2:(r14_6.8{3}+#8.8)].4, rbx7.4{6}, @8']
    # ['jge eax1@1.1, #0.1, @2']
    for op in insts_list:
        op = re.sub(r'\s+', ' ', op) # space
        opcode = op.split(', ')[0].split(' ')[0]
        operand1 = op.split(', ')[0][len(opcode) + 1:]
        operands = [operand1] + op.split(', ')[1:]
        n = len(operands)
        res = ""
        isjmp = 0
        if 'j' == opcode[0] and '@' in operands[-1]:
            # opcode = re.sub(r'j.*?', 'j', opcode)
            opcode = 'j'
            isjmp = 1
        res += opcode
        if isjmp == 1:
            n -= 1
        for i in range(n):
            operand = operands[i]
            operand = re.sub(r'\{.*?\}', '', operand) # {}ÖÐÈ¥µô
            operand = re.sub(r'\.[0-9A-Fa-f]+', '', operand) # .
            operand = re.sub(r'\@[0-9A-Fa-f]+', '', operand) # @
            # operand = re.sub(r'(?<=\w)_\w+', '', operand) # _
            # operand = re.sub(r'#(?:0[xX][0-9a-fA-F]+|\d+)', 'constant', operand) # #constant
            # operand = re.sub(r'var[0-9A-Fa-f]+', 'var', operand) # var
            # operand = re.sub(r'r[0-9A-Fa-f]+', 'reg', operand) # reg
            # if '<' in operand and '>' in operand:
            #     operand = re.sub(r'\<.*?\>', '', operand) # <>È¥µô
            if i == 0:
                res += " "
            else:
                res += ', '
            res += operand
                
        if isjmp == 1:
            res += ", @"
        ret.append(res)
    return ret

def traverse(G, succeed, depth, tmp_def, nodelist, has_visited, strandaddr2inputs, strandaddr2outputs, fva):
    if depth > 20:
        return
    if succeed in has_visited:
        return
    if len(tmp_def) == 0:
        return
    has_visited.add(succeed)
    def_after = set() # the next tmp_def
    uses_node2 = set(strandaddr2inputs[succeed])
    defs_node2 = set(strandaddr2outputs[succeed])
    tmp_def = set(tmp_def)
    if len(tmp_def & uses_node2) != 0:
        nodelist.add(succeed)
    for reg in tmp_def:
        # not defined again
        if reg not in defs_node2:
            def_after.add(reg)
        
    for node in G.successors(succeed):
        traverse(G, node, depth+1, def_after, nodelist, has_visited, strandaddr2inputs, strandaddr2outputs, fva)



def get_top_opcodes(input_folder, num_opc):
    """
    Extract the list of most frequent opcodes across the training data.

    Args:
        input_folder: a folder with JSON files from IDA_acfg_disasm
        num_opc: the number of most frequent opcodes to select.

    Return
        dict: map most common opcodes to their ranking.
    """
    opc_cnt = Counter()

    for f_json in tqdm(os.listdir(input_folder)):
        
        if not f_json.endswith(".json"):
            continue

        json_path = os.path.join(input_folder, f_json)
        with open(json_path) as f_in:
            jj = json.load(f_in)

            idb_path = list(jj.keys())[0]
            # print("[D] Processing: {}".format(idb_path))
            j_data = jj[idb_path]
            del j_data['arch']

            # Iterate over each function
            for fva in j_data:
                fva_data = j_data[fva]
                # Iterate over each basic-block
                for bb in fva_data['basic_blocks']:
                    opcodes = fva_data['basic_blocks'][bb]['opcode']
                    opcodes_update = []
                    for opcode in opcodes:
                        if opcode == 'goto':
                            continue
                        if 'j' == opcode[0] and 'jmp' not in opcode:
                            opcode = "j"
                        opcodes_update.append(opcode)    
    
                    opc_cnt.update(opcodes_update)


    print("[D] Found: {} mnemonics.".format(len(opc_cnt.keys())))
    print("[D] Top 10 mnemonics: {}".format(opc_cnt.most_common(10)))
    return {d[0]: c for c, d in enumerate(opc_cnt.most_common(num_opc))}


def create_graph(nodes, edges, bb_infos, fva, jsonfile):
    
    G_info = dict()
    nodes_not_have_info = []
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(str(node))
    for edge in edges:
        G.add_edge(str(edge[0]), str(edge[1]))
    
    # if [fva, fva] in G.edges():
    #     G.remove_edge(fva, fva)

    end_block = set()
    start_block = set()
    without_union = set()
    # start_block.add(fva)
    for bb_add in bb_infos.keys():
        bb_info = bb_infos[bb_add]
        insts = bb_info['inst']
        insts = process_insts(insts)
        opcodes = bb_info['opcode']
        heads = bb_info['bb_heads']
        is_null = 0

        # last block
        if 'fffffff' in hex(int(bb_add)) and len(heads) == 0:
            # is_null = 1
            # for end in list(G.predecessors(bb_add)):
            #     end_block.add(end)
            # G.remove_node(bb_add)
            # nodes_not_have_info.append(bb_add)
            pres = list(G.predecessors(bb_add))
            for pre in pres:
                if len(bb_infos[pre]['bb_heads']) == 1 and bb_infos[pre]['opcode'] == 'goto':
                    for end in list(G.predecessors(pre)):
                        end_block.add(end)
                        # if len(list(G.successors(end))) == 1:
                        #     without_union.add(end)
                            
                else:
                    end_block.add(pre)
                    # if len(list(G.successors(pre))) == 1:
                    #     without_union.add(pre)
            G.remove_node(bb_add)
            nodes_not_have_info.append(bb_add)
            continue

        # first block the same as the first useful block, and the first block has only 'goto'
        if fva == bb_add and len(heads) == 1 and opcodes[-1] == 'goto' and [bb_add, bb_add] in G.edges():
            G.remove_edge(bb_add, bb_add)
            for succ in list(G.successors(bb_add)):
                start_block.add(succ)
            G.remove_node(bb_add)
            nodes_not_have_info.append(bb_add)
            continue

        # first block the same as the first useful block, and the first block is useful
        if fva == bb_add and [bb_add, bb_add] in G.edges():
            G.remove_edge(bb_add, bb_add)
            start_block.add(bb_add)
            G_info[bb_add] = insts
            # if len(list(G.successors(bb_add))) == 1:
            #     without_union.add(bb_add)
            continue

        # first block and not the same as the first useful block or the xtrn block seen as the end block
        if len(heads) == 0:
            if len(list(G.successors(bb_add))) != 0:
                firstblock = list(G.successors(bb_add))[0]
                next_bb_info = bb_infos[firstblock]
                next_opcodes = next_bb_info['opcode']
                next_heads = next_bb_info['bb_heads']
                # only goto
                if len(next_heads) == 1 and next_opcodes[-1] == 'goto':
                    for start in list(G.successors(firstblock)):
                        start_block.add(start)
                    # G.remove_node(firstblock)
                    G.remove_node(bb_add)
                    nodes_not_have_info.append(bb_add)
                    # nodes_not_have_info.append(firstblock)
                    
                # useful
                else:
                    start_block.add(firstblock)
                    G.remove_node(bb_add)
                    nodes_not_have_info.append(bb_add)          
                continue
            # xtrn block
            if len(list(G.predecessors(bb_add))) != 0:
                pres = list(G.predecessors(bb_add))
                for pre in pres:
                    if len(bb_infos[pre]['bb_heads']) == 1 and bb_infos[pre]['opcode'][-1] == 'goto':
                        for end in list(G.predecessors(pre)):
                            end_block.add(end)
                    else:
                        end_block.add(pre)
                G.remove_node(bb_add)
                nodes_not_have_info.append(bb_add)
                continue


        # if len(heads) == 0 and len(list(G.successors(bb_add))) == 0:
        #     pres = list(G.predecessors(bb_add))
        #     for pre in pres:
        #         end_block.add(pre)
        #     G.remove_node(bb_add)
        #     nodes_not_have_info.append(bb_add)
        #     continue

        if len(heads) == 1 and opcodes[-1] == 'goto':
            # is_null = 1
        # if is_null == 1:
            pres = list(G.predecessors(bb_add))
            succs = list(G.successors(bb_add))
            for pre in pres:
                for succ in succs:
                    G.add_edge(pre, succ)
            G.remove_node(bb_add)
            nodes_not_have_info.append(bb_add)
            continue
            
        if [bb_add, bb_add] in G.edges() and len(list(G.successors(bb_add))) == 1:
            end_block.add(bb_add)
            
        G_info[bb_add] = insts

    assert len(G_info.keys()) == len(G.nodes()), "graph info wrong, {}, {}".format(fva, jsonfile) 
    
    # nx.draw(G, with_labels=True)

    for node in G.nodes():
        if G.out_degree(node) == 0:
            end_block.add(node)
        if G.in_degree(node) == 0:
            start_block.add(node)
            
    # get_strand_dot(G, G_info, 'test.dot', 'strand')
    
              


    bb2strand = dict()
    strandaddr2inst = dict()
    strandaddr2inputs = dict()
    strandaddr2outputs = dict()
    strandaddr2bb = dict()
    inst2strandaddr = defaultdict(list)
    has_cond_jmp = dict() # bb_add -> jmp strand add

    # for bb_add in bb_infos.keys():
    for bb_add in G.nodes():
        inst2firstused = dict()
        bb_info = bb_infos[bb_add]
        insts = bb_info['inst']
        insts = process_insts(insts)
        opcodes = bb_info['opcode']
        uses = process_defuse(bb_info['use'])
        defs = process_defuse(bb_info['def'])
        heads = bb_info['bb_heads']
        # assert len(opcodes) >= 1, "graph info wrong, {}, {}".format(fva, jsonfile) 
        if len(opcodes) < 1:
            return None, None, None, None, None
            
        if opcodes[-1] == 'goto':
            opcodes.pop()
            insts.pop()
            uses.pop()
            defs.pop()
            heads.pop()

        bb2strand[bb_add] = []
        if len(heads) == 1:
            add = str(heads[0])
            strandaddr2inst[add] = insts      
            strandaddr2inputs[add] = set(uses[0])
            strandaddr2outputs[add] = set(defs[0])          
            if 'j' in insts[0] and '@' == insts[0][-1]:
                has_cond_jmp[bb_add] = add
            inst2strandaddr[insts[0]].append(add)
            bb2strand[bb_add].append(add)
            strandaddr2bb[add] = bb_add
            continue
        # if bb_add in without_union:
        #     add = str(heads[0])
        #     strandaddr2inst[add] = insts                
        #     if 'j' in insts[-1] and '@' == insts[-1][-1]:
        #         has_cond_jmp[bb_add] = add
        #     # inst2strandaddr[insts[0]].append(add)
        #     bb2strand[bb_add].append(add)
        #     strandaddr2bb[add] = bb_add
        #     continue
        unusedInsts = list(np.arange(0, len(insts)))
        strands = list()
        # allInputs = list()
        # allOutputs = list()
        while(unusedInsts):
            start_idx = unusedInsts.pop()
            inst2firstused[start_idx] = start_idx
            min_idx = start_idx
            use = set(uses[start_idx])
            defi = set(defs[start_idx])
            # newStrand = [insts[start_idx]]
            newStrand = [start_idx]
            has_jmp = 0
            for i in range(start_idx - 1, -1, -1):
                defi_i = defs[i]
                needed = set(defi_i) & use
                if len(needed) != 0:
                    inst2firstused[i] = start_idx
                    # newStrand.append(insts[i])
                    newStrand.append(i)
                    use_i = set(uses[i])
                    use = use.union(use_i)
                    defi = defi.union(needed)
                    if i in unusedInsts:
                        min_idx = min(min_idx, i)
                        unusedInsts.remove(i)
            newStrand.reverse()
            for inst in newStrand:
                if 'j' in insts[inst] and '@' == insts[inst][-1]:
                    has_jmp = 1
            strands.append(newStrand)
            # allInputs.append(use)
            # allOutputs.append(defi)

        for i in range(len(strands)):
            strand = strands[i]
            remove = []
            for inst_idx in strand:
                if inst2firstused[inst_idx] != strand[-1]:
                    remove.append(inst_idx)
            new = [x for x in strand if x not in remove]
            strands[i] = new

        for i in range(len(strands)):
            strand = strands[i]
            newStrand = list()
            usevar = set()
            defvar = set()
            strandkey = ""
            for idx in strand:
                newStrand.append(insts[idx])
                usevar.update(uses[idx])
                defvar.update(defs[idx])
                strandkey += insts[idx]
            min_idx = strand[0]

            strandaddr2inst[str(heads[min_idx])] = newStrand
            strandaddr2inputs[str(heads[min_idx])] = usevar
            strandaddr2outputs[str(heads[min_idx])] = defvar
            inst2strandaddr[strandkey].append(str(heads[min_idx]))
            bb2strand[bb_add].append(str(heads[min_idx]))
            strandaddr2bb[str(heads[min_idx])] = bb_add
            if has_jmp == 1:
                has_cond_jmp[bb_add] = str(heads[min_idx])

        bb2strand[bb_add].reverse()

    assert len(bb2strand) == len(G.nodes())

    strand_G = nx.DiGraph()

    for node in strandaddr2inst.keys():
        strand_G.add_node(node)

    # consider strands in a block having orders
    # strands edges intra blocks
    bb2firstStrand = dict()
    bb2lastStrand = dict()
    for bb_add, strands in bb2strand.items():
        bb2firstStrand[bb_add] = strands[0]
        bb2lastStrand[bb_add] = strands[-1]
        if len(strands) == 1:
            continue
        for strandIdx in range(len(strands) - 1):
            strand_G.add_edge(strands[strandIdx], strands[strandIdx + 1])

    assert len(bb2firstStrand) == len(G.nodes())
    assert len(bb2lastStrand) == len(G.nodes())

    # strands edges inter blocks
    for edge in G.edges():
        node1 = str(edge[0])
        node2 = str(edge[1])
        assert bb2lastStrand[node1] in strandaddr2inst.keys()
        assert bb2firstStrand[node2] in strandaddr2inst.keys()
        strand_G.add_edge(bb2lastStrand[node1], bb2firstStrand[node2])

    RET_node = set()
    START_node = set()
    for end in end_block:
        # assert end in bb2lastStrand.keys(), "graph info wrong, {}, {}".format(fva, jsonfile)
        if end not in bb2lastStrand.keys():
            return None, None, None, None, None
        RET_node.add(bb2lastStrand[end])
    for start in start_block:
        # assert start in bb2firstStrand.keys(), "graph info wrong, {}, {}".format(fva, jsonfile)
        if start not in bb2firstStrand.keys():
            return None, None, None, None, None
        START_node.add(bb2firstStrand[start])
    # assert len(START_node) > 0, "graph info wrong, {}, {}".format(fva, jsonfile)
    if len(START_node) == 0:
        return None, None, None, None, None
    # assert len(RET_node) > 0, "graph info wrong, {}, {}".format(fva, jsonfile)
    if len(START_node) == 0:
        return None, None, None, None, None

    # repeatedStrand = set()
    # for inst, strandaddrs in inst2strandaddr.items():
    #     if len(strandaddrs) > 1:
    #         for i in range(len(strandaddrs) - 1):
    #             for j in range(i + 1, len(strandaddrs)):
    #                 first = strandaddrs[i]
    #                 second = strandaddrs[j]
    #                 succ1 = set(strand_G.successors(first))
    #                 succ2 = set(strand_G.successors(second))
    #                 if succ1 != succ2:
    #                     continue
    #                 repeatedStrand.add(first)
    #                 RET_node.discard(first)
    #                 START_node.discard(first)
    #                 pres = list(strand_G.predecessors(first))
    #                 for pre in pres:
    #                     strand_G.add_edge(pre, second)


    # for strand in repeatedStrand:
    #     strandaddr2inst.pop(strand, None)
    #     strandaddr2inputs.pop(strand, None)
    #     strandaddr2outputs.pop(strand, None)
    #     strand_G.remove_node(strand)

    assert len(strand_G.nodes()) == len(strandaddr2inst.keys()), "repeat remove wrong, {}, {}".format(fva, jsonfile) 
    assert len(strand_G.nodes()) == len(strandaddr2inputs.keys()), "repeat remove wrong, {}, {}".format(fva, jsonfile) 
    assert len(strand_G.nodes()) == len(strandaddr2outputs.keys()), "repeat remove wrong, {}, {}".format(fva, jsonfile) 
    

    DDG = nx.DiGraph()
    for node in strand_G.nodes():
        DDG.add_node(node)

    for node in DDG.nodes():
        tmp_def = set(strandaddr2outputs[node])

        if len(tmp_def) == 0:
            continue

        tmp_node_list = set()
        has_visited = set({node})
        for succeed in strand_G.successors(node):
            traverse(strand_G, succeed, 1, tmp_def, tmp_node_list, has_visited, strandaddr2inputs, strandaddr2outputs, fva)

        for succeed in tmp_node_list:
            DDG.add_edge(node, succeed)
    
    
    CFG = strand_G.copy()
    for node in RET_node:
        CFG.add_edge(node, 'END')
    for node in START_node:
        CFG.add_edge('START', node)
    CFG.add_edge('0', 'END')
    CFG.add_edge('0', 'START')
    RevG = CFG.reverse()
    post_doms = nx.immediate_dominators(RevG, 'END')
    nodes_not_in_post = [i for i in CFG.nodes() if i not in post_doms.keys()]
    # nodes_hex = [hex(i) for i in nodes_not_in_post]
    # assert len(nodes_not_in_post) == 0, "graph info wrong, {}, {}".format(fva, jsonfile) 
    if len(nodes_not_in_post) != 0:
        print('The nodes not in post_doms are: {}, {}, {}'.format(nodes_not_in_post, fva, jsonfile))
        return None, None, None, None, None
        # CFG.remove_nodes_from(nodes_not_in_post)
    
    
    S = []
    for edge in CFG.edges():
        nodeA = edge[0]
        nodeB = edge[1]
        if nodeA not in post_doms.keys() or nodeB == post_doms[nodeA]:
            continue
        S.append(edge)
        
    CD_record = set()
    dependence_record = set()
    for pair in S:
        # print('processing pair <{}, {}>'.format(pair[0], pair[1]))
        nodeB = pair[1]
        nodeA = pair[0]
        parent = post_doms[nodeB]
        tmp_control_dep = list()
        tmp_control_dep.append(nodeB)
        # print(post_doms[nodeA])
        while parent != post_doms[nodeA] and parent != nodeA:
            # print(parent)
            tmp_control_dep.append(parent)
            parent = post_doms[parent]
        for i in tmp_control_dep:
            tmp_list = tuple([nodeA, i])
            CD_record.add(tmp_list)
            # dependence_record.add(i)
    CD_record.remove(('0', 'START'))
    CD_record = list(CD_record)
    
    CDG = nx.DiGraph()
    # print(edges)
    for edge in CD_record:
        CDG.add_edge(edge[0], edge[1])
        
    if 'END' in CDG.nodes():
        CDG.remove_node('END')
    if 'START' in CDG.nodes():
        CDG.remove_node('START')
        
    nodelist_cdg = sorted(list(CDG.nodes()))
    nodelist = sorted(list(strand_G.nodes()))
    assert len(strand_G.nodes()) == len(strandaddr2inst.keys()), "CFG nodes number wrong, {}, {}".format(fva, jsonfile)
    strandaddr2inst['0'] = []
    if len(CDG.nodes()) != len(strandaddr2inst.keys()):
        return None, None, None, None, None
    CFG_adj_mat = nx.to_numpy_matrix(strand_G, nodelist=nodelist, dtype=np.int8)
    CDG_adj_mat = nx.to_numpy_matrix(CDG, nodelist=nodelist_cdg, dtype=np.int8)
    DDG_adj_mat = nx.to_numpy_matrix(DDG, nodelist=nodelist, dtype=np.int8)
    # t = (nx.to_numpy_matrix(G, nodelist=[0, 1, 2, 3], dtype=np.int8)[1:])[:, 1:]  
    

    return nodelist, nodelist_cdg, CFG_adj_mat, CDG_adj_mat, DDG_adj_mat, strandaddr2inst


def create_features_matrix(node_list, strandaddr2inst, opc_dict):
    """
    Create the matrix with numerical features.

    Args:
        node_list: list of basic-blocks addresses
        fva_data: dict with features associated to a function
        opc_dict: selected opcodes.

    Return
        np.matrix: Numpy matrix with selected features.
    """
    f_mat = np.zeros((len(node_list), len(opc_dict)))
    # Iterate over each BBs
    for node_idx, node_fva in enumerate(node_list):
        node_data = strandaddr2inst[node_fva]
        for inst in node_data:
            opcode = inst.split(' ')[0]
            if opcode in opc_dict:
                opcode_idx = opc_dict[opcode]
                f_mat[node_idx][opcode_idx] += 1
                
    # f_mat = np.ones((len(node_list), 1))
    
    # WARNING
    # Forcing the type to np.int8 to limit memory usage.
    #   Use the same when parsing the data!
    return f_mat.astype(np.int8)


def np_to_scipy_sparse(np_mat):
    """
    Convert the Numpy matrix in input to a Scipy sparse matrix.

    Args:
        np_mat: a Numpy matrix

    Return
        str: serialized matrix
    """
    cmat = coo_matrix(np_mat)
    # Custom string serialization
    row_str = ';'.join([str(x) for x in cmat.row])
    col_str = ';'.join([str(x) for x in cmat.col])
    data_str = ';'.join([str(x) for x in cmat.data])
    n_row = str(np_mat.shape[0])
    n_col = str(np_mat.shape[1])
    mat_str = "::".join([row_str, col_str, data_str, n_row, n_col])
    return mat_str




def create_functions_dict(input_folder, opc_dict):
    """
    Convert each function into a graph with BB-level features.

    Args:
        input_folder: a folder with JSON files from IDA_acfg_disasm
        opc_dict: dictionary that maps most common opcodes to their ranking.

    Return
        dict: map each function to a graph and features matrix
    """
    # try:
    functions_dict = defaultdict(dict)
    functions_dict2 = defaultdict(dict)
    ind = []
    for index, f_json in enumerate(tqdm(os.listdir(input_folder))):
        # if index < 89:
        #     continue
        
        if not f_json.endswith(".json"):
            continue

        json_path = os.path.join(input_folder, f_json)
        # json_path = '../../../DBs/Dataset-1-new/features/testing/acfg_microcode_Dataset-1_testing/x64-gcc-7-O1_openssl_acfg_microcode.json'
        with open(json_path) as f_in:
            error = []
            print(json_path)
            jj = json.load(f_in)

            idb_path = list(jj.keys())[0]
            # print("[D] Processing: {}".format(idb_path))
            j_data = jj[idb_path]
            arch = j_data['arch']
            del j_data['arch']

            # Iterate over each function
            for fva in tqdm(j_data):
                # fva = '1797624'
                fva_data = j_data[fva]  
                nodes, nodes_cdg, CFG_adj_mat, CDG_adj_mat, DDG_adj_mat, strandaddr2inst = create_graph(
                    fva_data['nodes'], fva_data['edges'], fva_data['basic_blocks'], fva, json_path)
                if nodes == None:
                    ind.append(index)
                    print(index)
                    continue
                f_mat_cdg = create_features_matrix(
                    nodes_cdg, strandaddr2inst, opc_dict)
                f_mat_cfg = create_features_matrix(
                    nodes, strandaddr2inst, opc_dict)
                functions_dict[idb_path][hex(int(fva))] = {
                    'cfg': np_to_scipy_sparse(CFG_adj_mat),
                    'cdg': np_to_scipy_sparse(CDG_adj_mat),
                    'ddg': np_to_scipy_sparse(DDG_adj_mat),
                    'opc_cdg': np_to_scipy_sparse(f_mat_cdg),
                    'opc': np_to_scipy_sparse(f_mat_cfg),
                }
                strandaddr2inst.pop('0')
                functions_dict2[idb_path][hex(int(fva))] = {
                    'cfg': np_to_scipy_sparse(CFG_adj_mat),
                    'ddg': np_to_scipy_sparse(DDG_adj_mat),
                    'opc': np_to_scipy_sparse(f_mat_cfg),
                    'cdg': np_to_scipy_sparse(CDG_adj_mat),
                    'opc_cdg': np_to_scipy_sparse(f_mat_cdg),
                    'strandinfo': strandaddr2inst
                }
                # break
        # break

    print(ind)
            
    return functions_dict, functions_dict2

    # except Exception as e:
    #     print("[!] Exception in create_functions_dict\n{}".format(e))
    #     return dict()


@click.command()
@click.option('-i', '--input-dir', required=True,
              help='IDA_acfg_disasm JSON files.')
@click.option('--training', is_flag=True,
              help='Process training data')
@click.option('-n', '--num-opcodes',
              default=64,
              help='Number of most frequent opcodes.')
@click.option('-d', '--opcodes-json',
              default="opcodes_dict.json",
              help='JSON with selected opcodes.')
@click.option('-o', '--output-dir', required=True,
              help='Output directory.')
def main(input_dir, training, num_opcodes, opcodes_json, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if training:
        opc_dict = get_top_opcodes(input_dir, num_opcodes)
        output_path = os.path.join(output_dir, opcodes_json)
        with open(output_path, "w") as f_out:
            json.dump(opc_dict, f_out)
    else:
        if not os.path.isfile(opcodes_json):
            print("[!] Error loading {}".format(opcodes_json))
            return
        with open(opcodes_json) as f_in:
            opc_dict = json.load(f_in)

    if not training and num_opcodes > len(opc_dict):
        print("[!] Num opcodes is greater than training ({} > {})".format(
            num_opcodes, len(opc_dict)))
        return

    o_dict, o_dict2 = create_functions_dict(input_dir, opc_dict)
    o_json = "cfg_cdg_ddg_opc.json"
    o_json2 = "cfg_cdg_ddg_strand.json"
    output_path = os.path.join(output_dir, o_json)
    with open(output_path, 'w') as f_out:
        json.dump(o_dict, f_out)
        
    output_path = os.path.join(output_dir, o_json2)
    with open(output_path, 'w') as f_out:
        json.dump(o_dict2, f_out)


if __name__ == '__main__':
    main()
