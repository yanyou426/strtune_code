from genericpath import isfile
import idc
import idaapi
import ida_pro
import idautils
import ida_funcs
import time
import ida_hexrays as hr
import ida_lines
import ida_segment
import os
import json
from pathlib import Path
import ntpath


class printer_t(hr.vd_printer_t):
    """Converts microcode output to an array of strings."""
    def __init__(self, *args):
        hr.vd_printer_t.__init__(self)
        self.mc = []

    def get_mc(self):
        return self.mc

    def _print(self, indent, line):
        self.mc.append(line)
        return 1


def get_function_segment_name(function_ea):
    segment = ida_segment.getseg(function_ea)
    if segment:
        return ida_segment.get_segm_name(segment)
    return None


def get_architecture():
    arch_id = idaapi.ph_get_id()
    if arch_id == idaapi.PLFM_386:
        return "x86"
    elif arch_id == idaapi.PLFM_ARM:
        return "ARM"
    elif arch_id == idaapi.PLFM_MIPS:
        return "MIPS"
    else:
        return "Unknown"


def analyze_bb(mblock):

    bb_dict = {}
    opcode_list = []
    inst_list = []
    def_list = []
    use_list = []
    bb_heads_list = []

    curr = mblock.head

    while True:
        if not curr:
            break
        minsn = curr
        opcode = ida_lines.tag_remove(minsn._print()).split(' ')[0]
        inst = ida_lines.tag_remove(minsn._print())
        mdef = mblock.build_def_list(minsn, hr.MUST_ACCESS)
        muse = mblock.build_use_list(minsn, hr.MUST_ACCESS)
        bb_head = minsn.ea

        opcode_list.append(opcode)
        inst_list.append(inst)
        def_list.append(mdef._print())
        use_list.append(muse._print())
        bb_heads_list.append(bb_head)

        curr = curr.next

    bb_dict['bb_len'] = mblock.end - mblock.start
    bb_dict['opcode'] = opcode_list
    bb_dict['inst'] = inst_list
    bb_dict['def'] = def_list
    bb_dict['use'] = use_list
    bb_dict['bb_heads'] = bb_heads_list

    return bb_dict


def analyze_func(func_ea):
    start_time = time.time()

    func_dict = {}

    pfn = ida_funcs.get_func(func_ea)
    mbr = hr.mba_ranges_t(pfn)
    hf = hr.hexrays_failure_t()
    ml = hr.mlist_t()

    mba = hr.gen_microcode(mbr, hf, ml, hr.DECOMP_WARNINGS, hr.MMAT_LVARS)

    if not mba:
        return None

    vp = printer_t()
    mba.set_mba_flags(mba.get_mba_flags() | hr.MBA_SHORT)
    mba._print(vp)

    nodes_list = []
    edges_list = []
    qty = mba.qty
    for src in range(0, qty):
        mblock = mba.get_mblock(src)
        nodes_list.append(mblock.start)

        for dest in mblock.succset:
            # edges_list.append('test')
            # edges_list.append(dest)
            edges_list.append([mblock.start, mba.get_mblock(dest).start])

    bbs_dict = {}
    for _id in range(qty):
        mblock = mba.get_mblock(_id)
        bb_dict = analyze_bb(mblock)
        bbs_dict[mblock.start] = bb_dict

    end_time = time.time()

    func_dict['elapsed_time'] = end_time - start_time
    func_dict['nodes'] = nodes_list
    func_dict['edges'] = edges_list
    func_dict['basic_blocks'] = bbs_dict

    return func_dict


def analyze_binary_file():

    binary_dict = {}
    binary_dict['arch'] = get_architecture()

    for func_ea in idautils.Functions():
        if get_function_segment_name(func_ea) != '.text':
            continue
        func_dict = analyze_func(func_ea)
        if not func_dict:
            continue
        binary_dict[func_ea] = func_dict
        

    return binary_dict


if __name__ == '__main__':

    if not idaapi.get_plugin_options("json"):
        print("[!] -Ojson option is missing")
        ida_pro.qexit(1)

    plugin_options = idaapi.get_plugin_options("json").split(";")
    if len(plugin_options) != 2:
        print("[!] -Ojson is required")
        ida_pro.qexit(1)
    JSON_FOLDER_PATH = plugin_options[0]
    idb_path = plugin_options[1]
    binary_file_dict = analyze_binary_file()

    input_file_path = idc.get_input_file_path()
    input_file_name = input_file_path.split('\\')[-1]
    JSON_PATH = os.path.join(JSON_FOLDER_PATH, input_file_name + '_acfg_microcode.json')

    # out_name = ntpath.basename(input_file_path.replace(".i64", "_acfg_microcode.json"))
    # print(out_name)
    idb_index = idb_path.lower().find("idbs")
    idb_path = idb_path[idb_index:]
    idb_path = idb_path.replace("\\", "/")

    with open(JSON_PATH, "w") as f:
        json.dump({idb_path: binary_file_dict}, f, indent=4)

    ida_pro.qexit(0)