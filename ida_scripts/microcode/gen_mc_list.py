import os, shutil, errno, re
import idc
from pathlib import Path
import ida_idaapi
import ida_bytes
import ida_range
import ida_kernwin as kw
import ida_hexrays as hr
import ida_funcs
import ida_diskio
import ida_ida
import ida_graph
import ida_lines
import ida_moves
import idautils
import idaapi
import ida_pro
from ida_pro import IDA_SDK_VERSION

def get_architecture():
    arch_id = idaapi.ph_get_id()
    if arch_id == idaapi.PLFM_386:
        return "x86"
    elif arch_id == idaapi.PLFM_ARM:
        return "ARM"
    elif arch_id == idaapi.PLFM_MIPS:
        return "MIPS"
    elif arch_id == idaapi.PLFM_AMD64:  # Add this condition for x64
        return "x64"
    # 可以根据需要添加其他架构的判断
    else:
        return "Unknown"

def is_assignment_insn(opcode):
    return opcode in ['m_stx','m_ldx','m_ldc','m_mov','m_neg','m_lnot',
                      'm_bnot','m_xds','m_xdu','m_low','m_high','m_add',
                      'm_sub','m_mul','m_udiv','m_sdiv','m_umod','m_smod',
                      'm_f2i','m_f2u','m_i2f','m_u2f','m_f2f','m_fneg',
                      'm_fadd','m_fadd','m_fmul','m_fdiv']

def mark_assignment_insn_stored_in_mem(mtoken_list):
    # 检测该指令是否为赋值指令
    if is_assignment_insn(mtoken_list[0]):
        # 检测目标操作数是否是全局变量或局部变量，如果是则标记该指令
        if mtoken_list[3] in ['mop_v','mop_l']:
            return True
    return False

def is_reg_storing_return_values(op_type,op):
    architecture = get_architecture()
    if op_type == 'mop_r':
        if architecture == 'x86':
            return op in ['rax']


def mark_special_registers_before_return(minsn_list,mtoken_list_list):
    curr_mtoken = mtoken_list_list[-1]
    if curr_mtoken[0] != 'm_ret':
        return False

    index = len(minsn_list)
    while index >= 0:
        prev_mtoken = mtoken_list_list[index]
        prev_minsn = minsn_list[index]
        # 除去跳转指令
        if 'j' not in  prev_mtoken[0]:
            # 判断是否是特殊寄存器
            return is_reg_storing_return_values(prev_mtoken[3],prev_minsn['operand'][3])
        index -= 1
    return False


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


def get_mcode_name(mcode):
    """returns the name of the mcode_t passed in parameter."""
    for x in dir(hr):
        if x.startswith('m_'):
            if mcode == getattr(hr, x):
                return x
    return None


def get_mopt_type(mopt):
    """returns the name of the mopt_t passed in parameter."""
    for x in dir(hr):
        if x.startswith('mop_'):
            if mopt == getattr(hr, x):
                return x
    return None


def traverse_mblock(mblock):

    mblock_dict = {}

    mblock_dict['id'] = mblock.serial - 1

    succs = []
    for succ in mblock.succset:
        succs.append(succ)
    mblock_dict['succs'] = succs
    minsn_list_list = []
    mtoken_list_list = []
    curr = mblock.head
    while True:
        if not curr:
            break
        minsn_list = []
        minsn = curr



        mtoken_list = []
        mtoken_list.append(get_mcode_name(minsn.opcode))
        mtoken_list.append(get_mopt_type(minsn.l.t))
        mtoken_list.append(get_mopt_type(minsn.r.t))
        mtoken_list.append(get_mopt_type(minsn.d.t))
        mtoken_list_list.append(mtoken_list)

        minsn_list.append(mtoken_list[0]+'>>'+ida_lines.tag_remove(minsn._print()).split(' ')[0])
        
        minsn_list.append(mtoken_list[1]+'>>'+ida_lines.tag_remove(minsn.l._print()))
        minsn_list.append(mtoken_list[2]+'>>'+ida_lines.tag_remove(minsn.r._print()))
        minsn_list.append(mtoken_list[3]+'>>'+ida_lines.tag_remove(minsn.d._print()))

        # minsn_dict['marked'] = False|mark_assignment_insn_stored_in_mem(mtoken_list)|mark_special_registers_before_return(minsn_list,mtoken_list_list)
        
        minsn_list_list.append(minsn_list)

        curr = curr.next
    mblock_dict['minsn'] = minsn_list_list
    mblock_dict['mtoken'] = mtoken_list_list
    return mblock_dict



def traverse_mba(mba, lines):

    blocks_list = []
    qty = mba.qty

    for _id in range(qty-2):
        mblock = mba.get_mblock(_id+1)
        blocks_list.append(traverse_mblock(mblock))
    return blocks_list

hr.DECOMP_WARNINGS = False

def traverse_mc(output_txt):

    print("Running proc_mc.py")

    fn_list = []

    for idx, func_ea in enumerate(idautils.Functions()):
        try:
            seg = idc.get_segm_name(func_ea)
            if seg != '.text':
                continue
            fn_dict = {}

            pfn = ida_funcs.get_func(func_ea)
            fn_name = ida_funcs.get_func_name(pfn.start_ea)
            fn_dict['name'] = fn_name

            mbr = hr.mba_ranges_t(pfn)

            if mbr is None:
                continue

            hf = hr.hexrays_failure_t()
            ml = hr.mlist_t()

            mba = hr.gen_microcode(mbr, hf, ml, hr.DECOMP_WARNINGS, hr.MMAT_GLBOPT1)
            vp = printer_t()

            mba.set_mba_flags(mba.get_mba_flags() | hr.MBA_SHORT)
            mba._print(vp)

            fn_dict['blocks'] = traverse_mba(mba, vp.get_mc())
            filepath = idc.get_input_file_path()
            fn_dict['filename'] = filepath
            fn_dict['architecture'] = get_architecture()
            func_start = pfn.start_ea
            func_end = pfn.end_ea
            func_bytes = idaapi.get_bytes(func_start, func_end - func_start)
            fn_dict['bytes'] = func_bytes.hex()

            fn_list.append(fn_dict)

        except Exception as e:
            print("[!] Exception: skipping function fva: %d" % func_ea)
            print(e)

    print(fn_list)
    with open(output_txt, 'a+') as micro_txt:
        micro_txt.write(str(fn_list))
    return fn_list


if __name__ == '__main__':
    if not idaapi.get_plugin_options("flowchart"):
        print("[!] -Oflowchart option is missing")
        ida_pro.qexit(1)

    plugin_options = idaapi.get_plugin_options("flowchart").split(';')
    if len(plugin_options) != 2:
        print("[!] -Oflowchart:IDB_PATH:OUTPUT_TXT is required")
        ida_pro.qexit(1)
    
    idb_path = plugin_options[0]
    output_txt = plugin_options[1]
    
    traverse_mc(output_txt)
    ida_pro.qexit(0)