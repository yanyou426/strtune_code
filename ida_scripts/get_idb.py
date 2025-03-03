import click
import subprocess

from os import getenv
from os import makedirs
from os import mkdir
from os import walk
from os.path import abspath
from os.path import dirname
from os.path import isdir
from os.path import isfile
from os.path import join
from os.path import relpath
from os.path import samefile

BIN_FOLDER = join(dirname(dirname(abspath(__file__))), 'Binaries')
IDB_FOLDER = join(dirname(dirname(abspath(__file__))), 'IDBs')
IDA_PATH = getenv("IDA_PATH", "D:\idapro-7.7\ida64")
IDA_PATH32 = getenv("IDA_PATH", "D:\idapro-7.7\ida")
LOG_PATH = "generate_idbs_log.txt"

def export_idb(input_path, output_path, bitness):
    """Launch IDA Pro and export the IDB."""
    try:
        print("Export IDB for {}".format(input_path))

        if bitness == 32:
            ida_output = str(subprocess.check_output([
                IDA_PATH32,
                "-L{}".format(LOG_PATH),  # name of the log file.
                "-B",  # batch mode. generate .IDB and .ASM files
                "-o{}".format(output_path),
                input_path
            ]))
        elif bitness == 64:
            ida_output = str(subprocess.check_output([
                IDA_PATH,
                "-L{}".format(LOG_PATH),  # name of the log file.
                "-B",  # batch mode. generate .IDB and .ASM files
                "-o{}".format(output_path),
                input_path
            ]))

        if not isfile(output_path):
            print("[!] Error: file {} not found".format(output_path))
            print(ida_output)
            return False

        return True

    except Exception as e:
        print("[!] Exception in export_idb\n{}".format(e))

def directory_walk(input_folder, output_folder):
    """Walk the directory tree and launch IDA Pro."""
    try:
        print("[D] input_folder: {}".format(input_folder))
        print("[D] output_folder: {}".format(output_folder))
        export_error, export_success = 0, 0
        if not input_folder or not output_folder:
            print("[!] Error: missing input/output folder")
            return

        if not isdir(output_folder):
            mkdir(output_folder)

        for root, _, files in walk(input_folder):
            for fname in files:
                if fname.endswith(".log") \
                        or fname.endswith(".idb") \
                        or fname.endswith(".i64") \
                        or fname.endswith(".py"):
                    continue

                tmp_out = output_folder
                if not samefile(root, input_folder):
                    tmp_out = join(
                        output_folder,
                        relpath(root, input_folder))
                    if not isdir(tmp_out):
                        makedirs(tmp_out)

                input_path = join(root, fname)
                arch = input_path.split('\\')[-1].split('-')[0]
                if arch in ['arm32', 'mips32', 'x86']:
                    bitness = 32
                    output_path = join(tmp_out, fname + ".idb")
                elif arch in ['arm64', 'mips64', 'x64']:
                    bitness = 64
                    output_path = join(tmp_out, fname + ".i64")
                    continue

                if export_idb(input_path, output_path, bitness):
                    export_success += 1
                else:
                    export_error += 1

        print("# IDBs correctly exported: {}".format(export_success))
        print("# IDBs error: {}".format(export_error))

    except Exception as e:
        print("[!] Exception in directory_walk\n{}".format(e))

@click.command()
@click.option('--db1', is_flag=True)
@click.option('--dbvuln', is_flag=True)
@click.option('--db4', is_flag=True)
@click.option('--db5', is_flag=True)

def generate_idb(db1, dbvuln, db4, db5):
    """Launch IDA Pro and export the IDBs."""
    no_action = True
    if db1:
        no_action = False
        directory_walk(
            join(BIN_FOLDER, 'Dataset-1'),
            join(IDB_FOLDER, 'Dataset-1'))
    if dbvuln:
        no_action = False
        directory_walk(
            join(BIN_FOLDER, 'Dataset-Vulnerability'),
            join(IDB_FOLDER, 'Dataset-Vulnerability'))
    if db4:
        no_action = False
        directory_walk(
            join(BIN_FOLDER, 'Dataset-4'),
            join(IDB_FOLDER, 'Dataset-4'))
    if db5:
        no_action = False
        directory_walk(
            join(BIN_FOLDER, 'Dataset-5'),
            join(IDB_FOLDER, 'Dataset-5'))
    if no_action:
        print("Please, select a Dataset to process. --help for options")
    else:
        print("Nothing to do.")
    return


if __name__ == "__main__":
    generate_idb()

