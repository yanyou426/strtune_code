import click
import json
import subprocess
import time

from os import getenv
from os.path import abspath
from os.path import dirname
from os.path import isfile
from os.path import join

IDA_PATH = getenv("IDA_PATH", "D:\idapro-7.7\ida64")
IDA_PLUGIN = join(dirname(abspath(__file__)), 'IDA_acfg_disasm.py')
REPO_PATH = dirname(dirname(dirname(abspath(__file__))))
LOG_PATH = "acfg_disasm_log.txt"

@click.command()
@click.option('-j', '--json-path', required=True,
              help='JSON file with selected functions.')
@click.option('-o', '--output-dir', required=True,
              help='Output directory.')

def main(json_path, output_dir):
    """Call IDA_acfg_disasm.py IDA script."""
    try:
        print("[D] JSON path: {}".format(json_path))
        print("[D] Output directory: {}".format(output_dir))

        if not isfile(json_path):
            print("[!] Error: {} does not exist".format(json_path))
            return

        with open(json_path) as f_in:
            jj = json.load(f_in)

            success_cnt, error_cnt = 0, 0
            start_time = time.time()
            for idb_rel_path in jj.keys():


                print("\n[D] Processing: {}".format(idb_rel_path))

                # Convert the relative path into a full path
                idb_path = join(REPO_PATH, idb_rel_path)
                print("[D] IDB full path: {}".format(idb_path))

                if not isfile(idb_path):
                    print("[!] Error: {} does not exist".format(idb_path))
                    continue

                cmd = [IDA_PATH,
                       '-A',
                       '-L{}'.format(LOG_PATH),
                       '-S{}'.format(IDA_PLUGIN),
                       '-Oacfg_disasm:{};{};{}'.format(
                           json_path,
                           idb_rel_path,
                           output_dir),
                       idb_path]

                print("[D] cmd: {}".format(cmd))

                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = proc.communicate()

                if proc.returncode == 0:
                    print("[D] {}: success".format(idb_path))
                    success_cnt += 1
                else:
                    print("[!] Error in {} (returncode={})".format(
                        idb_path, proc.returncode))
                    error_cnt += 1

            end_time = time.time()
            print("[D] Elapsed time: {}".format(end_time - start_time))
            with open(LOG_PATH, "a+") as f_out:
                f_out.write("elapsed_time: {}\n".format(end_time - start_time))

            print("\n# IDBs correctly processed: {}".format(success_cnt))
            print("# IDBs error: {}".format(error_cnt))

    except Exception as e:
        print("[!] Exception in cli_acfg_disasm\n{}".format(e))


if __name__ == '__main__':
    main()
