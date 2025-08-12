import json
import time
import click
import subprocess
from os import walk
from os.path import join
from os.path import abspath
from os.path import dirname
from os.path import isfile
from os.path import relpath
from os import getenv
import os

# IDA_PATH = "D:\Application\IDA_Pro_7.7\idat64.exe"
# IDA_PATH = getenv("IDA_PATH", "D:\IDA_Pro_7.7\ida64")
# IDA_PATH32 = getenv("IDA_PATH", "D:\IDA_Pro_7.7\ida")
IDA_PATH = "D:\IDA_Pro_7.7\ida64"
IDA_PATH32 = "D:\IDA_Pro_7.7\ida"
IDA_SCRIPT_PATH = join(dirname(abspath(__file__)), 'gen_cfg_json.py')
LOG_PATH = "gen_cfg_log.txt"
# REPO_PATH = dirname(dirname(dirname(abspath(__file__))))
# JSON_FOLDER_PATH = 'D:/kaiyanh/Download/binary/IDA_scripts/gen_cfg_json/JSON'
# idbs_folder = './idbs'
@click.command()
@click.option('-i', '--idb-path', required=True,
              help='idb path.')
@click.option('-o', '--output-dir', required=True,
              help='Output directory.')


def main(idb_path, output_dir):
    try:
        print("[D] IDBs folder: {}".format(idb_path))
        success_cnt, error_cnt = 0, 0
        for root, _, files in walk(idb_path):
            for f_name in files:
                if (not f_name.endswith(".i64")) and (not f_name.endswith(".idb")):
                    continue
                idb_path = os.path.abspath(os.path.join(root, f_name))
                # idb_index = full_path.lower().find("idbs")
                # if idb_index == -1:
                #     print("[!] Error: file not in idbs path")
                #     continue
                
                # idb_path = full_path[idb_index:]
                # idb_path = join(root, f_name)
                print("\n[D] Processing: {}".format(idb_path))

                if not isfile(idb_path):
                    print("[!] Error: {} not exists".format(idb_path))
                    continue
                
                if 'arm32' in idb_path or 'mips32' in idb_path or 'x86' in idb_path:
                    bitness = 32
                else:
                    bitness = 64
                    
                if bitness == 64:
                    cmd = [IDA_PATH,
                        '-A',
                        '-L{}'.format(LOG_PATH),
                        '-S{}'.format(IDA_SCRIPT_PATH),
                        '-Ojson:{};{}'.format(output_dir, idb_path),
                        idb_path]
                else:
                    cmd = [IDA_PATH32,
                        '-A',
                        '-L{}'.format(LOG_PATH),
                        '-S{}'.format(IDA_SCRIPT_PATH),
                        '-Ojson:{};{}'.format(output_dir, idb_path),
                        idb_path]

                print("[D] cmd: {}".format(cmd))

                # get idapython plugin
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

        print("\n# IDBs correctly processed: {}".format(success_cnt))
        print("# IDBs error: {}".format(error_cnt))

    except Exception as e:
        print("[!] Exception\n{}".format(e))


if __name__ == '__main__':
    main()