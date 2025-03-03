import click
import subprocess

from os import getcwd
from os import getenv
from os import walk
from os.path import abspath
from os.path import dirname
from os.path import isfile
from os.path import join
from os.path import relpath

IDA_PATH = getenv("IDA_PATH", "D:\idapro-7.7\ida64.exe")
IDA_PLUGIN = join(dirname(abspath(__file__)), 'IDA_flowchart.py')
REPO_PATH = dirname(dirname(dirname(abspath(__file__))))
LOG_PATH = "flowchart_log.txt"

@click.command()
@click.option("-i", "--idbs-folder", required=True,
              help="Path to the IDBs folder")
@click.option("-o", "--output-csv", required=True,
              help="Path to the output CSV file")
def main(idbs_folder, output_csv):
    """Call IDA_flowchart.py IDA script."""
    try:
        print("[D] IDBs folder: {}".format(idbs_folder))
        print("[D] Output CSV: {}".format(output_csv))

        success_cnt, error_cnt = 0, 0
        for root, _, files in walk(idbs_folder):
            for f_name in files:
                if (not f_name.endswith(".i64")) and (not f_name.endswith(".idb")):
                    continue

                idb_path = join(root, f_name)
                print("\n[D] Processing: {}".format(idb_path))

                if not isfile(idb_path):
                    print("[!] Error: {} not exists".format(idb_path))
                    continue

                rel_idb_path = relpath(
                    join(getcwd(), root, f_name),  # absolute path if IDB
                    REPO_PATH)  # absolute path of the repo folder

                cmd = [IDA_PATH,
                       '-A',
                       '-L{}'.format(LOG_PATH),
                       '-S{}'.format(IDA_PLUGIN),
                       '-Oflowchart:{};{}'.format(
                           rel_idb_path,
                           output_csv),
                       idb_path]

                print("[D] cmd: {}".format(cmd))

                #get idapython plugin
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
        print("[!] Exception in cli_flowchart\n{}".format(e))


if __name__ == '__main__':
    main()
