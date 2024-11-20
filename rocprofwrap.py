import os
import sys
import time
import argparse
import signal
import subprocess


def stopProf():
    # Send stop signal to profiling processes
    for ps in profProc:
        ps.send_signal(signal.SIGINT)

    # Waiting for profiling processes to finsih
    for ps in profProc:
        ps.wait()

def cleanUp():
    stopProf()
    app.terminate()
    app.wait()

parser = argparse.ArgumentParser(
    prog="rocprofwrap",
    description="Profiles power,frequency, voltage, and performance counters for MI products.",
    epilog="Example: python run.py --cmd=\"ls -lh\" --gpus=0 --prefix=metrics.csv --redirect=logfile.txt --counters_file=counters.json"
)

parser.add_argument("--cmd", help="Application command", required=True)
parser.add_argument("--gpus", default="0", help="List of GPUs to profile "+"(default: %(default)s)", required=True)
parser.add_argument("--prefix", default="metrics.csv", help="Filename prefix for data output "+"(default: %(default)s)", required=True)
parser.add_argument("--redirect", default=None, help="Filename for stdout log file.")
# add one more argument to specify the counters included in file to profile
parser.add_argument("--counters_file",
                    default=None,
                    help="Optional JSON file containing hardware counter configurations. "
                         "If not provided, default counters will be used.",required=False)



args = parser.parse_args()

CMD = args.cmd
GPUS = args.gpus.split(",")
PREFIX = args.prefix
REDIRECT = args.redirect
COUNTERS_FILE = args.counters_file


profProc = []

if "WRAPPER_ROOT" not in os.environ:
    print("rocprofwrap: Please set your WRAPPER_ROOT environment variable.")
    sys.exit()


try:
    for gid in GPUS:
        ps = [os.getenv("WRAPPER_ROOT")+"/gpuprof", PREFIX+"_"+gid, gid]

        # Add counters file if provided
        if COUNTERS_FILE is not None:
            if os.path.exists(COUNTERS_FILE):
                ps.append(COUNTERS_FILE)
            else:
                print(f"Warning: Counters file {COUNTERS_FILE} not found. Using default counters.")
        profProc.append(subprocess.Popen(ps))
except Exception as e:
    stopProf()
    print("rocprofwrap: ", e)
    sys.exit()


try:
    if REDIRECT is not None:
        with open(REDIRECT, "w") as fp:
            app = subprocess.Popen(CMD, shell=True, stdout=fp)
    else:
        app = subprocess.Popen(CMD, shell=True)
except Exception as e:
    cleanUp()
    print("rocprofwrap: ", e)
    sys.exit()


# Wait until the application finishes
while app.poll() is None:
    time.sleep(0.1)

cleanUp()

print("Detected that the application ended.")

sys.exit()


