import re
import subprocess
import sys


def kill_torch_zombie_processes(device):
    # Run the command and capture its output
    try:
        output = subprocess.check_output(
            ["bash", "-c", f"fuser -v /dev/nvidia{device}"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        output = e.output  # Capture output even if the command fails

    # Extract all PIDs of processes named 'pt_main_thread'
    pids = re.findall(r"\s+root\s+(\d+)\s+F....\s+pt_main_thread", output)

    # Kill each process
    for pid in pids:
        try:
            subprocess.run(["bash", "-c", f"kill -9 {pid}"], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to kill process {pid}")

    print(f"Attempted to kill {len(pids)} processes on device nvidia{device}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python kill_torch_zombie_process.py <device_number>")
        sys.exit(1)

    device = sys.argv[1]
    kill_torch_zombie_processes(device)
