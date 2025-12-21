import sys
import runpy
import os
from pathlib import Path
import shlex
import subprocess
import re


def get_cmd_list(cmd):
    def replace_backticks(match):
        command = match.group(1)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    
    cmd = re.sub(r'`([^`]+)`', replace_backticks, cmd)
    result = shlex.split(cmd)
    for idx, i in enumerate(result):
        if i == "\n":
            result.pop(idx)
    return result


def main(cmd):
    current_dir = Path(__file__).parent.resolve()
    os.chdir(current_dir)

    cmd = get_cmd_list(cmd)

    if cmd[0] == "python" or cmd[0] == "python3":
        cmd.pop(0)

    if cmd[0] == "-m":
        cmd.pop(0)
        fun = runpy.run_module
    else:
        fun = runpy.run_path

    sys.argv.extend(cmd[1:])
    fun(cmd[0], run_name="__main__")


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    # sample command to run the script
    cmd = """
    python app.py
    """
    
    main(cmd)