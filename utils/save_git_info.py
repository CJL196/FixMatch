import os
import sys
import socket
import subprocess
import yaml
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

def run_git_command(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        return "[Error executing: {}]".format(" ".join(command))

def get_env_info():
    info = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Hostname": socket.gethostname(),
        "Python Version": sys.version.replace("\n", " "),
    }

    if torch is not None:
        info["PyTorch Version"] = torch.__version__
        info["CUDA Available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["CUDA Version"] = torch.version.cuda
            info["cuDNN Version"] = torch.backends.cudnn.version()
        else:
            info["CUDA Version"] = "N/A"
            info["cuDNN Version"] = "N/A"
    else:
        info["PyTorch"] = "Not Installed"

    return info

def save_git_info(output_dir=".", txt_name="git_info.txt", yaml_name="git_info.yaml"):
    try:
        # 获取信息
        env_info = get_env_info()
        git_info = {
            "Git Branch": run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "Commit Hash": run_git_command(["git", "rev-parse", "HEAD"]),
            "Git Status": run_git_command(["git", "status"]),
            "Last Commit Log": run_git_command(["git", "log", "-1"]),
            "Git Diff": run_git_command(["git", "diff"])
        }

        # 保存为 TXT
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, txt_name)
        with open(txt_path, "w", encoding="utf-8") as f:
            for k, v in env_info.items():
                f.write(f"{k:<17}: {v}\n")
            for k, v in git_info.items():
                f.write(f"{k}:\n{v}\n\n" if "\n" in v else f"{k:<17}: {v}\n")
        print(f"[Git Info] Saved TXT to {txt_path}")

        # 保存为 YAML
        yaml_path = os.path.join(output_dir, yaml_name)
        combined_info = {**env_info, **git_info}
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(combined_info, f, allow_unicode=True, sort_keys=False)
        print(f"[Git Info] Saved YAML to {yaml_path}")

    except Exception as e:
        print(f"[Git Info] Exception: {e}")
