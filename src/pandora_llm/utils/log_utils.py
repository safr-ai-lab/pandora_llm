import os
import sys
import re
import logging
import subprocess
import torch
from accelerate.logging import get_logger
from accelerate import PartialState
PartialState()

def get_my_logger(log_file: str="output.log",log_level: str="INFO") -> logging.Logger:
    """Gets logger"""
    my_logger = get_logger(__name__,log_level=log_level)
    my_logger.logger.addHandler(logging.StreamHandler(sys.stdout))
    my_logger.logger.addHandler(logging.FileHandler(log_file,mode="w"))
    return my_logger

def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],cwd=os.path.dirname(os.path.realpath(__file__))).decode('ascii').strip()
    except:
        return None

def clean_filename(filename: str) -> str:
    return re.sub(r"[,/\\?%*:|\"<>\x7F\x00-\x1F\']", "-", filename)

def mem_stats(marker: str=None) -> None:
    """
    Prints memory statistics for memory management
    
    Args:
        marker: a string to mark where this is being printed
    """
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory / 1024**3
        r = torch.cuda.memory_reserved(0) / 1024**3
        a = torch.cuda.memory_allocated(0) / 1024**3
        print(f"---------------------------------")
        if marker:
            print(marker)
        print(
            f"Total Memory: {t:.2f} GB\n"
            f"Reserved Memory: {r:.2f} GB ({(100*(r/t)):.2f}%)\n"
            f"Remaining Memory: {t-r:.2f} GB ({(100*(t-r)/t):.2f}%)\n"
            f"---------------------------------\n"
            f"Allocated Memory: {a:.2f} GB ({(100*(a/t)):.2f}%)\n"
            f"Percent of Reserved Allocated: {(100*(a+1e-9)/(r+1e-9)):.2f}%\n"
            f"---------------------------------\n"
        )
            