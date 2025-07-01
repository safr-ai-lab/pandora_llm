def membership_inference_entry_point():
    import sys
    import subprocess
    subprocess.call(["accelerate", "launch", "-m", "pandora_llm.routines.membership_inference"]+sys.argv[1:])