import importlib
import os

remote_name = os.environ.get("PYELEGANT_REMOTE", "")

if remote_name != "":
    try:
        remote = importlib.import_module(f".{remote_name}", "pyelegant")
    except:
        remote = None
else:
    remote = None

if remote is None:
    print("\n## pyelegant:WARNING ##")
    print(
        "Invalid $PYELEGANT_REMOTE. All the ELEGANT commands will only be run locally."
    )
