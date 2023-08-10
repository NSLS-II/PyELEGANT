import os
import json
import importlib

this_folder = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(this_folder, "facility.json"), "r") as f:
    facility_name = json.load(f)["name"]

try:
    remote = importlib.import_module("." + facility_name, "pyelegant")
except:
    print("\n## pyelegant:WARNING ##")
    print('Failed to load remote run setup for "{}"'.format(facility_name))
    print("All the Elegant commands will only be run locally.")
    remote = None
