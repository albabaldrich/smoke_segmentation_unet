import os
import json

def SaveParams(config, name):
    with open(name, "w") as f: json.dump(config,f,indent=4)

def LoadParams(name):
    with open(name, "r") as f: cf = json.load(f)
    cf['test_name'] = os.path.basename(os.path.splitext(name)[0])
    return cf

