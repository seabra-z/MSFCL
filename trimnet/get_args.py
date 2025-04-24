import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import os,sys
os.chdir(sys.path[0])

with open('args.json') as f:
    config = json.load(f)


