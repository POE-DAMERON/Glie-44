# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:09:09 2021

@author: Benji
"""

import os
import json
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from utils import read_txt_visdrone, create_tfrecordfile, annotationToFeature

parser = argparse.ArgumentParser(description='Converts the VisDrone annotations and image frames to a tfrecord.')
parser.add_argument('-dir','--directory', help='The directory for the dataset.', required=True)
parser.add_argument('-n','--name', help='The name of the video sequence.', required=True)

args = parser.parse_args()

if not(os.path.exists(args.directory)):
    print("Directory cannot be found for given name.")
    exit(404)
  
vid_dir = os.path.join(args.directory, "sequences", args.name)
ann_dir = os.path.join(args.directory, "annotations")

print(vid_dir)
if not(os.path.exists(os.path.join(vid_dir,args.name))):
    print("Cannot find video sequence")
    exit(404)
    
if not(os.path.exists(ann_dir+"/"+args.name+".txt")):
    print("Cannot find video sequence")
    exit(404)



frame_list = os.listdir(vid_dir)

ann = read_txt_visdrone(os.path.join(ann_dir,args.name + ".txt"))

columns = [
        "frame_index",
        "target_id",
        "bbox_left",
        "bbox_top",
        "bbox_width",
        "bbox_height",
        "score",
        "object_category",
        "truncation",
        "oclusion"
    ]

features = []

for x in ann:
    feature = annotationToFeature(vid_dir,x,frame_list)
    features.append(feature)
    
create_tfrecordfile(args.directory,args.name, features)