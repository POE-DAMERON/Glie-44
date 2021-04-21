import os
import json
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageFont

def which_color(class_id):
    color_value = int(class_id) * 9
    return (min(color_value, 255), max(min(color_value - 255, 255),0),max(min(color_value - 256 * 2 - 1, 255),0))

def prepare_coords(array):
    return (array[1], array[0], array[3], array[2])

def add_blocks(pred, draw, height, width, font_path, precision):
    boxes = pred['detection_boxes'].numpy()[0]
    classes = pred['detection_classes'].numpy()[0]

    boxes[:, 0] *= height
    boxes[:, 1] *= width
    boxes[:, 2] *= height
    boxes[:, 3] *= width

    for i in range(len(boxes)):
        if pred['detection_scores'].numpy()[0][i] > precision:
            draw.rectangle(prepare_coords(boxes[i]), outline = which_color(classes[i]), width = 3)
            draw.text((boxes[i][1], boxes[i][0]),str(classes[i])[:-2], fill=(255,255,255), stroke_fill= (0,0,0,255), stroke_width = 2, font= ImageFont.truetype(font_path, 20))

def removeIgnoredRegions(boxarray):
    """
        Input: a numpy array of the box information as in the VisDrone dataset\n
        Output: a numpy array without the ignored region information
    """
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
    df = pd.DataFrame(data=boxarray,columns=columns)
    filtered = df[(df.object_category != "0")]
    return filtered.to_numpy()

def read_txt_visdrone(path):
    """
        Input: Path of the txt file with annotations as in the VisDrone dataset\n
        Output: A numpy array containing the information for bounding boxes
    """
    lines = []
    with open(path) as f:
        lines = f.readlines()
        f.close()
    df = []
    for x in lines:
        splitLine = x.split(",")
        splitLine[-1] = splitLine[-1].split("\n")[0]
        splitLine = list(map(int,splitLine))
        df.append(splitLine)
    return df

def bottom_right(left,top,width,height):
    """
        Input: co-ordinates for the top left corner of a bounding box and its width and height\n
        Output: co-ordinates for the bottom right corner
    """
    bottom = top + height
    right = left + width
    return right, bottom

def text_feature(value):
  "Outputs byte list from string"
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def image_feature(value):
  "Ouputs byte list from image"
  return tf.train.Feature( bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def int_list_feature(value):
  "Outputs Int64List feature from an int list"
  return tf.train.Feature( int64_list = tf.train.Int64List(value=value))

def int_feature(value):
  "Outputs int64 from int, boolean, enum or uint"
  return tf.train.Feature( int64_list = tf.train.Int64List(value=[value]))

def create_tfrecordfile(root_dir,vid_name, feature_list):
  tfrecord_path = os.path.join(root_dir,"tfrecords")
  if not os.path.exists(os.path.join(root_dir,"tfrecords")):
    os.makedirs(os.path.join(tfrecord_path))
  
  with tf.io.TFRecordWriter(os.path.join(tfrecord_path,vid_name+".tfrec")) as writer:
    for feature in feature_list:
      writer.write(feature.SerializeToString())
      
def annotationToFeature(vid_dir, annotations, frame_list):
  """
  Inputs:
    root_dir - root directory of dataset
    vid_name - name of video (collection of frames)
    annotations - list of annotations that are returned from read_visdrone_txt
    frame_list - array of frames for the video
  """
  
  img_path = os.path.join(vid_dir, frame_list[(annotations[0]-1)])
  image = tf.io.decode_jpeg(tf.io.read_file(img_path))
  right, bottom = bottom_right(annotations[2],annotations[3],annotations[4],annotations[5])
  feature = {
      "frame_index": int_feature(annotations[0]),
      "target_id": int_feature(annotations[1]),
      "bbox": int_list_feature([annotations[2],annotations[3], right, bottom]),
      "score": int_feature(annotations[6]),
      "object_category": int_feature(annotations[7]),
      "truncation": int_feature(annotations[8]),
      "oclusion": int_feature(annotations[9]),
      "image": image_feature(image),
      "path": text_feature(img_path)
  }

  return tf.train.Example(features = tf.train.Features(feature=feature))
