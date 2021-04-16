import pandas as pd
import numpy as np

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
    return df.to_numpy()

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
        df.append(splitLine)
    return df

def bottom_right(left,top,width,height):
    """
        Input: co-ordinates for the top left corner of a bounding box and its width and height\n
        Output: co-ordinates for the bottom right corner
    """
    bottom = top - height
    right = left - width
    return right, bottom