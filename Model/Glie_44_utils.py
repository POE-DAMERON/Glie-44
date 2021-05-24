from PIL import ImageDraw, ImageFont
import torchvision


class Utils():

    """
      converts a PIL image into a PyTorch Tensor
    """

    @staticmethod
    def to_tensor():
        return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    """
    Input: Path of the txt file with annotations as in the VisDrone dataset\n
    Output: A numpy array containing the information for bounding boxes
  """

    @staticmethod
    def read_txt_visdrone(path):
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

    """
    Adds boxes to a PIL image and takes as inputs:
    the PIL image,
    an array of boxes,
    a list of classes,
    a list of scores,
    a path to the font to use,
    the precision as a float between 0 and 1.
  """

    @staticmethod
    def add_blocks(image, boxes, classes, scores, font_path, precision):
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for i in range(len(boxes)):
            if scores[i] > precision and classes[i] != 0:
                draw.rectangle(
                    boxes[i], outline=Utils.which_color(classes[i]), width=3)
                draw.text((boxes[i][0], boxes[i][1]), Utils.box_title(classes[i]), fill=(
                    255, 255, 255), stroke_fill=(0, 0, 0, 255), stroke_width=2, font=ImageFont.truetype(font_path, 20))

    """
    Returns the text to add on the screen to identify the class.
  """

    @staticmethod
    def box_title(label_index):
        return str(label_index)

    """
    Returns a tuple representing the RGB colors depending on the class.
  """

    @staticmethod
    def which_color(class_id):
        color_value = int(class_id) * 64
        return (min(color_value, 255), max(min(color_value - 255, 255), 0), max(min(color_value - 256 * 2 - 1, 255), 0))

    @staticmethod
    def prepare_coords(array):
        return (array[1], array[0], array[3], array[2])

    @staticmethod
    def convert_milliseconds(ms):
        output = ""
        if ms//86400000 != 0:
            output += f"{ms//86400000} day(s) "
        if (ms % 86400000)//3600000 != 0:
            output += f"{(ms%86400000)//3600000} hour(s) "
        if (ms % 86400000 % 3600000)//60000 != 0:
            output += f"{(ms%86400000%3600000)//60000} minute(s) "
        if (ms % 86400000 % 3600000 % 60000)//1000 != 0:
            output += f"{(ms%86400000%3600000%60000)//1000} second(s) "
        return output[:-1]
