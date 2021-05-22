import cv2 as cv
import transforms as T
from pathlib import Path
from os import path, listdir
from Glie_44_utils import Utils
from PIL import Image
import numpy as np
import torch
import torchvision

"""
  Glie_44 represents the core of our program. It will run the predictions
  depending on customizable settings.
"""


class Glie_44():

    """
      The inputs are:

      precision as a float between 0 and 1 representing the minimum accepted
      confidence to display a predicted box,

      font_path as the path to the font to be used.
    """

    def __init__(self, precision=.2,
                 font_path=Path().absolute().joinpath(
                     'Data/fonts/Roboto-Regular.ttf'
                 )):
        self._precision = precision
        self._font_path = str(font_path)

    def set_font_path(self, font_path):
        self._font_path = font_path

    def get_font_path(self):
        return self._font_path

    def set_precision(self, precision):
        self._precision = precision

    def get_precision(self):
        return self._precision

    def set_model(self, model):
        self._model = model

    def get_model(self):
        return self._model

    """
    Loads model from a given path, depending on the available device.
  """

    def load_model(self, path):
        if torch.cuda.is_available():
            mapLocation = torch.device('cuda')
        else:
            mapLocation = torch.device('cpu')
        self._model = torch.load(path, map_location=mapLocation)

    def pred(self, image):
        self._model.eval()
        return self._model(image)

    def convert_to_tensor(self, image_data):
        return torchvision.transforms.ToTensor()(image_data)

    """
      Takes a PIL image as an input and returns the image with the predicted
      boxes. 
    """

    def run_on_image(self, image):
        image_tensor = self.convert_to_tensor(image)
        pred = self.pred([image_tensor])
        Utils.add_blocks(image,
                         pred[0]['boxes'].tolist(),
                         pred[0]['labels'].tolist(),
                         pred[0]['scores'].tolist(),
                         self._font_path,
                         self._precision)
        return image

    """
      Takes the path of an image as an input and returns the image with the 
      predicted boxes. 
    """

    def run_on_image_with_path(self, image_path):
        image = Image.open(image_path)
        return self.run_on_image(image)

    """
      Takes a folder path where the images should be in chronological order to 
      compile the predicted images in a video.
    """

    def run_on_folder(self, folder_path=Path().absolute().joinpath(
            'Glie-44/Model/Data/VisDrone2019-MOT-test-challenge' +
            '/sequences/uav0000006_06900_v'),
            output_folder=Path().absolute().joinpath('Glie-44/Model/outputs'),
            output_path='',
            framerate=20):

        # Makes a list of all the frames, in the right order

        video_frames = sorted(listdir(folder_path),
                              key=lambda x: x.lstrip("_"))

        # Prepares the variables for the video writer

        s = Image.open(Path(folder_path).joinpath(video_frames[0])).size
        if output_path == '':
            output_path = path.basename(folder_path) + '.avi'
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        writer = cv.VideoWriter(str(output_folder) + '/' + output_path,
                                fourcc,
                                float(framerate),
                                s)

        sample = video_frames
        total_frames = len(sample)

        # Runs on all frames
        for i in range(total_frames):
            print(f"Predicting frame {i + 1} / {total_frames}")

            # Predicts and draws the boxes on the image
            result = self.run_on_image_with_path(
                Path(folder_path).joinpath(sample[i]))

            # Writes the output image on the new video
            writer.write(cv.cvtColor(np.uint8(result), cv.COLOR_RGB2BGR))

        writer.release()

        """
      Takes a video file to predict each of its frames.
    """

    def run_on_video(self, video_path, output_folder=Path().absolute().joinpath('Glie-44/Model/outputs'), output_path=''):

        # Opens the video capture
        cap = cv.VideoCapture(video_path)
        if output_path == '':
            output_path = Path(video_path).stem + '_output.avi'

        if (cap.isOpened()):

            # Prepares arguments for the video writer object
            s = (int(cap.get(3)), int(cap.get(4)))
            fourcc = cv.VideoWriter_fourcc(*'DIVX')
            framerate = cap.get(cv.CAP_PROP_FPS)
            writer = cv.VideoWriter(str(output_folder) + '/' + output_path,
                                    fourcc,
                                    framerate,
                                    s)
            # reads the first frame
            ret, frame = cap.read()
            number_of_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            # Runs on all the frames
            while ret == True:
                print(f"Predicting frame {current_frame} / {number_of_frames}")

                # Predicts  and draws the boxes on the frame
                result = self.run_on_image(Image.fromarray(frame))

                # Writes the frame on the new video
                writer.write(np.uint8(result))

                # Reads the next frame
                ret, frame = cap.read()
                current_frame += 1
