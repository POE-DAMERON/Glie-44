
import os  # Necessary for the 'sorted' function
from os import listdir
from pathlib import Path
import cv2
from tkinter import filedialog, ttk, StringVar
from PIL import Image, ImageTk

"""
    ImageTab is the class that manages the display and the prediction of
    images.

    The main_frame is the specified frame that it is going to display
    elements on.
    max_width and max_height concern the maximum size of the displayed images.
    glie is an instance of the Glie_44 class.
"""


class ImageTab:

    """
        Displays the first elements to be shown on the screen.
    """

    def __init__(self, main_frame, max_width, max_height, glie):
        self.max_width = max_width
        self.max_height = max_height
        self.glie = glie
        self.main_frame = main_frame

        # Displayes the title

        ttk.Label(self.main_frame, text="Prediction on images").grid(
            column=0, row=0, columnspan=2)
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(column=0, row=1, columnspan=2)

        # Displayes the 'Browse files' button

        ttk.Button(self.main_frame, text="Browse files",
                   command=lambda: self.on_click()).grid(column=0, row=2)

    """
        Allows the user to select an image and displays it.
    """

    def on_click(self):

        # Waits for the user to select an image

        self.filepath = self.browse()

        # Displays the selected image

        self.image = Image.open(self.filepath)
        self.display_image(self.image)

        # Displays some elements to configure and run the prediction

        self.prepare_prediction()
        self.predict_button = ttk.Button(self.main_frame, text="Begin prediction",
                                         command=lambda: self.run_model())
        self.predict_button.grid(column=1, row=2)

    """
        Displays a resized version of the given image
    """

    def display_image(self, image):
        image_resized = image.resize(self.get_image_new_size())
        self.tk_image = ImageTk.PhotoImage(image_resized)
        self.image_label.config(image=self.tk_image)

    """
        Displays some elements to configure the prediction
    """

    def prepare_prediction(self):

        # Area to select the precision.

        self.precision_frame = ttk.Frame(self.main_frame)
        self.precision_frame.grid(column=0, row=3)

        ttk.Label(self.precision_frame, text="Minimum confidence rate: ").grid(
            column=0, row=0)
        self.precision_selection = ttk.Spinbox(
            self.precision_frame, from_=0.0, to=1.0, width=10, increment=0.01, wrap=True)
        self.precision_selection.grid(
            column=1, row=0)
        self.precision_selection.set(0.2)

    """
        Runs the prediction on the image and displays the predicted image.
    """

    def run_model(self):
        self.glie.set_precision(float(self.precision_selection.get()))
        self.result = self.glie.run_on_image_with_path(self.filepath)
        self.display_image(self.result)
        self.predict_button.config(text="Save image",
                                   command=lambda: self.save())

    """
        Allows the user to select the output directory and file name before
        saving the predicted image.
    """

    def save(self):
        self.output_filename = StringVar()

        self.dir_path = filedialog.askdirectory(
            initialdir="/", title="Select the output directory")

        self.entry_frame = ttk.Frame(self.main_frame)
        self.entry_frame.grid(column=1, row=3)
        ttk.Label(self.entry_frame, text="Input filename").grid(
            column=0, row=0)
        ttk.Entry(self.entry_frame, textvariable=self.output_filename).grid(
            column=1, row=0, padx=10)
        self.predict_button.config(text="Confirm filename",
                                   command=lambda: self.confirm())

    """
        Saves the predicted image.
    """

    def confirm(self):
        out_name = str(Path(self.dir_path).joinpath(
            self.output_filename.get()+'.jpg'))
        self.result.save(out_name, 'JPEG')
        self.predict_button.config(text="Save image",
                                   command=lambda: self.save())
        self.entry_frame.grid_forget()

    """
        Outputs the sizes of the image to be displayed
    """

    def get_image_new_size(self):
        ratio = min(self.max_width/self.image.width,
                    self.max_height/self.image.height)
        new_width = self.image.width * ratio
        new_height = self.image.height * ratio
        return int(new_width), int(new_height)

    """
        Allows the user to chose an image.
    """

    def browse(self):
        filename = filedialog.askopenfilename(initialdir="/",
                                              title="Select an image",
                                              filetypes=(("JPEG and PNG Images",
                                                          "*.JPEG* *.png* *.jpg*"),
                                                         ("JPEG Images",
                                                          "*.jpeg*"),
                                                         ("PNG Images",
                                                          "*.png*"),
                                                         ("JPG Images",
                                                          "*.jpg*")))
        return filename


"""
    VideoTab is the class that manages the display and the prediction of
    videos.

    The root is the instance of the Tk class.
    The main_frame is the specified frame that it is going to display
    elements on.
    max_width and max_height concern the maximum size of the displayed images.
    glie is an instance of the Glie_44 class.
"""


class VideoTab:

    """
        Displays the first elements to be shown on the screen.
    """

    def __init__(self, root, main_frame, max_width, max_height, glie):
        self.max_width = max_width
        self.max_height = max_height
        self.glie = glie
        self.main_frame = main_frame
        self.root = root

        # Video related attributes

        self.running = False
        self.delay = 15
        self.cap = None

        ttk.Label(self.main_frame, text="Prediction on videos").grid(
            column=0, row=0, columnspan=3)
        self.display_label = ttk.Label(self.main_frame)
        self.display_label.grid(column=0, row=1, columnspan=3)
        ttk.Button(self.main_frame, text="Browse files",
                   command=lambda: self.on_click()).grid(column=0, row=2)

    """
        Displays the first frame of a video.
    """

    def display_video(self):
        ret, frame = self.get_frame()
        if ret:
            self.display_image(Image.fromarray(frame))
        else:
            self.display_label.config(text="No video data", image=None)

    """
        Displays a resized version of the given image.
    """

    def display_image(self, image):
        image_resized = image.resize(self.get_image_new_size(image))
        self.tk_image = ImageTk.PhotoImage(image_resized)
        self.display_label.config(text=None, image=self.tk_image)

    """
        Outputs the sizes of the image to be displayed
    """

    def get_image_new_size(self, image):
        ratio = min(self.max_width/image.width,
                    self.max_height/image.height)
        new_width = image.width * ratio
        new_height = image.height * ratio
        return int(new_width), int(new_height)

    """
        Allows the user to select a video and displays it.
    """

    def on_click(self):
        temp_path = self.browse()
        if temp_path != '':
            if self.cap != None and self.cap.isOpened():
                self.cap.release()
            self.filepath = temp_path
            self.cap = cv2.VideoCapture(self.filepath)
            self.display_video()

        self.predict_button = ttk.Button(self.main_frame, text="Begin prediction",
                                         width=20,
                                         command=lambda: self.prepare_prediction())
        self.predict_button.grid(column=2, row=2)

        self.controller_frame = ttk.Frame(self.main_frame)
        self.controller_frame.grid(column=1, row=2)

        self.play_button = ttk.Button(
            self.controller_frame, text="Play", command=lambda: self.play_pause())
        self.play_button.grid(column=0, row=0)

        self.beggining_button = ttk.Button(
            self.controller_frame, text="Beginning", command=lambda: self.start_over())
        self.beggining_button.grid(column=1, row=0)

    """
        Handles the Play/Pause button.
    """

    def play_pause(self):
        self.running = not self.running
        if self.running:
            self.play_button.config(text="Pause")
            self.run_video()
        else:
            self.play_button.config(text="Play")

    """
        Runs the video frame by frame until paused.
    """

    def run_video(self):
        if self.cap.isOpened():
            self.delay = int(1000/self.cap.get(cv2.CAP_PROP_FPS))
            ret, frame = self.get_frame()
            if ret:
                self.display_image(Image.fromarray(frame))
            else:
                self.running = False
            if self.running:
                self.root.after(self.delay, self.run_video)

    """
        Displays the next frame.
    """

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return (ret, frame)
        else:
            self.display_label.config(
                text="Error displaying the video", image=None)

    """
        Displays to the first frame and pauses the video.
    """

    def start_over(self):
        self.running = False
        self.cap.set(1, 0)
        self.display_video()

    """
        Displays some elements to configure the prediction
    """

    def prepare_prediction(self):
        self.precision_frame = ttk.Frame(self.main_frame)
        self.precision_frame.grid(column=0, row=3, columnspan=2)

        ttk.Label(self.precision_frame, text="Minimum accepted confidence: ").grid(
            column=0, row=0)
        self.precision_selection = ttk.Spinbox(
            self.precision_frame, from_=0.0, to=1.0, width=10, increment=0.01, wrap=True)
        self.precision_selection.grid(
            column=1, row=0)
        self.precision_selection.set(0.2)
        self.save()

    """
        Allows the user to select the output directory and file name before
        predicting and saving the video.
    """

    def save(self):
        self.output_filename = StringVar()
        self.dir_path = filedialog.askdirectory(
            initialdir="/", title="Select the output directory to save the resulted video")
        self.entry_frame = ttk.Frame(self.main_frame)
        self.entry_frame.grid(column=2, row=3)
        ttk.Label(self.entry_frame, text="Input output name").grid(
            column=0, row=0)
        ttk.Entry(self.entry_frame, textvariable=self.output_filename).grid(
            column=1, row=0, padx=10)
        self.predict_button.config(text="Confirm filename",
                                   command=lambda: self.confirm())

    """
        Predicts and saves the video.
    """

    def confirm(self):
        self.cap.set(1, 0)
        self.glie.set_precision(float(self.precision_selection.get()))

        """
        self.progress = CustomProgressBar(self.root, self.main_frame, 200)
        self.root.after(500, self.glie.run_on_video(self.filepath, self.dir_path,
                                                    self.output_filename.get()+'.avi',
                                                    progress_bar=self.progress))
        
        self.loading_label = ttk.Label(self.main_frame, text="Predicting...")
        self.loading_label.grid(column=0, row=4, columnspan=3)
        """

        self.glie.run_on_video(self.filepath, self.dir_path,
                               self.output_filename.get()+'.avi')
        self.predict_button.config(text="Begin prediction",
                                   command=lambda: self.save())
        self.entry_frame.grid_forget()

    """
        Allows the user to chose a video.
    """

    def browse(self):
        return filedialog.askopenfilename(initialdir="/", title="Select a video", filetypes=(("All videos", "*.avi* *.mp4*"), ("AVI videos only", "*.avi*"), ("MP4 videos only", "*.mp4*")))

    # Releases the video source when the object is destroyed

    def __del__(self):
        if self.cap != None and self.cap.isOpened():
            self.cap.release()


"""
    DirectoryTab is the class that manages the display, the prediction and
    compilation of images from a given directory.

    The root is the instance of the Tk class.
    The main_frame is the specified frame that it is going to display
    elements on.
    max_width and max_height concern the maximum size of the displayed images.
    glie is an instance of the Glie_44 class.
"""


class DirectoryTab:

    """
        Displays the first elements to be shown on the screen.
    """

    def __init__(self, root, main_frame, max_width, max_height, glie):
        self.max_width = max_width
        self.max_height = max_height
        self.glie = glie
        self.main_frame = main_frame
        self.root = root

        # Video related attributes

        self.running = False
        self.delay = 24

        ttk.Label(self.main_frame, text="Prediction on directories of images").grid(
            column=0, row=0, columnspan=3)
        self.display_label = ttk.Label(self.main_frame)
        self.display_label.grid(column=0, row=1, columnspan=3)
        ttk.Button(self.main_frame, text="Browse files",
                   command=lambda: self.on_click()).grid(column=0, row=2)

    """
        Displays the first image of the sorted directory.
    """

    def display_video(self):
        self.current_frame = 0
        self.display_image(Image.open(Path(self.folder_path).joinpath(
            self.sorted_images[self.current_frame])))

    """
        Displays a resized version of the given image.
    """

    def display_image(self, image):
        image_resized = image.resize(self.get_image_new_size(image))
        self.tk_image = ImageTk.PhotoImage(image_resized)
        self.display_label.config(text=None, image=self.tk_image)

    """
        Outputs the sizes of the image to be displayed
    """

    def get_image_new_size(self, image):
        ratio = min(self.max_width/image.width,
                    self.max_height/image.height)
        new_width = image.width * ratio
        new_height = image.height * ratio
        return int(new_width), int(new_height)

    """
        Allows the user to select a directory and displays its images
        in a video-like manner.
    """

    def on_click(self):
        temp_path = self.browse()
        if temp_path != '':
            self.folder_path = temp_path
            self.sorted_images = sorted(listdir(self.folder_path),
                                        key=lambda x: x.lstrip("_"))
            self.display_video()

        self.predict_button = ttk.Button(self.main_frame, text="Begin prediction",
                                         command=lambda: self.prepare_prediction())
        self.predict_button.grid(column=2, row=2)

        self.controller_frame = ttk.Frame(self.main_frame)
        self.controller_frame.grid(column=1, row=2)

        self.play_button = ttk.Button(
            self.controller_frame, text="Play", command=lambda: self.play_pause())
        self.play_button.grid(column=0, row=0)

        self.beggining_button = ttk.Button(
            self.controller_frame, text="Beginning", command=lambda: self.start_over())
        self.beggining_button.grid(column=1, row=0)

    """
        Handles the Play/Pause button.
    """

    def play_pause(self):
        self.running = not self.running
        if self.running:
            self.play_button.config(text="Pause")
            self.run_video()
        else:
            self.play_button.config(text="Play")

    """
        Displays each frame like a video until paused.
    """

    def run_video(self):
        self.current_frame += 1
        if self.current_frame < len(self.sorted_images):
            self.display_image(Image.open(
                Path(self.folder_path).joinpath(self.sorted_images[self.current_frame])))
        else:
            self.play_pause()
        if self.running:
            self.root.after(self.delay, self.run_video)

    """
        Displays to the first frame and pauses the "video".
    """

    def start_over(self):
        self.running = False
        self.display_video()

    """
        Displays some elements to configure the prediction
    """

    def prepare_prediction(self):
        self.preparation_frame = ttk.Frame(self.main_frame)
        self.preparation_frame.grid(column=0, row=3, columnspan=2)

        # Precision selection area

        ttk.Label(self.preparation_frame, text="Minimum accepted confidence: ").grid(
            column=0, row=0)
        self.precision_selection = ttk.Spinbox(
            self.preparation_frame, from_=0.0, to=1.0, width=10, increment=0.01, wrap=True)
        self.precision_selection.grid(
            column=1, row=0)
        self.precision_selection.set(0.2)

        # Framerate selection area

        ttk.Label(self.preparation_frame, text="Framerate: ").grid(
            column=0, row=1)
        self.framerate_selection = ttk.Spinbox(
            self.preparation_frame, from_=1, width=10, increment=1, wrap=True)
        self.framerate_selection.grid(
            column=1, row=1)
        self.framerate_selection.set(24)

        self.save()

    """
        Allows the user to select the output directory and file name before
        predicting and saving the predicted and newly compiled video.
    """

    def save(self):
        self.output_filename = StringVar()
        self.dir_path = filedialog.askdirectory(
            initialdir="/", title="Select the output directory to save the resulted video")
        self.entry_frame = ttk.Frame(self.main_frame)
        self.entry_frame.grid(column=2, row=3)
        ttk.Label(self.entry_frame, text="Input output name").grid(
            column=0, row=0)
        ttk.Entry(self.entry_frame, textvariable=self.output_filename).grid(
            column=1, row=0, padx=10)
        self.predict_button.config(text="Confirm filename",
                                   command=lambda: self.confirm())

    """
        Predicts, saves and compiles the directory into a video.
    """

    def confirm(self):
        self.glie.set_precision(float(self.precision_selection.get()))

        """
        self.progress = CustomProgressBar(self.root, self.main_frame, 200)
        self.root.after(500, self.glie.run_on_video(self.filepath, self.dir_path,
                                                    self.output_filename.get()+'.avi',
                                                    progress_bar=self.progress))
        
        self.loading_label = ttk.Label(self.main_frame, text="Predicting...")
        self.loading_label.grid(column=0, row=4, columnspan=3)
        """
        self.glie.run_on_folder(self.folder_path, self.dir_path,
                                self.output_filename.get()+'.avi', framerate=self.framerate_selection.get())
        self.predict_button.config(text="Begin prediction",
                                   command=lambda: self.save())
        self.entry_frame.grid_forget()

    """
        Allows the user to chose a directory.
    """

    def browse(self):
        return filedialog.askdirectory(initialdir="/", title="Select a directory with the video images")


class LegendTab:
    """
        Displays the elements to be shown on the screen.
    """

    def __init__(self, main_frame):
        classes = [
            "ignored regions",
            "pedestrian",
            "people",
            "bicycle",
            "car",
            "van",
            "truck",
            "tricycle",
            "awning-tricycle",
            "bus",
            "motor",
            "other"
        ]

        ttk.Label(main_frame, text="Classes:").grid(
            column=0, row=0, columnspan=2, pady=20, padx=80)

        for i in range(len(classes)):
            ttk.Label(main_frame, text=str(i)).grid(
                column=0, row=i+1, pady=3, padx=10)
            ttk.Label(main_frame, text=classes[i]).grid(
                column=1, row=i+1)
