import PIL
from tkinter import *
from tkinter import filedialog, ttk
import Glie_44
import tabs

"""
    The UI class is a user friendly interface using the ttk library from Tkinter.
    It allows to use some of the model's functions on various types of inputs
    including images and videos.

    The model must be downloaded from the Google Drive directory 'Models'. The link
    is available on Poe Dameron's github.

    For more information during the display, run this file through the shell using
    either of these command:
     - python UI.py
     - python3 UI.py
"""


class UI:
    def __init__(self):

        # Specifies the maximum sizes for the displayed images or videos

        self.max_width = 1500
        self.max_height = 500

        self._tk_root = Tk()

        # Fetches the path of the saved model

        model_path = ''
        while model_path == '':
            model_path = filedialog.askopenfilename(initialdir="/",
                                                    title="Select your model",
                                                    filetypes=((".pth and .pt files",
                                                                "*.pth* *.pt*"),
                                                               (".pth files",
                                                                "*.pth*"),
                                                               (".pt files",
                                                                "*.pt*")))

        self.glie = Glie_44.Glie_44()
        self.glie.load_model(model_path)

        # Initializes the Notebook to use the tabs

        notebook = ttk.Notebook(self._tk_root)

        # Creates the different frames for each tabs

        image_tab = ttk.Frame(notebook)
        video_tab = ttk.Frame(notebook)
        folder_tab = ttk.Frame(notebook)

        # Adds the tabs to the notebook

        notebook.add(image_tab, text='Image')
        notebook.add(video_tab, text='Video')
        notebook.add(folder_tab, text='Directory')
        notebook.pack(expand=1, fill="both")

        # Development of the first tab to predict on images

        tabs.ImageTab(image_tab, self.max_width, self.max_height, self.glie)

        # Development of the second tab to predict on videos

        tabs.VideoTab(self._tk_root, video_tab, self.max_width,
                      self.max_height, self.glie)

        # Development of the third tab to predict on a directory

        tabs.DirectoryTab(self._tk_root, folder_tab, self.max_width,
                          self.max_height, self.glie)

        # Runs the output indefinitely

        self._tk_root.mainloop()


ui = UI()
