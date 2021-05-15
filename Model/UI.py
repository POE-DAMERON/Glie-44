import os

init_commands = ['git clone https://github.com/pytorch/vision.git',
                 'cd vision',
                 'git checkout v0.3.0',

                 'copy references/detection/utils.py ../',
                 'copy references/detection/transforms.py ../',
                 'copy references/detection/coco_eval.py ../',
                 'copy references/detection/engine.py ../',
                 'copy references/detection/coco_utils.py ../']


def init():
    for command in init_commands:
        os.system(command)


class UI:
    pass
