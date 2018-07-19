import os
import sys
from tinyenv.flags import flags
_FLAGS = flags()
#current path:   /tinysrc/mask-rcnn/notebooks/testTinyMind.py
def save_to_file(file_name, contents):
    fh = open(file_name, 'w+')
    fh.write(contents)
    fh.close()

save_to_file(_FLAGS.output_dir+'mobiles.txt', os.getcwd())
