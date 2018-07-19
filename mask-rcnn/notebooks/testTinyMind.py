import os
import sys
from tinyenv.flags import flags
_FLAGS = flags()
def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

save_to_file('mobiles.txt', os.getcwd())
