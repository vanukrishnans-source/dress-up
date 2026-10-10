import sys, glob, numpy as np
from PIL import Image
for f in glob.glob(sys.argv[1] + "/**/*.png.raw", recursive=True):
    w, h = map(int, open(f[:-4] + ".size").read().split())
    Image.fromarray(np.fromfile(f, np.uint8).reshape(h, w, 3)).save(f[:-4])
