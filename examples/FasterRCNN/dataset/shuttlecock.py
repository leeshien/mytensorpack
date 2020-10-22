import os
import numpy as np
import json
from dataset import DatasetSplit, DatasetRegistry


from tensorpack.utils import viz
from tensorpack.utils.palette import PALETTE_RGB

from config import config as cfg
from utils.np_box_ops import area as np_area
from utils.np_box_ops import iou as np_iou
from common import polygons_to_mask

__all__ = ["register_ic"]


class Shuttlecock(DatasetSplit):
    def __init__(self, base_dir, split):
        assert split in ["train", "val"]
        base_dir = os.path.expanduser(base_dir)
        self.imgdir = os.path.join(base_dir, split)
        assert os.path.isdir(self.imgdir), self.imgdir

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        div = det(xdiff, ydiff)
        if div == 0:
           raise Exception('lines do not intersect')
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        return x,y

    def training_roidbs(self):
        files = [f for f in os.listdir(self.imgdir) if os.path.isfile(os.path.join(self.imgdir, f))]
        jsonfiles = [f for f in files if f.endswith('.json')]
        imgfiles = [f for f in files if f.endswith('.jpeg') or f.endswith('.jpg')]

        ret = []
        boxes = []
        segs = []        
        for fn in jsonfiles:
            json_file = os.path.join(self.imgdir, fn)
            with open(json_file) as f:
                obj = json.load(f)

            try:
                fname = [filename for filename in imgfiles if fn.split('.')[0] in filename][0] #image filename
                fname = os.path.join(self.imgdir, fname)

                roidb = {"file_name": fname}

                annos = obj["shapes"]

                poly = np.asarray(poly)
                maxxy = poly.max(axis=0)
                minxy = poly.min(axis=0)
                
                boxes.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])            
                    
                N = 1
                roidb["boxes"] = np.asarray(boxes, dtype=np.float32)
                roidb["segmentation"] = [[poly]]

                roidb["class"] = np.ones((N, ), dtype=np.int32)
                roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
                ret.append(roidb) 
            except Exception as e:
                print(fn, ' not does matched with any image')
                pass  

            return ret

def register_shuttlecock(basedir):
    for split in ["train", "val"]:
        print('split: ', split)
        name = "shuttlecock_" + split
        DatasetRegistry.register(name, lambda x=split: Shuttlecock(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["BG", "shuttlecock"])
        print(DatasetRegistry._metadata_registry)
