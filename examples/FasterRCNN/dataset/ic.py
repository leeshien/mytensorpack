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


class ICDemo(DatasetSplit):
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

        ret = []
        for fn in jsonfiles:
            json_file = os.path.join(self.imgdir, fn)
            with open(json_file) as f:
                obj = json.load(f)

            fname = obj["imagePath"] #image filename
            fname = os.path.join(self.imgdir, fname)

            roidb = {"file_name": fname}

            annos = obj["shapes"]

            lines, poly, box = [], [], []

            lines.append([annos[0]["points"][0], annos[7]["points"][0]]) # left line
            lines.append([annos[1]["points"][0], annos[2]["points"][0]]) # top line
            lines.append([annos[3]["points"][0], annos[4]["points"][0]]) # right line
            lines.append([annos[6]["points"][0], annos[5]["points"][0]]) # bottom line

            for i, anno in enumerate(annos):
                poly.append(np.asarray(anno["points"][0]))
            poly = np.asarray(poly)
            maxxy = poly.max(axis=0)
            minxy = poly.min(axis=0)
            
            box.append([minxy[0], minxy[1], maxxy[0], maxxy[1]])            
                
            N = 1
            roidb["boxes"] = np.asarray(box, dtype=np.float32)
            roidb["segmentation"] = [[poly]]

            roidb["class"] = np.ones((N, ), dtype=np.int32)
            roidb["is_crowd"] = np.zeros((N, ), dtype=np.int8)
            ret.append(roidb)       

        return ret

def register_ic(basedir):
    for split in ["train", "val"]:
        print('split: ', split)
        name = "ic_" + split
        DatasetRegistry.register(name, lambda x=split: ICDemo(basedir, x))
        DatasetRegistry.register_metadata(name, "class_names", ["BG", "IC"])

if __name__ == '__main__':
    basedir = '~/data/ic'
    roidbs = ICDemo(basedir, "train").training_roidbs()
    print("#images:", len(roidbs))

    from viz import draw_annotation
    from tensorpack.utils.viz import interactive_imshow as imshow
    import cv2
    for r in roidbs:
        im = cv2.imread(r["file_name"])
        vis = draw_annotation(im, r["boxes"], r["class"], r["segmentation"])
        imshow(vis)