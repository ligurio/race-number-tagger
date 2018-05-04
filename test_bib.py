import unittest
from bib_prepare_dataset import overlap_area
from bib_prepare_dataset import label_with_max_intersection
from bib_prepare_dataset import split_box_per_number
from bib_prepare_dataset import image_split

class DatasetPreparationTest(unittest.TestCase):

    def test_boxes_intersection(self):
        box1 = { 'left': 100, 'top': 600, 'width': 200, 'height': 500 }
        box2 = { 'left': 240, 'top': 800, 'width': 460, 'height': 330 }
        self.assertEqual(overlap_area(box1, box2), 7800)

    def test_max_intersection_label(self):
        # FIXME
        boxes = [ { 'label': '1', 'left': 100, 'top': 600, 'width': 200, 'height': 500 }, { 'label': '1', 'left': 100, 'top': 600, 'width': 200, 'height': 500 } ]
        box = { 'label': '', 'left': 240, 'top': 800, 'width': 460, 'height': 330 }
        self.assertEqual(label_with_max_intersection(boxes, box), '1')

    def test_split_box_per_number(self):
        # [{'height': 123, 'width': 69, 'top': 959, 'left': 1584, 'label': '1'},
        #  {'height': 123, 'width': 69, 'top': 959, 'left': 1653, 'label': '5'},
        #  {'height': 123, 'width': 69, 'top': 959, 'left': 1722, 'label': '2'}]
        box = {u'width': 207, u'top': 959, u'height': 123, u'label': u'152', u'left': 1584} 
        label = ''
        for b in split_box_per_number(box):
            label = label + b['label']
        self.assertEqual(box['label'], label)
         
    def test_image_split(self):
        box_h = 157
        box_w = 68
        img_width = 1000
        img_height = 1000
        pieces = [ p for p in image_split(img_width, img_height, box_w, box_h) ]
        self.assertEqual(len(pieces), 84)
        

if __name__ == '__main__':
    unittest.main()
