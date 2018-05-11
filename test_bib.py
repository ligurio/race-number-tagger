import unittest
import random
from bib_prepare_dataset import overlap_area
from bib_prepare_dataset import is_intersected
from bib_prepare_dataset import max_overlap_box
from bib_prepare_dataset import split_box_per_number
from bib_prepare_dataset import image_split

def random_box(box):
    x1 = random.randint(0, box['left'] + box['width'] + 100)
    x2 = random.randint(x1, box['left'] + box['width'] + 100)
    width = x2 - x1

    y1 = random.randint(0, box['top'] + 100)
    y2 = random.randint(y1, box['top'] + 100)
    height = y2 - y1

    return {'label': '', 'left': x1, 'top': y2, 'width': width, 'height': height }


class DatasetPreparationTest(unittest.TestCase):

    def test_boxes_overlap(self):

	box1 = { 'left': 3, 'top': 4, 'width': 3, 'height': 3 }
	box2 = { 'left': 5, 'top': 7, 'width': 3, 'height': 4 }
	self.assertEqual(overlap_area(box1, box2), 1)
	self.assertEqual(overlap_area(box2, box1), 1)
	self.assertEqual(is_intersected(box1, box2), True)

	box1 = { 'left': 9, 'top': 11, 'width': 3, 'height': 5 }
	box2 = { 'left': 9, 'top': 15, 'width': 3, 'height': 5 }
	self.assertEqual(overlap_area(box1, box2), 3)
	self.assertEqual(overlap_area(box2, box1), 3)
	self.assertEqual(is_intersected(box1, box2), True)

	box1 = { 'left': 13, 'top': 6, 'width': 3, 'height': 4 }
	box2 = { 'left': 14, 'top': 3, 'width': 3, 'height': 2 }
	self.assertEqual(is_intersected(box1, box2), True)
	self.assertEqual(overlap_area(box1, box2), 2)
	self.assertEqual(overlap_area(box2, box1), 2)

	box1 = { 'left': 2, 'top': 15, 'width': 4, 'height': 5 }
	box2 = { 'left': 3, 'top': 14, 'width': 1, 'height': 3 }
	self.assertEqual(is_intersected(box1, box2), True)
	self.assertEqual(overlap_area(box1, box2), 3)
	self.assertEqual(overlap_area(box2, box1), 3)

	box1 = { 'left': 13, 'top': 14, 'width': 2, 'height': 3 }
	box2 = { 'left': 9, 'top': 15, 'width': 3, 'height': 5 }
	self.assertEqual(is_intersected(box1, box2), False)

	box1 = { 'left': 9, 'top': 10, 'width': 3, 'height': 4 }
	box2 = { 'left': 9, 'top': 15, 'width': 3, 'height': 4 }
	self.assertEqual(is_intersected(box1, box2), False)


    def test_max_intersection_label(self):

        boxes = [{'height': 68, 'width': 186, 'top': 490, 'left': 421, 'label': '0'},
                {'height': 178, 'width': 59, 'top': 443, 'left': 582, 'label': '1'},
                {'height': 565, 'width': 96, 'top': 783, 'left': 251, 'label': '2'},
                {'height': 316, 'width': 142, 'top': 762, 'left': 525, 'label': '3'},
                {'height': 70, 'width': 41, 'top': 241, 'left': 627, 'label': '4'},
                {'height': 450, 'width': 52, 'top': 769, 'left': 143, 'label': '5'},
                {'height': 266, 'width': 354, 'top': 278, 'left': 64, 'label': '6'},
                {'height': 69, 'width': 29, 'top': 353, 'left': 644, 'label': '7'},
                {'height': 13, 'width': 304, 'top': 710, 'left': 265, 'label': '8'},
                {'height': 133, 'width': 52, 'top': 679, 'left': 510, 'label': '9'}
        ]
        box = { 'label': '', 'left': 240, 'top': 800, 'width': 460, 'height': 330 }
        overlapBox, area = max_overlap_box(boxes, box, threshold = 0.4)
        self.assertEqual(overlapBox['label'], '3')
        self.assertEqual(area, 41464)


    def test_random_max_intersection_label(self):

        box = { 'label': '', 'left': 240, 'top': 800, 'width': 460, 'height': 330 }
        boxes = []
        areas = []
        for i in range(0, 10):
            rdict = random_box(box)
            area = overlap_area(rdict, box)
            boxes.append(rdict)
            areas.append(area)

        _, area = max_overlap_box(boxes, box, threshold = 0.4)
        self.assertEqual(area, max(areas))

    def test_split_box_per_number(self):
        # [{'height': 123, 'width': 69, 'top': 959, 'left': 1584, 'label': '1'},
        #  {'height': 123, 'width': 69, 'top': 959, 'left': 1653, 'label': '5'},
        #  {'height': 123, 'width': 69, 'top': 959, 'left': 1722, 'label': '2'}]
        box = {u'width': 207, u'top': 959, u'height': 123, u'label': u'152', u'left': 1584} 
        label = ''
        for b in split_box_per_number(box):
            label += b['label']
        self.assertEqual(box['label'], label)
         
    def test_image_split(self):
        box_h = 157
        box_w = 68
        img_width = 1000
        img_height = 1000
        pieces = [ p for p in image_split(img_width, img_height, box_w, box_h) ]
        self.assertEqual(len(pieces), 336)
        

if __name__ == '__main__':
    unittest.main()
