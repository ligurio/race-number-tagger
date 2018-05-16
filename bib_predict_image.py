# -*- coding: utf-8 -*-

from keras.models import load_model
from keras.preprocessing import image
from bib_prepare_dataset import image_split
from bib_prepare_dataset import image_crop
from bib_prepare_dataset import img_size
from bib_prepare_dataset import draw_box
import sys
import settings
import numpy as np


def main():
    img_path = sys.argv[1]
    print "Image:", img_path
    model = load_model(settings.MODEL_FILE)
    img_width, img_height = img_size(img_path)

    found_boxes = []
    for box in image_split(img_width, img_height, settings.BOX_WIDTH, settings.BOX_HEIGHT):
        img = image_crop(box, img_path)
        bestclass, bestconf = predict_image(numpy.asarray(img))
        arr = np.array(img).reshape((settings.BOX_HEIGHT, settings.BOX_WIDTH, 1))
        arr = np.expand_dims(arr, axis=0)
        prediction = model.predict(arr)[0]
        bestclass = ''
        bestconf = -1
        for n in [0,1,2,3,4,5,6,7,8,9]:
                if (prediction[n] > bestconf):
                        bestclass = str(n)
                        bestconf = prediction[n]
        print 'I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.'

    filename = os.path.basename(img_path).split('.')[0] + '-' + id_generator() + '.jpg'
    dst_path = os.path.join(os.path.dirname(img_path), filename)
    draw_box(found_boxes, img_path, dst_path)
    print "Annotated image %s", dst_path


if __name__ == "__main__":
    main()
