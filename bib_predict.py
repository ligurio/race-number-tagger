# -*- coding: utf-8 -*-

from keras.models import load_model
from keras.preprocessing import image
from bib_prepare_dataset import img_size
import sys
import settings
import numpy as np

def main():
    img_path = sys.argv[1]
    print "Image:", img_path
    model = load_model(settings.MODEL_FILE_REAL)
    img = image.load_img(img_path, target_size=img_size(img_path), grayscale=True)
    arr = np.array(img).reshape((settings.BOX_HEIGHT, settings.BOX_WIDTH, 1))
    arr = np.expand_dims(arr, axis=0)
    prediction = model.predict(arr)[0]
    bestclass = ''
    bestconf = -1
    for n in [0,1,2,3,4,5,6,7,8,9]:
            if (prediction[n] > bestconf):
                    bestclass = str(n)
                    bestconf = prediction[n]

    bestclass, bestconf = predict_image(image.img_to_array(img))
    print 'I think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.'

if __name__ == "__main__":
    main()
