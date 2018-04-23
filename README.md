Каждый раз после соревнования возникает одна и та же проблемы: у участника
соревнования - найти фотографии, на которых он присутствует, у фотографов -
найти спортсменов, которые попал в кадр его фотоаппарата. В случае решения этих
проблем фотографии могут быть как доступны бесплатно для участников так и за
разумную плату. Но нас интересует именно технология, способствующая решению
обоих задач, поэтом коммерческую составляющую в случае решения этих задач мы
оставим за рамками нашего проекта.  Очевидно, что для решения обоих задач нужно
каким-то образом идентифицировать участникаов соревнования на каждой из
фотографий и сделать это можно если вы знаете человека лично ("узнаете его в
лицо") или можете прочитать номер участника на фотографии.  Для первого случая
вы должны знать всех участников, что для крупных соревнований маловероятно, или
иметь достаточно свободного времени, чтобы посмотреть на каждую фотографию и
для каждой из них подготовить список номер участников на этой фотографии.  Есть
предположение, что для второго случая возможно решение с помощью машинного
обучения. Если получится сделать технологию с распознаванием номеров с
вероятность выше 50%, то можно будет подумать создании коммерческого сервиса.


### How-To Use:

- скачать csv с https://requester.mturk.com/
- проревьюить качество аннотаций ```mturk-csv-review.py```
- подготовить набор данных и аннотацию ```bib_prepare_dataset.py.py```
в задании на Amazon Mechanical Turk есть указание выделять номер целиком. Но
так как мы будем обучать модель на отдельных цифрах, то нам нужно сделать
аннотацию с боксами для отдельных цифр.
делает это.
- обучение на данных MNIST - ```bib_learn_mnist.py```
- обучение на реальных изображениях ```bib_learn_custom_dataset.py```
- использование ```bib_predict.py```


### Другие похожие проекты:

- https://github.com/KateRita/bib-tagger
- https://github.com/matthewearl/deep-anpr
- https://github.com/fizyr/keras-retinanet
- https://github.com/potterhsu/SVHNClassifier (multi-digit)
- https://github.com/penny4860/SVHN-deep-digit-detector
- https://itaicaspi.github.io/SVHN-Multi-Digit-torch/
- https://github.com/kjw0612/awesome-deep-vision
- https://github.com/KateRita/bib-tagger
- можно использовать как основу своей модели https://github.com/keras-team/keras/issues/3928
- пример https://github.com/nate-parrott/juypter-notebooks/blob/master/svhn-keras.ipynb
- https://gist.github.com/bellbind/6698114f1c601d45b7bdaa5516284707
- https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
- https://github.com/keras-team/keras/tree/master/examples
- https://gggdomi.github.io/keras-workshop/notebook.html
- https://github.com/yeephycho/tensorflow_input_image_by_tfrecord
- [Number plate recognition with Tensorflow](https://matthewearl.github.io/2016/05/06/cnn-anpr/)
- https://github.com/emedvedev/attention-ocr
- Faster R-CNN
  - +video https://github.com/riadhayachi/faster-rcnn-keras
  - https://github.com/jinfagang/keras_frcnn
  - https://github.com/fizyr/keras-retinanet
