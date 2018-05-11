### TODO

- двойные номера для эстафет и экиденов https://www.facebook.com/runningexpertise/photos/pcb.1970190036343349/1970200376342315/?type=3&theater, https://www.facebook.com/runningexpertise/photos/pcb.1970190036343349/1970205253008494/?type=3&theater
- поддержать видео
- использовать номера и распознавалку вместо чипов
- интегрироваться с https://www.imgix.com/
- сделать демо: показывать картинку с размеченными номерами но список номеров текстом не показывать

### Похожие проекты:

- https://github.com/KateRita/bib-tagger

### Модель

- сверточные сети и Faster R-CNN
- [архитектура](https://matthewearl.github.io/2016/05/06/cnn-anpr/) и [модель](https://github.com/matthewearl/deep-anpr/blob/master/model.py) для распознавания автомобильных номеров
- рекомендуемые алгоритмы: свертки и нейросети

### Наборы данных

#### Готовые

- https://www.kaggle.com/debdoot/bdrw
- https://www.kaggle.com/olgabelitskaya/svhn-preproccessed-fragments
- [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/) (73257 digits for training, 26032 digits for testing, 531131 additional)
- сделать с помощью [imgaug](thttps://github.com/aleju/imgaug)

#### Подготовка своего набора данных:

- для обучения нужны десятки тысяч фотографий (чем больше, тем лучше для улучшения качества)
- подготовка изображений для обучения https://habrahabr.ru/post/311558/
- найти фото на [фотостоках](http://www.geran.in/2015/10/09/marathon_photo_sites/), [Probeg](http://probeg.org/kalend/rezult.php), http://probeg.org/probegim.php
- [загрузить](https://yandex.ru/blog/fotki/51820): ```python3 ../../ya.py -a 190988 -d . pvolan```
- убрать дубликаты: ```fdupes -S -R data/``` или https://gist.github.com/bobuk/4522091
- перевернуть:```find . -type f -iname "*.jpg" -exec exiftran -aig "{}" \;```
- убрать фотографии без номеров
- сохранить список фото: ```mtree -k type,sha256digest -c -p runners > spec.txt; ls -laR > ls-laR.txt```
- ```echo "objects_to_find,image_url" > list.csv```
- ```ls -1 | while read f; do echo $f,https://s3.amazonaws.com/running-ml/$f; done >> list.csv```
- загрузить для аннотирования:
	- **[Tutorial: Annotating images with bounding boxes using Amazon Mechanical Turk](https://blog.mturk.com/tutorial-annotating-images-with-bounding-boxes-using-amazon-mechanical-turk-42ab71e5068a)**
	- [Tutorial: How to verify crowdsourced training data using a Known Answer Review Policy](https://blog.mturk.com/tutorial-how-to-verify-crowdsourced-training-data-using-a-known-answer-review-policy-85596fb55ed)
	- [Tutorial: How to label thousands of images using the crowd](https://blog.mturk.com/tutorial-how-to-label-thousands-of-images-using-the-crowd-bea164ccbefc)
	- https://requester.mturk.com/batches/3180686
	- https://s3.console.aws.amazon.com/s3/object/running-ml/0003_1683929.jpg?region=us-east-1&tab=overview
 
- https://s3.console.aws.amazon.com
- потренироваться:
  - https://workersandbox.mturk.com/
  - https://mechanicalturk.sandbox.amazonaws.com
- распознавать:
  - https://mechanicalturk.amazonaws.com
  - https://requester.mturk.com/
  - https://worker.mturk.com


#### Ручная разметка фотографий

- https://github.com/ZlodeiBaal/Base_prepearing
- https://github.com/tzutalin/labelImg
- https://github.com/cvhciKIT/sloth
- http://www.robots.ox.ac.uk/~vgg/software/via/
- https://github.com/CSAILVision/LabelMeAnnotationTool
- Яндекс.Толока
- https://www.mturk.com/ ($0.04 per image)
- [Google Vision](https://cloud.google.com/vision/)


### Поиск аналогичных сервисов

- racenumbertagger.com: Как работает [не понятно](http://www.racenumbertagger.com/screenshots/), но похоже на ML.  Стоимость: $35 AUD monthly, $350 AUD yearly. RNT can be downloaded from here http://www.racenumbertagger.com/installer/DistB/setup.exe, http://www.racenumbertagger.com/installer/Documentation/RNT_Quick_Start_Guide.pdf
- pic2go.com.au: Для распознования использует QR метки. Описание [метода](http://www.pic2go.com.au/how-it-works/index.html).
- runnerscanner.com: Тоже QR код? Выглядит как хуйня на палочке. "Create bibs with our special tags" -- похоже, что специальным образом заранее размечают номера. Стоимость непонятна.
- marathon-photos.com: Есть ли там автоматическое распознование?
- bibtaggers: "А давайте все руками разметим!" http://bibtaggers.ru/about/
- marathon-photo.ru: руками
- sport-images.ru
- photorun.ru
- nabegu.spb.ru (бесплатно)
- [sport-vision](https://www.facebook.com/SportVision.Russia/) https://www.facebook.com/andrey.perep (Калашникова его знает)
- https://www.facebook.com/BibTagger/
- сервисы для аннотирования фотографий (тот же амазон тюрк и толока)
- http://www.capcitysportsmedia.com/Bib-Tagging-Smugmug, стоимость 0.05$, время 24-48 часов, разметку делают руками
- https://www.flashframe.io/#bg3 (похоже что используют Deep Learning - https://www.flashframe.io/blog/the-cutting-edge-of-race-photo-tagging-software/)
- https://www.comerphotos.com/Bib-Tagging/Pricing ~0.04$
