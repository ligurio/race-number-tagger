### TODO

- слишком много изображений без цифр в наборе для обучения
- научиться увеличивать изображения в наборе данных MNIST до размера моих кусков
- обучить модель сначала на наборе данных MNIST, потом на моем наборе данных

- ~~сделать скрипт, который будет резать изображения на куски, сохранять их в
файлы для обучения и ставить в соответствие цифру~~
- ~~вычислить размер куска для резки фотографий (75-й процентиль)~~
- ~~сделать скрипт для обновления аннотаций (каждый бокс - только одна цифра)~~
- ~~создать проект по аннотации фотографий https://requester.mturk.com/create/projects/1150899/edit~~
- ~~аннотировать фотографии из обучающей выборки~~
- ~~прочитать Deep Learning with Python~~
- ~~подготовить обучающую выборку~~
- ~~научиться обучать и распознавать на простом примере~~
- ~~сделать датасет из картинок с номерами в разных местах фотографии, заодно будут аннотации. Для шума достаточно взять фото с забегов или [imgaug](thttps://github.com/aleju/imgaug). Проверить модель на реальных фотографиях.~~
- ~~сделать датасет из датасета c номерами домов с больше, чем одним номером на фото и проверить модель на реальных фотографиях~~
- ~~попробовать датасет с номерами домов + распознование с crop c наложением~~
- ~~загрузить фотографии для обучения~~
- ~~найти фотографии с забегов для тестов~~
- ~~найти датасеты и скрипты похожих задач с [kaggle](https://www.kaggle.com/)~~
- двойные номера для эстафет и экиденов https://www.facebook.com/runningexpertise/photos/pcb.1970190036343349/1970200376342315/?type=3&theater, https://www.facebook.com/runningexpertise/photos/pcb.1970190036343349/1970205253008494/?type=3&theater
- вырезка спортсмена из фотографии [Semantic segmentation](https://github.com/aurora95/Keras-FCN)
- распознавать спортсменов на видео
- использовать номера и распознавалку вместо чипов
- интегрироваться с https://www.imgix.com/
- в первую очередь сделать демо: показывать картинку с размеченными номерами и список номеров в тексте не показывать
- сервис: проверять тип файла и не принимать неграфические файлы, сохранять SHA256 для файлов для дальнейших разборок если будут

### Теория

### Похожие проекты:

- https://github.com/KateRita/bib-tagger

- https://github.com/matthewearl/deep-anpr
- https://github.com/fizyr/keras-retinanet
- https://github.com/potterhsu/SVHNClassifier (multi-digit)
- https://github.com/penny4860/SVHN-deep-digit-detector
- https://itaicaspi.github.io/SVHN-Multi-Digit-torch/
- https://github.com/kjw0612/awesome-deep-vision
- можно использовать как основу своей модели https://github.com/keras-team/keras/issues/3928
- пример https://github.com/nate-parrott/juypter-notebooks/blob/master/svhn-keras.ipynb
- https://gist.github.com/bellbind/6698114f1c601d45b7bdaa5516284707
- https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
- https://github.com/keras-team/keras/tree/master/examples
- https://gggdomi.github.io/keras-workshop/notebook.html
- https://github.com/yeephycho/tensorflow_input_image_by_tfrecord
- [Number plate recognition with Tensorflow](https://matthewearl.github.io/2016/05/06/cnn-anpr/)
- https://github.com/emedvedev/attention-ocr
- +video https://github.com/riadhayachi/faster-rcnn-keras
- https://github.com/jinfagang/keras_frcnn
- https://github.com/fizyr/keras-retinanet

#### Classification

- https://habrahabr.ru/company/recognitor/blog/221891/
- https://www.opennet.ru/opennews/art.shtml?num=47950
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf)
- http://cocodataset.org/#home
- https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/
- Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, and Vinay Shet (2013). Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks. [arXiv:1312.6082](https://arxiv.org/abs/1312.6082) [cs.CV]
- Pierre Sermanet, Soumith Chintala, and Yann LeCun (2012). Convolutional Neural Networks Applied to House Numbers Digit Classification. [arXiv:1204.3968](https://arxiv.org/abs/1204.3968) [cs.CV]
- Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng (2011). Reading Digits in Natural Images with Unsupervised Feature Learning. *NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011*. ([Page](http://ufldl.stanford.edu/housenumbers/)|[PDF](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf))
- Mark Grundland, and Neil A. Dodgson (2007). Decolorize: Fast, contrast enhancing, color to grayscale conversion. *Pattern Recognition*, **40** (11). [Page](http://dx.doi.org/10.1016/j.patcog.2006.11.003)
- [Convolutional Neural Networks Applied to House Numbers Digit Classification](https://arxiv.org/pdf/1204.3968.pdf)
- [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf)
- [Street View House Number Recognition using Deep Convolutional
Neural Networks](http://www.iitp.ac.in/~arijit/dokuwiki/lib/exe/fetch.php?media=courses:2017:cs551:03_report.pdf)
- [Who is the best in SVHN?](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#5356484e)
- https://github.com/chongyangtao/Awesome-Scene-Text-Recognition
- статьи https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
- курс http://cs231n.stanford.edu/
- A New Multi-modal Technique for Bib Number/Text Detection in Natural Images - Sangheeta RoyPalaiahnakote ShivakumaraEmail authorPrabir MondalR. RaghavendraUmapada PalTong Lu
- A new multi-modal approach to bib number/text detection and recognition in Marathon images - Palaiahnakote Shivakumara, R. Raghavendra, Longfei Qin, Kiran B. Raja, Tong Lu, Umapada Pal
- Racing Bib Number Recognition - Idan Ben-Ami, Tali Basha, Shai Avidan

#### Segment detection

- https://softwaremill.com/counting-objects-with-faster-rcnn/
- https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Erhan_Scalable_Object_Detection_2014_CVPR_paper.pdf
- https://arxiv.org/pdf/1506.01497.pdf
- https://pdfs.semanticscholar.org/713f/73ce5c3013d9fb796c21b981dc6629af0bd5.pdf
- [Multiple View Semantic Segmentation for Street View Images](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.294.2706&rep=rep1&type=pdf)
- Robust Wide Baseline Stereo from. Maximally Stable Extremal Regions. J. Matas. 1,2. , O. Chum. 1. , M. Urban. 1. , T. Pajdla.
- Linear Time Maximally Stable Extremal Regions. David Nistér and Henrik
- Maximally Stable Colour Regions for Recognition and Matching. Per-Erik Forssén
- [COLOR BLOB SEGMENTATION BY MSER ANALYSIS](https://pdfs.semanticscholar.org/174e/27471718a33aa5f18aa9682f410cc50c3cb1.pdf)

### Обучение

- Google Cloud https://cloud.google.com/ml-engine/ ([пример](https://github.com/emedvedev/attention-ocr))
- GPU, CPU (может быть в 40 раз дольше, чем CPU)
- Google AutoML https://www.blog.google/topics/google-cloud/cloud-automl-making-ai-accessible-every-business/

### Модель

- сверточные сети и Faster R-CNN
- keras, tensorflow
- разрешение фотографии критично для распознования, нужен crop
- [архитектура](https://matthewearl.github.io/2016/05/06/cnn-anpr/) и [модель](https://github.com/matthewearl/deep-anpr/blob/master/model.py) для распознавания автомобильных номеров

### Рекомендации от Паши

- для обучения нужны десятки тысяч фотографий (чем больше, тем лучше для улучшения качества)
- рекомендуемые алгоритмы: свертки и нейросети
- рекомендуемые инструменты: tensorflow и [keras](https://keras.io/) (Python binding)
- для обучения потребуются вычислительные ресурсы (возможно видеокарты, потому что на cpu можно неделю обучать), графические файлы требовательны к ресурсам
- книга Deep learning with Python
- разрешение фотографии критично для распознования и здесь нужен crop

### Наборы данных

#### Готовые

- https://www.kaggle.com/debdoot/bdrw
- https://www.kaggle.com/olgabelitskaya/svhn-preproccessed-fragments
- [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/) (73257 digits for training, 26032 digits for testing, 531131 additional)

#### Подготовка своего набора данных:

- подготовка изображений для обучения https://habrahabr.ru/post/311558/
- источники реальных фотографий:
 - [Фотостоки](http://www.geran.in/2015/10/09/marathon_photo_sites/)
 - [Probeg](http://probeg.org/kalend/rezult.php).
 - http://probeg.org/probegim.php
- ```python3 ../../ya.py -a 190988 -d . pvolan```
- вычистка
 - ```fdupes -S -R data/```
 - ```find . -type f -iname "*.jpg" -exec exiftran -aig "{}" \;```
 - ```montage photo1.jpg photo2.jpg photo3.jpg photo4.jpg montage.jpg```
 - ```ls -laR > ls-laR.txt```
 - ```for i in `seq 1 1 402`; do dir=0$i; [[ -e $dir  ]] || echo $dir; done | wc -l```
 - убрать фотографии без номеров
 - ```mtree -k type,sha256digest -c -p runners > spec.txt```
 - убрать дублирующиеся фотографии https://gist.github.com/bobuk/4522091
- **[Tutorial: Annotating images with bounding boxes using Amazon Mechanical Turk](https://blog.mturk.com/tutorial-annotating-images-with-bounding-boxes-using-amazon-mechanical-turk-42ab71e5068a)**
- [Tutorial: How to verify crowdsourced training data using a Known Answer Review Policy](https://blog.mturk.com/tutorial-how-to-verify-crowdsourced-training-data-using-a-known-answer-review-policy-85596fb55ed)
- [Tutorial: How to label thousands of images using the crowd](https://blog.mturk.com/tutorial-how-to-label-thousands-of-images-using-the-crowd-bea164ccbefc)
 

```
- ~/sources/runphoto/data/runners/processed/running/mturk_1_100
- https://requester.mturk.com/batches/3180686
- https://s3.console.aws.amazon.com/s3/object/running-ml/0003_1683929.jpg?region=us-east-1&tab=overview
- echo "objects_to_find,image_url" > list.csv
- ls -1 | while read f; do echo $f,https://s3.amazonaws.com/running-ml/$f; done >> list.csv
```
 
С Яндекс.Фотки удобно скачивать с помощью скрипта https://yandex.ru/blog/fotki/51820.
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
- с помощью Яндекс.Толока
- https://www.mturk.com/ ($100-200)
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
- сервисы для аннотирования фотографий (тот же амазон тюрк и толока). стоимость на моем сервисе должна быть не выше стоимости у них
- http://www.capcitysportsmedia.com/Bib-Tagging-Smugmug, стоимость 0.05$, время 24-48 часов, разметку делают руками
- https://www.flashframe.io/#bg3 (похоже что используют Deep Learning - https://www.flashframe.io/blog/the-cutting-edge-of-race-photo-tagging-software/)
- https://www.comerphotos.com/Bib-Tagging/Pricing ~0.04$
