Убрать аннотацию для 0029_1311289.jpg

Каждый раз после соревнования возникают одни и те же проблемы: у участника
соревнования - найти фотографии, на которых он присутствует, у фотографов -
найти спортсменов, которые попал в кадр его фотоаппарата. В случае решения этих
проблем фотографии могут быть как доступны бесплатно для участников так и за
разумную плату. Но нас интересует именно технология, способствующая решению
обоих задач, поэтом коммерческую составляющую в случае решения этих задач мы
оставим за рамками нашего проекта.  Очевидно, что для решения обоих задач нужно
каким-то образом идентифицировать участников соревнования на каждой из
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
- нарисовать боксы на фотографиях с помощью ```mturk-csv-review.py```
- проревьюить качество разметки с ```feh --action1 'echo %F | xclip -i' image_dir```
- заполнить поле RequestorFeedback в csv файле и загрузить файл обратно в MTurk
- удалить старый набор данных ```rm -rf data/race_numbers```
- ```virtualenv pip```
- ```source pip/bin/activate```
- установить модули: ```pip install -r requirement.txt```
- ```make prepare```
- ```make train```
- использование - ```bib_predict.py sample.jpg```
- мониторинг обучения модели: ```mkdir train_log; tensorboard --logdir=train_log```

Copyright 2018 Sergey Bronnikov
