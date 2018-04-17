### Description

- скачать csv с https://requester.mturk.com/
- проревьюить качество аннотаций
	mturk-csv-review.py
	mturk-csv-single-review.py
	mturk-csv-mark-rejected.py
- конвертировать в JSON
	mturk-csv-json.py
- в задании на Amazon Mechanical Turk есть указание выделять номер целиком. Но
  так как мы будем обучать модель на отдельных цифрах, то нам нужно сделать
аннотацию с боксами для отдельных цифр. Скрипт mturk-split-boxes-full-numbers.py
делает это. Скрипт mturk-json-calc-average-size.py вычисляет по аннотации 75-й
процентиль размера бокса для каждой цифры.
- Далее скриптом mturk-box-intersection.py мы делим каждое изображение на кусочки с
  размерами, полученными предыдущим скриптом и обновляем файл аннотации.

### How-To Use:

- обучение на данных MNIST ```bib_learn_mnist.py```
- обучение на реальных изображениях ```bib_learn_custom_dataset.py```
- использование ```bib_predict.py```
