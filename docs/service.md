#### Описание сервиса

• Предварительный поиск номера – обнаружение области в которой содержится номер
• Нормализация номера – определение точных границ номера, нормализация констраста
• Распознавание текста – чтение всего что нашлось в нормализованном изображении
Это базовая структура. Конечно, в ситуации, когда номер линейно расположен и
хорошо освещён, а у Вас в распоряжение отличный алгоритм распознавания текста,
первые два пункта отпадут. В некоторых алгоритмах могут объединяться поиск
номера и его нормализация.

Фокус сервиса - разметка фотографий и видео для фотографов.
- распознавание номера
- автоматическое тегирование фотографий
	https://github.com/karpathy/neuraltalk2
	https://twitter.com/NinoVerde/status/974008265393754113
	https://github.com/anuragmishracse/caption_generator

- для страницы с тестом ограничивать количество подключений
([nginx](https://nginx.org/en/docs/http/ngx_http_limit_conn_module.html) или pf + httpd)
- домен свободный tagphoto.su
- архитектура:
  - UI авторизует пользователя и кладет в БД токен и список файлов из хранилища
  - manager мониторит новые заказы на распознавание в БД и при появлении начинает распознавать по очереди. после завершения
  отсылает уведомление по почте.
  - фотографии распознают воркеры, которых в зависимости от нагрузки может быть разное количество. Все они находятся за балансером (haproxy, relayd), так будет удобнее мониторить нагрузку и проще будет потом вынести этих воркеров на другой хост.
  каждый воркер висит на localhost:PORT.
- Ruby template https://github.com/ept/saas-template
- SLA: https://www.aptible.com/legal/service-level-agreement/, https://www.saashost.net/service-level-agreement-2/
- Python template:
  - https://pypi.python.org/pypi/djaodjin-saas/0.2.6
  - https://stackoverflow.com/questions/9924169/how-to-create-saas-application-with-python-and-django
  - http://tastypieapi.org/
  - https://github.com/cypreess/django-plans
  - (!) https://github.com/sloria/cookiecutter-flask
  - https://github.com/zhaque/django-saas-kit
  - https://www.infoq.com/presentations/saas-python


### Хранилище для загрузки фото на сервис

##### Рекомендации облачных хранилищ:

- sync.com, dropbox, google drive https://www.cloudwards.net/best-cloud-storage-for-photographers/
- sync.com, pcloud, idrive https://www.cloudwards.net/best-online-storage-for-photos/
- dropbox, google drive, onedrive http://www.techradar.com/how-to/photography-video-capture/cameras/best-cloud-storage-for-photos-6-top-options-tested-and-rated-1320891
- sync.com, onedrive, dropbox http://www.cloudstorageboss.com/photographers/

**[Распределение рынка среди облачных хранилищ](https://blog.cloudrail.com/cloud-storage-report-2017/)**

##### API

###### Storage

- Google Drive API: [PyDrive](https://pythonhosted.org/PyDrive/), [GDrive API](https://developers.google.com/drive/v3/web/quickstart/python), [Go Lang](https://gist.github.com/atotto/86fa30668473b41eeac7d750e5ad5f5c)
- [Yandex.Fotki API](https://tech.yandex.ru/fotki/)
- Dropbox API: https://www.dropbox.com/developers-v1/core/start/python

###### Payment

- http://fastspring.com/plans/
- https://stripe.com/atlas/guides/saas-pricing
