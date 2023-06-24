# vacancy_parser

Решение кейса «РосКапСтрой»: Разработка решения по обработке вакансий

### Установка нужных библиотек:

```pip install requirements.txt``` - установка зависимостей

```python -m nltk.downloader stopwords``` - загрузка стоп-слов nltk

Загрузка модели spacy:

```pip install -U pip setuptools wheel```

```pip install -U spacy```

```python -m spacy download ru_core_news_sm```

### Описание файлов в репозитории

*generate_dataset.ipynb* -  подробное описание генерации датасета для классификации

*models.ipynb* - сравнение моделей 

*solution.py* - файл с классом, где есть возможность обработать датасет, обучить модель и получить решение

