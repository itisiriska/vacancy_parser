import numpy as np
import pandas as pd
import spacy
import re

from tqdm import tqdm
from nltk.corpus import stopwords
from string import punctuation
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


tqdm.pandas()
nlp = spacy.load('ru_core_news_sm')
stop_words = stopwords.words('russian')

MODEL_PATH = 'model.pkl'
EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+", flags=re.UNICODE
)
RANDOM_STATE = 42


class VacancySplitter:
    def __init__(self) -> None:
        super().__init__()
        self.vect = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        self.model = LogisticRegression(random_state=RANDOM_STATE, max_iter=500, n_jobs=-1)

    def _intersection(self, b1, b_list):
        for b2 in b_list:
            if b1[-1] == b2[0] or b1[0] == b2[-1]:
                return True
        return False

    def _map_ngrams(self, bigram, req_bigrams, terms_bigrams):
        if req_bigrams and bigram in req_bigrams:
            return 0
        elif terms_bigrams and bigram in terms_bigrams:
            return 1
        elif req_bigrams and self._intersection(bigram, req_bigrams):
            return 3
        elif terms_bigrams and self._intersection(bigram, terms_bigrams):
            return 3
        else:
            return 2

    def _generate_dataset(self, df_slice):
        req_bigrams = list(ngrams(df_slice.loc['requirements_cleaned'], 4)) if not np.all(
            pd.isna(df_slice.loc['requirements_cleaned'])) else []
        terms_bigrams = list(ngrams(df_slice.loc['terms_cleaned'], 4)) if not np.all(
            pd.isna(df_slice.loc['terms_cleaned'])) else []

        res_df = pd.DataFrame()
        res_df['responsibilities_bigrams'] = list(ngrams(df_slice['responsibilities_cleaned'], 4))
        res_df['id'] = df_slice['id']
        res_df['class'] = res_df['responsibilities_bigrams'].apply(lambda x: self._map_ngrams(x, req_bigrams, terms_bigrams))

        return res_df[['id', 'responsibilities_bigrams', 'class']]

    def preprocess_excel(self, path):
        df_clf = pd.read_excel(path)
        df_clf['responsibilities(Должностные обязанности)'] = df_clf['responsibilities(Должностные обязанности)'].apply(
            lambda x: str.lower(str(x)) if not pd.isna(x) else x
        ).apply(
            lambda x: re.sub(EMOJI_PATTERN, '', x)
        )
        df_clf['requirements(Требования к соискателю)'] = df_clf['requirements(Требования к соискателю)'].apply(
            lambda x: str.lower(str(x)) if not pd.isna(x) else x
        )
        df_clf['terms(Условия)'] = df_clf['terms(Условия)'].apply(lambda x: str.lower(str(x)) if not pd.isna(x) else x)
        df_clf['responsibilities_spacy'] = df_clf['responsibilities(Должностные обязанности)'].progress_apply(
            lambda x: nlp(x) if not pd.isna(x) else x)
        df_clf['requirements_spacy'] = df_clf['requirements(Требования к соискателю)'].progress_apply(
            lambda x: nlp(x) if not pd.isna(x) else x)
        df_clf['terms_spacy'] = df_clf['terms(Условия)'].progress_apply(lambda x: nlp(x) if not pd.isna(x) else x)

        for col in ['responsibilities', 'requirements', 'terms']:
            df_clf[f'{col}_cleaned'] = df_clf[f'{col}_spacy'].progress_apply(
                lambda doc: [
                    w.text.strip() for w in doc
                    if w.text not in list(punctuation) + ['—', '•', '·', '–', '”', '“', '×'] and
                       w.lemma_.strip()
                ]
                if not pd.isna(doc) else doc
            )
        final = pd.DataFrame()
        for idx in tqdm(df_clf.index):
            final = pd.concat([final, self._generate_dataset(df_clf.loc[idx])], ignore_index=True)

        final.index = range(len(final))
        final = final[final['class'] != 3]
        return final

    def train(self, X_train, y_train):
        X_train = self.vect.fit_transform(doc[:-1] for doc in X_train)
        self.model.fit(X_train, y_train)
        return

    def predict(self, data):
        X_test = self.vect.transform(data)
        pred = self.model.predict(X_test)
        return pred
