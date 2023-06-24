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
        self.path = ''

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
        res_df['class'] = res_df['responsibilities_bigrams'].apply(
            lambda x: self._map_ngrams(x, req_bigrams, terms_bigrams))

        return res_df[['id', 'responsibilities_bigrams', 'class']]

    def preprocess_excel(self, path):
        self.path = path
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
        X_test = self.vect.transform(data['responsibilities_bigrams'])
        data['predict'] = self.model.predict(X_test)
        data.index = data['id']

        v_ids = np.unique(data['id'])
        valid_df = pd.DataFrame(index=v_ids, columns=['id', 'responsibilities'])
        valid_df['id'] = valid_df.index
        valid_df['reqiurements'] = valid_df['id'].apply(lambda x: self._get_phrases(data, x, 0))
        valid_df['terms'] = valid_df['id'].apply(lambda x: self._get_phrases(data, x, 1))

        return valid_df

    def _get_phrases(self, data, id_, cls):
        vacancy = data.loc[id_]
        bigrams = vacancy[vacancy['predict'] == cls]['responsibilities_bigrams'].to_list()

        phrases = []
        for i in range(1, len(bigrams)):
            phrase = ' '.join(bigrams[i - 1])
            curr_ngram = bigrams[i - 1]
            next_ngram = bigrams[i]
            while curr_ngram[3] == next_ngram[2] and i < len(bigrams):
                curr_ngram = bigrams[i - 1]
                next_ngram = bigrams[i]
                phrase += f' {next_ngram[3]}'
                i += 1
            phrases.append(phrase)

        return '\n'.join(self._validate(phrases))

    def _validate(self, phrases):
        validated_phrases = []

        if len(phrases) == 2 and phrases[0].find(phrases[1]) != -1:
            validated_phrases.append(phrases[0])
            return validated_phrases
        elif len(phrases) < 2 or len(phrases) == 2 and phrases[0].find(phrases[1]) == -1:
            return phrases

        for i in reversed(range(len(phrases) - 1)):
            if phrases[i].find(phrases[i + 1]) == -1:
                validated_phrases.append(phrases[i + 1])
            if i == 0 and not validated_phrases:
                validated_phrases.append(phrases[i])

        return validated_phrases
