import re
import pandas as pd
import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import contractions
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary


class LemmaTokenizer(object):
    def __init__(self, stop_words=stopwords.words('english')):
        self.stop_words = stop_words
        self.wnl = WordNetLemmatizer()

    def __call__(self, sentence):
        token_sent = word_tokenize(sentence)
        pos_sent = pos_tag(token_sent)
        lemma_sent = [
            self.wnl.lemmatize(w, self.convert_tags(p))
            if w not in self.stop_words
            else w
            for w, p in pos_sent
        ]
        return lemma_sent

    def convert_tags(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return 'a'
        elif nltk_tag.startswith('V'):
            return 'v'
        elif nltk_tag.startswith('N'):
            return 'n'
        elif nltk_tag.startswith('R'):
            return 'r'
        else:
            # default
            return 'n'


class DocVecGenerator(object):
    def __init__(
        self, lowercase=True, analyzer='word', max_features=200, tokenizer=LemmaTokenizer(),
        expand_contractions=True, strip_punctuations=True,
        strip_accents='unicode', stop_words=stopwords.words('english'),
        num_workers=4
    ):
        self.lowercase = lowercase
        self.analyzer = analyzer
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.expand_contractions = expand_contractions
        self.strip_punctuations = strip_punctuations
        self.strip_accents = strip_accents
        self.stop_words = stop_words
        self.vectorizer = None
        self.num_workers = num_workers

    def preprocess(self, line):
        """ any custom preprocessing steps, include here """

        if self.lowercase:
            line = line.lower()

        if self.expand_contractions:
            line = contractions.fix(line)

        if self.strip_punctuations:
            line = re.sub(r'[^\w\s]', '', line)

        # remove stopwords
        line = ' '.join([word for word in line.split(' ')
                         if word not in self.stop_words])

        return line

    def __call__(self, doc, method='tfidf', fit_transform=True):
        task_func = {
            'tfidf': self.tfidf,
            'doc2vec': self.doc2vec,
            'lda': self.lda
        }
        if method not in task_func.keys():
            raise NotImplementedError
        task_func[method]()
        if fit_transform:
            X = self.vectorizer.fit_transform(doc)
            return X, self.vectorizer
        else:
            self.vectorizer.fit(doc)
            return self.vectorizer

    def tfidf(self):
        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            max_features=self.max_features,
            preprocessor=self.preprocess,
            tokenizer=self.tokenizer,
            strip_accents=self.strip_accents
        )

    def doc2vec(self):
        # strip accents not implemented here yet

        self.vectorizer = Doc2Vec(
            vector_size=self.max_features,
            window=2,
            min_count=1,
            workers=self.num_workers
        )

        def doc2vec_fit(doc):
            doc = [TaggedDocument(self.tokenizer(self.preprocess(d)), [
                                  i]) for i, d in enumerate(doc)]
            self.vectorizer.build_vocab(doc)
            self.vectorizer.train(
                doc, total_examples=self.vectorizer.corpus_count,
                epochs=self.vectorizer.epochs
            )

        def doc2vec_transform(doc):
            doc = [np.array(self.vectorizer.infer_vector(
                self.tokenizer(self.preprocess(d)))) for d in doc]
            return np.array(doc)

        def doc2vec_fit_transform(doc):
            doc2vec_fit(doc)
            return doc2vec_transform(doc)

        self.vectorizer.fit = doc2vec_fit
        self.vectorizer.transform = doc2vec_transform
        self.vectorizer.fit_transform = doc2vec_fit_transform

    def lda(self):
        """
        Latent Dirchilet Allocation does not have an initialisation step for object
        We use Doc2Vec just as a dummy intialisation, it is not used later
        Everything is run in the "fit" step.
        """
        # strip accents not implemented here yet
        self.vectorizer = Doc2Vec()

        def lda_fit(doc):
            doc = [self.tokenizer(self.preprocess(d)) for d in doc]
            dictionary = Dictionary(doc)
            bow_corpus = [dictionary.doc2bow(d) for d in doc]

            self.vectorizer = LdaMulticore(
                corpus=bow_corpus,
                num_topics=self.max_features,
                id2word=dictionary,
                passes=10,
                workers=self.num_workers
            )

            # reassign functions
            self.vectorizer.fit = lda_fit
            self.vectorizer.transform = lda_transform
            self.vectorizer.fit_transform = lda_fit_transform

        def lda_transform(doc):
            dictionary = self.vectorizer.id2word
            doc = [self.tokenizer(self.preprocess(d)) for d in doc]
            bow_corpus = [dictionary.doc2bow(d) for d in doc]
            preds = defaultdict(dict)
            for test_ix, test_vec in enumerate(bow_corpus):
                _pred = dict.fromkeys(range(self.max_features), 0)
                # update scores if available
                for topic_ix, score in self.vectorizer[test_vec]:
                    _pred[topic_ix]=score
                preds[test_ix] = _pred
            doc = pd.DataFrame(preds).transpose()
            return np.array(doc)

        def lda_fit_transform(doc):
            lda_fit(doc)
            return lda_transform(doc)

        self.vectorizer.fit = lda_fit
        self.vectorizer.transform = lda_transform
        self.vectorizer.fit_transform = lda_fit_transform
