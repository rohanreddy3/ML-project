# src/features.py
import re, string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

class LSA_Tag_Generator:
    def __init__(self, n_components=100, max_features=10000, ngram_range=(1,2), min_df=2):
        self.n_components = n_components
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None
        self.pipeline = None
        self.feature_names = None

    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', ' ', str(text))
        text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
        return text.lower().strip()

    def fit(self, documents):
        clean_docs = [self.clean_text(d) for d in documents]
        self.vectorizer = TfidfVectorizer(max_features=self.max_features,
                                          ngram_range=self.ngram_range,
                                          min_df=self.min_df,
                                          stop_words='english')
        X = self.vectorizer.fit_transform(clean_docs)
        self.feature_names = self.vectorizer.get_feature_names_out()
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        normalizer = Normalizer(copy=False)
        self.pipeline = make_pipeline(svd, normalizer)
        self.pipeline.fit(X)
        print("✅ LSA model trained successfully!")

    def generate_tags_for_doc(self, text, top_k=5):
        if self.vectorizer is None or self.pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")
        clean_text = self.clean_text(text)
        vec = self.vectorizer.transform([clean_text])
        topic_vector = self.pipeline.transform(vec)
        # compute term-topic matrix once by transforming each feature token
        term_matrix = self.pipeline.transform(self.vectorizer.transform(self.feature_names))
        sim = np.dot(term_matrix, topic_vector.T).ravel()
        top_indices = np.argsort(sim)[::-1][:top_k]
        return [self.feature_names[i] for i in top_indices]
