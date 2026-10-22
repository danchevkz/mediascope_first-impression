from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


METHODS = {
    'ocr_svm': 'SVM w/ OCR-TFIDF',
    'image_svm': 'SVM w/ image embeddings',
    'image_ridge': 'Ridge w/ image embeddings',
    'fusion_svm': 'SVM w/ OCR & image embeddings',
}
ENCODERS = ('clip', 'dinov2', 'siglip')
SEED = 83


class FeatureClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, method='ocr_svm', C=None, alpha=3.0, pca_components=24,
                 svd_components=48, max_features=None, ngram_range=(1, 2), seed=SEED):
        self.method = method
        self.C = C
        self.alpha = alpha
        self.pca_components = pca_components
        self.svd_components = svd_components
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.seed = seed

    def _text_vectorizer(self, fusion=False):
        return TfidfVectorizer(lowercase=True, ngram_range=self.ngram_range, min_df=2,
                               max_features=self.max_features or (10000 if fusion else 15000),
                               sublinear_tf=True, strip_accents='unicode' if fusion else None)

    def _features(self, rows, fit):
        if self.method == 'ocr_svm':
            if fit:
                self.text_ = self._text_vectorizer()
            text = [row['text'] for row in rows]
            return self.text_.fit_transform(text) if fit else self.text_.transform(text)
        if fit:
            self.reducers_ = {}
        names = ('siglip',) if self.method == 'image_svm' else ENCODERS
        blocks = []
        for name in names:
            matrix = np.asarray([row[name] for row in rows], dtype=np.float32)
            if fit:
                if min(matrix.shape) <= self.pca_components:
                    raise ValueError('PCA needs more training rows and dimensions than components')
                self.reducers_[name] = make_pipeline(
                    StandardScaler(), PCA(n_components=self.pca_components, whiten=True, random_state=self.seed))
            reducer = self.reducers_[name]
            blocks.append(reducer.fit_transform(matrix) if fit else reducer.transform(matrix))
        if self.method == 'fusion_svm':
            if fit:
                self.text_ = make_pipeline(self._text_vectorizer(fusion=True),
                                           TruncatedSVD(n_components=self.svd_components, random_state=self.seed),
                                           StandardScaler())
            texts = [row['text'] for row in rows]
            block = self.text_.fit_transform(texts) if fit else self.text_.transform(texts)
            if block.shape[1] != self.svd_components:
                raise ValueError('OCR training vocabulary is too small for the requested SVD')
            blocks.append(block)
            numeric = np.asarray([row['numeric'] for row in rows], dtype=np.float32)
            if numeric.shape != (len(rows), 4):
                raise ValueError('Expected four numeric OCR features')
            blocks.append(numeric)
        return np.hstack(blocks)

    def fit(self, X, y):
        if self.method not in METHODS:
            raise ValueError(self.method)
        if len(X) != len(y):
            raise ValueError('Feature and target counts differ')
        matrix = self._features(X, fit=True)
        if self.method == 'image_ridge':
            estimator = RidgeClassifier(alpha=self.alpha, class_weight='balanced')
        else:
            C = self.C if self.C is not None else (0.7 if self.method == 'ocr_svm' else 0.2)
            estimator = LinearSVC(C=C, class_weight='balanced', random_state=self.seed)
        self.model_ = estimator if self.method == 'ocr_svm' else make_pipeline(StandardScaler(), estimator)
        self.model_.fit(matrix, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(self._features(X, fit=False))


def parameter_grid(method):
    regularization = {'C': [0.1, 0.2, 0.7, 1.0, 3.0]}
    if method == 'ocr_svm':
        return {**regularization, 'ngram_range': [(1, 1), (1, 2)], 'max_features': [10000, 15000, 30000]}
    if method == 'image_svm':
        return {**regularization, 'pca_components': [16, 24, 32, 64]}
    if method == 'image_ridge':
        return {'alpha': [0.1, 1.0, 3.0, 10.0, 30.0], 'pca_components': [8, 16, 24, 32, 64]}
    if method == 'fusion_svm':
        return {**regularization, 'pca_components': [16, 24, 32], 'svd_components': [32, 48, 96]}
    raise ValueError(method)
