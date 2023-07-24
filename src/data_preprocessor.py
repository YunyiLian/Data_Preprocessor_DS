import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from datetime import datetime as dt
import re

class standardizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.null_patterns = re.compile(r'^N[./]*A[./]*[NT]?[./]*$|^none[.]?$|^null[.]?$', re.IGNORECASE)
        self.empty_patterns = re.compile(r'^\s*$', re.IGNORECASE)
        self.true_patterns = re.compile(r'^True$', re.IGNORECASE)
        self.false_patterns = re.compile(r'^False$', re.IGNORECASE)

    def fit(self, X, y = None):
        columns = []
        for col in X.columns:
            try:
                X[col].apply(lambda x: np.nan if bool(self.null_patterns.search(str(x))) else x)
                X[col].apply(lambda x: np.nan if bool(self.empty_patterns.search(str(x))) else x)
                X[col].apply(lambda x: True if bool(self.true_patterns.search(str(x))) else x)
                X[col].apply(lambda x: False if bool(self.false_patterns.search(str(x))) else x)
                columns.append(col)
            except:
                pass
        self.columns = columns
        return self

    def transform(self, X, y=None):
        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: np.nan if bool(self.null_patterns.search(str(x))) else x)
            X_t[col] = X_t[col].apply(lambda x: np.nan if bool(self.empty_patterns.search(str(x))) else x)
            X_t[col] = X_t[col].apply(lambda x: True if bool(self.true_patterns.search(str(x))) else x)
            X_t[col] = X_t[col].apply(lambda x: False if bool(self.false_patterns.search(str(x))) else x)
        return X_t

class numerical_transformer(BaseEstimator, TransformerMixin):
    def _convert_to_float(self, x):
        bool_patterns = re.compile(r'^True$|^False$', re.IGNORECASE)
        date_patterns = re.compile(r'^\d{4}[-]?\d{2}[-]?\d{2}$')
        if bool_patterns.search(str(x)) or date_patterns.search(str(x)):
            raise ValueError("The value cannot be converted to float.")
        else:
            return float(x)

    def fit(self, X, y = None):
        columns = []
        for col in X.columns:
            try:
                X[col].apply(lambda x: '' if pd.isnull(x) or x in ['', ' '] else self._convert_to_float(x))
                columns.append(col)
            except ValueError:
                pass
        self.columns = columns
        return self

    def transform(self, X, y = None):
        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: '' if pd.isnull(x) or x in ['', ' '] else self._convert_to_float(x))
        return X_t

class date_transformer(BaseEstimator, TransformerMixin):
    def _convert_to_date(self, x):
        try:
            return dt.strptime(str(x), '%Y-%m-%d').date().strftime('%Y-%m-%d')
        except:
            try:
                return dt.strptime(str(x), '%Y%m%d').date().strftime('%Y-%m-%d')
            except ValueError:
                pass
        raise ValueError(f"Cannot convert {x} to a date with the given formats.")

    def fit(self, X, y = None):
        columns = []
        for col in X.columns:
            try:
                X[col].apply(lambda x: '' if pd.isnull(x) or x in ['', ' '] else self._convert_to_date(x))
                columns.append(col)
            except ValueError:
                pass
        self.columns = columns
        return self

    def transform(self, X, y = None):
        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: '' if pd.isnull(x) or x in ['', ' '] else self._convert_to_date(x))
        return X_t

class string_transformer(BaseEstimator, TransformerMixin):
    def _convert_to_string(self, x):
        bool_patterns = re.compile(r'^True$|^False$', re.IGNORECASE)
        date_patterns = re.compile(r'^\d{4}[-]?\d{2}[-]?\d{2}$')
        if bool_patterns.search(str(x)) or date_patterns.search(str(x)):
            raise ValueError("The value cannot be converted to string.")
        else:
            return str(x)

    def fit(self, X, y = None):
        columns = []
        for col in X.columns:
            try:
                X[col].apply(lambda x: '' if pd.isnull(x) or x in ['', ' '] else float(x))
            except:
                try:
                    X[col].apply(lambda x: ' ' if pd.isnull(x) or x in ['', ' '] else self._convert_to_string(x))
                    columns.append(col)
                except ValueError:
                    pass
        self.columns = columns
        return self

    def transform(self, X, y = None):
        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: ' ' if pd.isnull(x) or x in ['', ' '] else self._convert_to_string(x))
        return X_t

class boolean_transformer(BaseEstimator, TransformerMixin):
    def _convert_to_boolean(self, x):
        bool_patterns = re.compile(r'^True$|^False$', re.IGNORECASE)
        if not bool_patterns.search(str(x)):
            raise ValueError("The value cannot be 'True' or 'False'.")
        else:
            return bool(x)

    def fit(self, X, y = None):
        columns = []
        for col in X.columns:
            try:
                X[col].apply(lambda x: '' if pd.isnull(x) or x in ['', ' '] else self._convert_to_boolean(x))
                columns.append(col)
            except ValueError:
                pass
        self.columns = columns
        return self

    def transform(self, X, y = None):
        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: '' if pd.isnull(x) or x in ['', ' '] else self._convert_to_boolean(x))
        return X_t
