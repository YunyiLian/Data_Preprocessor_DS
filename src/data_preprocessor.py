import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from datetime import datetime as dt
import re

class standardizer(BaseEstimator, TransformerMixin):
    """
    Class to standardize all null values and white space(s) strings to np.nan;
    Standardize all potential boolean values to the correct boolean format and type.
    
    Attributes
    ----------
    null_patterns : compiled regular expressions for null values
    empty_patterns : compiled regular expressions for empty values
    true_patterns : compiled regular expressions for true values
    false_patterns : compiled regular expressions for false values
    columns : list of column names to be transformed
    
    Examples:
    >>> s = standardizer()
    >>> df = pd.DataFrame({
    ...          'Numerical':['123',' ','','NA','N.A.','None','20',2.5,3.8,np.nan],
    ...          'Boolean':['True',' ','','NA','N.A.','None',True,False,False,np.nan],
    ...          'Character':['abc',' ','','NA','N.A.','None','cde',1234,'234',12],
    ...          'Date':['2023-06-28',' ','','NA','N.A.','None',20230629,20230630,'20230630',np.nan]})
    >>> df_t = s.fit_transform(df)
    >>> df_t
      Numerical Boolean Character        Date
    0       123    True       abc  2023-06-28
    1       NaN     NaN       NaN         NaN
    2       NaN     NaN       NaN         NaN
    3       NaN     NaN       NaN         NaN
    4       NaN     NaN       NaN         NaN
    5       NaN     NaN       NaN         NaN
    6        20    True       cde    20230629
    7       2.5   False      1234    20230630
    8       3.8   False       234    20230630
    9       NaN     NaN        12         NaN
    
    """
    def __init__(self):
        self.null_patterns = re.compile(r'^N[./]*A[./]*[NT]?[./]*$|^none[.]?$|^null[.]?$', re.IGNORECASE)
        self.empty_patterns = re.compile(r'^\s*$', re.IGNORECASE)
        self.true_patterns = re.compile(r'^True$', re.IGNORECASE)
        self.false_patterns = re.compile(r'^False$', re.IGNORECASE)

    def fit(self, X, y = None):
        """
        Fit the transformer on `X`.
        Find the columns to be standardized in `X`,
        generate list containing columns names to be standardized.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """

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
        """
        Transform `X`.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_t : DataFrame
            Transformed version of `X`.
        """

        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: np.nan if bool(self.null_patterns.search(str(x))) else x)
            X_t[col] = X_t[col].apply(lambda x: np.nan if bool(self.empty_patterns.search(str(x))) else x)
            X_t[col] = X_t[col].apply(lambda x: True if bool(self.true_patterns.search(str(x))) else x)
            X_t[col] = X_t[col].apply(lambda x: False if bool(self.false_patterns.search(str(x))) else x)
        return X_t

class numerical_transformer(BaseEstimator, TransformerMixin):
    """
    Class to find the columns are potentially in type `float`;
    Transform all not-null values within the column to `float`,
    transform all null values within the column to empty string ''.
    
    Attributes
    ----------
    columns : list of column names to be transformed
    
    Examples:
    >>> n = numerical_transformer()
    >>> df = pd.DataFrame({
    ...          'Numerical':['123',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,2.5,3.8,np.nan],
    ...          'Boolean':[True,np.nan,np.nan,np.nan,np.nan,np.nan,True,False,False,np.nan],
    ...          'Character':['abc',np.nan,np.nan,np.nan,np.nan,np.nan,'cde',1234,'234',12],
    ...          'Date':['2023-06-28',np.nan,np.nan,np.nan,np.nan,np.nan,20230629,20230630,'20230630',np.nan]})
    >>> df_t = n.fit_transform(df)
    >>> df_t
      Numerical Boolean Character        Date
    0     123.0    True       abc  2023-06-28
    1               NaN       NaN         NaN
    2               NaN       NaN         NaN
    3               NaN       NaN         NaN
    4               NaN       NaN         NaN
    5               NaN       NaN         NaN
    6              True       cde    20230629
    7       2.5   False      1234    20230630
    8       3.8   False       234    20230630
    9               NaN        12         NaN
    
    """

    def _convert_to_float(self, x):
        """
        Convert `x` to float if `x` has no poyential to be a boolean or date;
        Else raise `ValueError`.

        Parameters
        ----------
        x : value in a column of a dataframe.
        
        Returns
        -------
        float(x) : float
            float type of `x`.    
        """

        bool_patterns = re.compile(r'^True$|^False$', re.IGNORECASE)
        date_patterns = re.compile(r'^\d{4}[-]?\d{2}[-]?\d{2}$')
        if bool_patterns.search(str(x)) or date_patterns.search(str(x)):
            raise ValueError("The value cannot be converted to float.")
        else:
            return float(x)

    def fit(self, X, y = None):
        """
        Fit the transformer on `X`.
        Find the columns to be transformed to type float in `X`,
        generate list containing the columns names.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """

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
        """
        Transform `X`.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_t : DataFrame
            Transformed version of `X`.
        """

        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: '' if pd.isnull(x) or x in ['', ' '] else self._convert_to_float(x))
        return X_t
    
class date_transformer(BaseEstimator, TransformerMixin):
    """
    Class to find the columns are potentially in type `date`;
    Transform all not-null values within the column to sting in `YYYY-MM-DD` format,
    transform all null values within the column to single space string ' '.
    
    Attributes
    ----------
    columns : list of column names to be transformed
    
    Examples:
    >>> d = date_transformer()
    >>> df = pd.DataFrame({
    ...          'Numerical':['123',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,2.5,3.8,np.nan],
    ...          'Boolean':[True,np.nan,np.nan,np.nan,np.nan,np.nan,True,False,False,np.nan],
    ...          'Character':['abc',np.nan,np.nan,np.nan,np.nan,np.nan,'cde',1234,'234',12],
    ...          'Date':['2023-06-28',np.nan,np.nan,np.nan,np.nan,np.nan,20230629,20230630,'20230630',np.nan]})
    >>> df_t = d.fit_transform(df)
    >>> df_t  
      Numerical Boolean Character        Date
    0       123    True       abc  2023-06-28
    1       NaN     NaN       NaN            
    2       NaN     NaN       NaN            
    3       NaN     NaN       NaN            
    4       NaN     NaN       NaN            
    5       NaN     NaN       NaN            
    6       NaN    True       cde  2023-06-29
    7       2.5   False      1234  2023-06-30
    8       3.8   False       234  2023-06-30
    9       NaN     NaN        12            

    """

    def _convert_to_date(self, x):
        """
        Convert `x` to date string `YYYY-MM-DD` if `x` has potential to be a date;
        Else raise `ValueError`.

        Parameters
        ----------
        x : value in a column of a dataframe.
        
        Returns
        -------
        str type of `x` in `YYYY-MM-DD` format.
        """

        try:
            return dt.strptime(str(x), '%Y-%m-%d').date().strftime('%Y-%m-%d')
        except:
            try:
                return dt.strptime(str(x), '%Y%m%d').date().strftime('%Y-%m-%d')
            except ValueError:
                pass
        raise ValueError(f"Cannot convert {x} to a date with the given formats.")

    def fit(self, X, y = None):
        """
        Fit the transformer on `X`.
        Find the columns to be transformed to date string in `X`,
        generate list containing the columns names.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """

        columns = []
        for col in X.columns:
            try:
                X[col].apply(lambda x: ' ' if pd.isnull(x) or x in ['', ' '] else self._convert_to_date(x))
                columns.append(col)
            except ValueError:
                pass
        self.columns = columns
        return self

    def transform(self, X, y = None):
        """
        Transform `X`.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_t : DataFrame
            Transformed version of `X`.
        """

        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: ' ' if pd.isnull(x) or x in ['', ' '] else self._convert_to_date(x))
        return X_t

class string_transformer(BaseEstimator, TransformerMixin):
    """
    Class to find the columns are potentially in type string;
    Transform all not-null values within the column to string,
    transform all null values within the column to single space string ' '.
    
    Attributes
    ----------
    columns : list of column names to be transformed
    
    Examples:
    >>> s = string_transformer()
    >>> df = pd.DataFrame({
    ...          'Numerical':['123',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,2.5,3.8,np.nan],
    ...          'Boolean':[True,np.nan,np.nan,np.nan,np.nan,np.nan,True,False,False,np.nan],
    ...          'Character':['abc',np.nan,np.nan,np.nan,np.nan,np.nan,'cde',1234,'234',12],
    ...          'Date':['2023-06-28',np.nan,np.nan,np.nan,np.nan,np.nan,20230629,20230630,'20230630',np.nan]})
    >>> df_t = s.fit_transform(df)
    >>> df_t
      Numerical Boolean Character        Date
    0       123    True       abc  2023-06-28
    1       NaN     NaN                   NaN
    2       NaN     NaN                   NaN
    3       NaN     NaN                   NaN
    4       NaN     NaN                   NaN
    5       NaN     NaN                   NaN
    6       NaN    True       cde    20230629
    7       2.5   False      1234    20230630
    8       3.8   False       234    20230630
    9       NaN     NaN        12         NaN
    
    """

    def _convert_to_string(self, x):
        """
        Convert `x` to string type if `x` has no potential to be a boolean or date;
        Else raise `ValueError`.

        Parameters
        ----------
        x : value in a column of a dataframe.
        
        Returns
        -------
        str(x) : str
            string type of `x`.
        """

        bool_patterns = re.compile(r'^True$|^False$', re.IGNORECASE)
        date_patterns = re.compile(r'^\d{4}[-]?\d{2}[-]?\d{2}$')
        if bool_patterns.search(str(x)) or date_patterns.search(str(x)):
            raise ValueError("The value cannot be converted to string.")
        else:
            return str(x)

    def fit(self, X, y = None):
        """
        Fit the transformer on `X`.
        Find the columns to be transformed to type string in `X`,
        generate list containing the columns names.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """

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
        """
        Transform `X`.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_t : DataFrame
            Transformed version of `X`.
        """

        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: ' ' if pd.isnull(x) or x in ['', ' '] else self._convert_to_string(x))
        return X_t
    
class boolean_transformer(BaseEstimator, TransformerMixin):
    """
    Class to find the columns are potentially in type boolean;
    Transform all not-null values within the column to boolean,
    transform all null values within the column to empty string ''.
    
    Attributes
    ----------
    columns : list of column names to be transformed
    
    Examples:
    >>> b = boolean_transformer()
    >>> df = pd.DataFrame({
    ...          'Numerical':['123',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,2.5,3.8,np.nan],
    ...          'Boolean':[True,np.nan,np.nan,np.nan,np.nan,np.nan,True,False,False,np.nan],
    ...          'Character':['abc',np.nan,np.nan,np.nan,np.nan,np.nan,'cde',1234,'234',12],
    ...          'Date':['2023-06-28',np.nan,np.nan,np.nan,np.nan,np.nan,20230629,20230630,'20230630',np.nan]})
    >>> df_t = b.fit_transform(df)
    >>> df_t
      Numerical Boolean Character        Date
    0       123    True       abc  2023-06-28
    1       NaN               NaN         NaN
    2       NaN               NaN         NaN
    3       NaN               NaN         NaN
    4       NaN               NaN         NaN
    5       NaN               NaN         NaN
    6       NaN    True       cde    20230629
    7       2.5   False      1234    20230630
    8       3.8   False       234    20230630
    9       NaN                12         NaN
    
    """

    def _convert_to_boolean(self, x):
        """
        Convert `x` to boolean type if `x` has potential to be a boolean;
        Else raise `ValueError`.

        Parameters
        ----------
        x : value in a column of a dataframe.
        
        Returns
        -------
        bool(x) : bool
            boolean type of `x`.
        """
        bool_patterns = re.compile(r'^True$|^False$', re.IGNORECASE)
        if not bool_patterns.search(str(x)):
            raise ValueError("The value cannot be 'True' or 'False'.")
        else:
            return bool(x)

    def fit(self, X, y = None):
        """
        Fit the transformer on `X`.
        Find the columns to be transformed to type boolean in `X`,
        generate list containing the columns names.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """

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
        """
        Transform `X`.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        X_t : DataFrame
            Transformed version of `X`.
        """

        check_is_fitted(self,['columns'])
        X_t = X.copy()
        for col in self.columns:
            X_t[col] = X_t[col].apply(lambda x: '' if pd.isnull(x) or x in ['', ' '] else self._convert_to_boolean(x))
        return X_t
