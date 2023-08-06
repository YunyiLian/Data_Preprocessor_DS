# Data_Preprocessor_DS
A data preprocessor module including standardizer and transformers to increase data quality, and make it ready for analysis and development. 
The module pipeline can perform automatic data cleaning, column `dtype` (`float`, `datetime`, `str` & `bool`) detection and correction.
<br/>

## Python File Location for the Module 

:star_struck:Directory: /src/data_preprocessor.py  
:laughing:Direct Link: [Click_Here](/src/data_preprocessor.py) 

## Structure
1. **standardizer**
   - Standardize `null` values and white space(s) `str` to `np.nan`; e.g. 'Null'/'NA'/'N.A.'/'NAN'/'NAT'/''/' '/...--->`np.nan`
   - Standardize potential `bool` values to `True` or `False`; e.g. 'True'/'true'/'TRUE'--->`True`

3. **numerical_transformer**
   - Convert the values in the potential `float` columns to `float`. e.g. '2'-->2.0; '2.0'--->2.0
   - Convert the `np.nan` values within that column to an empty `str`. e.g. `np.nan`--->''

4. **date_transformer**
   - Convert the values in the potential `date` columns to date `str`. e.g. '2023-07-17' (`datetime`)--->'2023-07-17'; '20230717'--->'2023-07-17'
   - Convert the `np.nan` values within that column to single space `str`. e.g. `np.nan`--->' '

6. **string_transformer**
   - Convert the values in the potential `str` columns to `str`.
   - Convert the `np.nan` values within that column to single space `str`. e.g. `np.nan`--->' '

7. **boolean_transformer**
   - Convert the values in the potential `bool` columns to `bool`. e.g. 'True'--->`True`; 'False'--->`False`
   - Convert the `np.nan` values within that column to an empty `str`. e.g. `np.nan`--->''

## Pipeline
The following pipeline will implement the standardizer and all transformers at once.  
> __Warning__: 
However, the pipeline is **order-sensitive** so make sure to put `standardizer` first, and other `transformer`s can be put in any order.
```python
from data_preprocessor import standardizer, numerical_transformer, date_transformer,
                              string_transformer, boolean_transformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[
    ('standardize', standardizer()),
    ('numerical', numerical_transformer()),
    ('date', date_transformer()),
    ('string', string_transformer()),
    ('boolean', boolean_transformer())
])

df_t = pipeline.fit_transform(df)
```

## Doc test
Doc test can now be run in the `terminal`. Instruction is shown below:
```console
device_name:~ username$ python -m doctest -v .\data_preprocessor.py
...
20 tests in 21 items.
20 passed and 0 failed.
Test passed.
```

## Example
<pre>
                The original DataFrame:                                   The transformed DataFrame:
</pre>
<table>
<tr>
<td>

|     | Numerical    | Boolean    | Character    | Date        |
|:---:|:------------:|:----------:|:------------:|:-----------:|
|  0  | 123          | True       | abc          | 2023-06-28  |
|  1  |              |            |              |             |
|  2  |              |            |              |             |
|  3  | NA           | NA         | NA           | NA          |
|  4  | N.A.         | N.A.       | N.A.         | N.A.        |
|  5  | None         | None       | None         | None        |
|  6  | 20           | true       | cde          | 20230629    |
|  7  | 2.5          | FALSE      | 1234         | 20230630    |
|  8  | 3.8          | false      | 234          | 20230630    |
|  9  | nan          | nan        | 12           | nan         |

</td>
<td>

|     | Numerical    | Boolean    | Character    | Date        |
|:---:|:------------:|:----------:|:------------:|:-----------:|
|  0  | 123.0        | True       | abc          | 2023-06-28  |
|  1  |              |            |              |             |
|  2  |              |            |              |             |
|  3  |              |            |              |             |
|  4  |              |            |              |             |
|  5  |              |            |              |             |
|  6  | 20.0         | True       | cde          | 2023-06-29  |
|  7  | 2.5          | False      | 1234         | 2023-06-30  |
|  8  | 3.8          | False      | 234          | 2023-06-30  |
|  9  |              |            | 12           |             |

</td>
</tr>
</table>
