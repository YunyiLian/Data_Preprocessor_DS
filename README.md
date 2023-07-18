# Data_Preprocessor_DS
A data preprocessor module including standardizer and transformers to increase data quality, and make it ready for analysis and developing

## Structure
1. **standardizer**
   - Standardize `null` values and white space(s) `str` to `np.nan`; e.g. 'Null'/'NA'/'N.A.'/'NAN'/'NAT'/''/' '/...--->`np.nan`
   - Standardize potential `bool` values to be `True` or `False`; e.g. 'True'/'true'/'TRUE'--->`True`
   - Convert all values to `str` except `np.nan`

3. **numerical_transformer**
   - Convert the potential `float` values in a column to `float`. e.g. '2'-->2.0; '2.0'--->2.0
   - Convert the `np.nan` values within that column to an empty `str`. e.g. `np.nan`--->''

4. **date_transformer**
   - Convert the potential `date` values in a column to `date`. e.g. '2023-07-17'--->2023-07-17; '20230717'--->2023-07-17
   - Convert the `np.nan` values within that column to single space `str`. e.g. `np.nan`--->' '

6. **string_transformer**
   - Convert the potential `str` values in a column to `str`.
   - Convert the `np.nan` values within that column to single space `str`. e.g. `np.nan`--->' '

7. **boolean_transformer**
   - Convert the potential `bool` values in a column to `bool`. e.g. 'True'--->`True`; 'False'--->`False`
   - Convert the `np.nan` values within that column to an empty `str`. e.g. `np.nan`--->''

## Pipeline
The following pipeline will implement all standardizer and transformers at once.

However, the pipeline is **order-sensitive** so avoid changing the order of the classes within it. 
```python
from data_preprocessor import standardizer, numerical_transformer, date_transformer,
                              string_transformer, boolean_transformer

pipeline = Pipeline(steps=[
    ('standardize', standardizer()),
    ('numerical', numerical_transformer()),
    ('date', date_transformer()),
    ('string', string_transformer()),
    ('boolean', boolean_transformer())
])

df_t = pipeline.fit_transform(df)
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
|  6  | 20           | True       | cde          | 20230629    |
|  7  | 2.5          | False      | 1234         | 20230630    |
|  8  | 3.8          | False      | 234          | 20230630    |
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
