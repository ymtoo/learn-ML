"""Preprocess input data."""
import numpy as _np
from sklearn import preprocessing as _process

def categorical_encode(listvocab, listcat):
    """Encode categorical features as an integer array."""
    le = _process.LabelEncoder()
    le.fit(listvocab)
    intarray = []
    for cat in listcat:
        if cat in listvocab:
            intarray.append(le.transform([cat]))
        else:
            intarray.append(-1)
    return _np.squeeze(_np.array(intarray))

def cat2int(df, dictcategorical):
    """Convert the categorcal data to numeric."""
    listcolumns = df.keys().tolist()
    listcategorical = list(dictcategorical.keys())
    data = _np.zeros((df.shape))
    for i, col in enumerate(listcolumns):
        if col in listcategorical:
            listvocab = dictcategorical[col]   
            data[:, i] = categorical_encode(listvocab, df[col].tolist())
        else:
            data[:, i] = _np.array(df[col])
    return data
        