"""Load data."""
import os as _os
import numpy as _np
import pandas as _pd

from sklearn.preprocessing import LabelEncoder

REPLACE_NAN_NUMERIC = -1
REPLACE_NAN_CATEGORICAL = 'None'

def _replace_nan(df, listnumeric, listcategorical):
    """Preprocess data to avoid numeric data columns with string 'NA' """
    listcolumns = df.keys().tolist()
    for i, col in enumerate(listcolumns):
        if col in listnumeric:
            df.loc[:, col].replace(_np.nan, REPLACE_NAN_NUMERIC, inplace=True)
            df.loc[:, col] = df.loc[:, col].apply(_pd.to_numeric)
        if col in listcategorical:
            df.loc[:, col].replace(_np.nan, REPLACE_NAN_CATEGORICAL, inplace=True)
    return df

def _normalize_numeric(traindf, validdf, testdf, listnumeric):
    """Normalize numeric data"""
    listcolumns = traindf.keys().tolist()
    for col in listcolumns:
        if col in listnumeric:
            miu = traindf[col].mean()
            sigma = traindf[col].std()
            traindf[col] = (traindf[col]-miu)/sigma
            validdf[col] = (validdf[col]-miu)/sigma
            testdf[col] = (testdf[col]-miu)/sigma
    return traindf, validdf, testdf
    
def ames_housing(dirfolder, numvalid=100):
    listnumeric = ['LotFrontage', 'LotArea', 'YearBuilt', 
               'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
               'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
               '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
               'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
               'FullBath', 'HalfBath', 'BedroomAbvGr', 
               'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
               'GarageYrBlt', 'GarageCars', 'GarageArea',
               'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
               '3SsnPorch', 'ScreenPorch', 'PoolArea']
    dictcategorical = {'MSSubClass': [20, 30, 40, 45, 50, 
                                  60, 70, 75, 80, 85, 
                                  90, 120, 150, 160, 
                                  180, 190], 
                   'MSZoning': ['A', 'C', 'FV', 'I', 
                                'RH', 'RL', 'RP', 'RM'], 
                   'Street': ['Grvl', 'Pave'],
                   'Alley': ['Grvl', 'Pave', 'NA'],
                   'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
                   'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
                   'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
                   'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 
                                 'FR3'],
                   'LandSlope': ['Gtl', 'Mod', 'Sev'],
                   'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale',
                                    'BrkSide', 'ClearCr', 'CollgCr',
                                    'Crawfor', 'Edwards', 'Gilbert',
                                    'IDOTRR', 'MeadowV', 'Mitchel',
                                    'Names', 'NoRidge', 'NPkVill',
                                    'NridgHt', 'NWAmes', 'OldTown',
                                    'SWISU', 'Sawyer', 'SawyerW'
                                    'Somerst', 'StoneBr', 'Timber', 
                                    'Veenker'],
                   'Condition1': ['Artery', 'Feedr', 'Norm', 'RRNn',
                                  'RRAn', 'PosN', 'PosA', 'RRNe', 
                                  'RRAe'],
                   'Condition2': ['Artery', 'Feedr', 'Norm', 'RRNn',
                                  'RRAn', 'PosN', 'PosA', 'RRNe', 
                                  'RRAe'],
                   'BldgType': ['1Fam', '2FmCon', 'Duplx', 
                                'TwnhsE', 'TwnhsI'],
                   'HouseStyle': ['1Story', '1.5Fin', '1.5Unf',
                                  '2Story',  '2.5Fin','2.5Unf', 
                                  'SFoyer', 'SLvl'],
                   'OverallQual': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                   'OverallCond': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                   'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip',
                                 'Mansard', 'Shed'],
                   'RoofMatl': ['ClyTile', 'CompShg', 'Membran',
                                'Metal', 'Roll', 'Tar&Grv', 
                                'WdShake', 'WdShngl'],
                   'Exterior1st': ['AsbShng', 'AsphShn', 'BrkComm',
                                   'CBlock', 'CemntBd', 'HdBoard',
                                   'ImStucc', 'MetalSd', 'Other',
                                   'Plywood', 'PreCast', 'Stone', 
                                   'Stucco', 'VinylSd', 'Wd Sdng',
                                   'WdShing'],
                   'Exterior2nd': ['AsbShng', 'AsphShn', 'BrkComm',
                                   'CBlock', 'CemntBd', 'HdBoard',
                                   'ImStucc', 'MetalSd', 'Other',
                                   'Plywood', 'PreCast', 'Stone', 
                                   'Stucco', 'VinylSd', 'Wd Sdng',
                                   'WdShing'],
                   'MasVnrType': ['BrkCmn', 'BrkFace', 'CBlock',
                                  'None', 'Stone'],
                   'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                   'ExterCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                   'Foundation': ['BrkTil', 'CBlock', 'PConc',
                                  'Slab', 'Stone', 'Wood'],
                   'BsmtQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
                   'BsmtCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
                   'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'NA'],
                   'BsmtFinType1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 
                                    'LwQ', 'Unf', 'NA'],
                   'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 
                                    'LwQ', 'Unf', 'NA'],
                   'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 
                               'OthW', 'Wall'],
                   'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                   'CentralAir': ['N', 'Y'],
                   'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP',
                                  'Mix'],
                   'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                   'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 
                                  'Maj1', 'Maj2', 'Sev', 'Sal'],
                   'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
                   'GarageType': ['2Types', 'Attchd', 'Basment',
                                  'BuiltIn', 'CarPort', 'Detchd',
                                  'NA'],
                   'GarageFinish': ['Fin', 'RFn', 'Unf', 'NA'],
                   'GarageQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
                   'GarageCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
                   'PavedDrive': ['Y', 'P', 'N'],
                   'PoolQC': ['Ex', 'Gd', 'TA', 'Fa', 'NA'],
                   'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'],
                   'MiscFeature': ['Elev', 'Gar2', 'Othr', 'Shed',
                                   'TenC', 'NA'],
                   'YrSold': [2000, 2001, 2002, 2003, 2004, 2005, 2006,
                              2007, 2008, 2009, 2010, 2011, 2012, 2013,
                              2014, 2015, 2016, 2017],
                   'MoSold': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                   'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD',
                                'Con', 'ConLw', 'ConLI', 'ConLD',
                                'Oth'],
                   'SaleCondition': ['Normal', 'Abnorml', 'AdjLand',
                                     'Alloca', 'Family', 'Partial']
                  }
    labelkey = 'SalePrice'
    
    datadf = _pd.read_csv(_os.path.join(dirfolder, 'train.csv'))
    testdf = _pd.read_csv(_os.path.join(dirfolder, 'test.csv')) 
    test_ID = testdf['Id'].values
    datadf.set_index('Id', inplace=True)
    testdf.set_index('Id', inplace=True)

    ydata = _np.array(datadf[labelkey]) 
    datadf.drop(columns=[labelkey], inplace=True)
    num_data = datadf.shape[0]
    
    # compile all data
    alldatadf = _pd.concat([datadf, testdf], ignore_index=True, sort=False)#.reset_index(drop=True)

    # remove outlier
#    alldatadf = alldatadf.drop(alldatadf[(datadf['GrLivArea']>4000) & (alldatadf['SalePrice']<300000)].index)
    drop_columns = ['Utilities']
    alldatadf = alldatadf.drop(columns=drop_columns)
#    testdf = testdf.drop(columns=drop_columns)

    # fillna
    alldatadf.loc[:, 'Functional'].fillna('Typ', inplace=True)
#    testdf.loc[:, 'Functional'].fillna('Typ', inplace=True)
    cols_fillna_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
                        'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'BsmtQual', 'BsmtCond',
                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', "MasVnrType", 'MSSubClass']
    cols_fillna_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
                        'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', "MasVnrArea"]
    list_cols_name = alldatadf.columns.values[1:]
    for cols_name in list_cols_name:
        if cols_name in cols_fillna_none:
            alldatadf.loc[:, cols_name].fillna('None', inplace=True)
#            testdf.loc[:, cols_name].fillna('None', inplace=True)
        elif cols_name in cols_fillna_zero:
            alldatadf.loc[:, cols_name].fillna(0, inplace=True)
#            testdf.loc[:, cols_name].fillna(0, inplace=True)
        else:
            if cols_name not in ['SalePrice']:
                alldatadf.loc[:, cols_name].fillna(datadf.loc[:, cols_name].mode()[0], inplace=True)
#                testdf.loc[:, cols_name].fillna(testdf.loc[:, cols_name].mode()[0], inplace=True)
                
    # Adding total sqfootage feature 
    alldatadf['TotalSF'] = alldatadf['TotalBsmtSF'] + alldatadf['1stFlrSF'] + alldatadf['2ndFlrSF']
#    testdf['TotalSF'] = testdf['TotalBsmtSF'] + testdf['1stFlrSF'] + testdf['2ndFlrSF']

    # categorical data
    cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold']
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(alldatadf[c].values)) 
        alldatadf[c] = lbl.transform(list(alldatadf[c].values))
    
    alldatadf = _pd.get_dummies(alldatadf)
    datadf_new = alldatadf[:num_data]
    testdf_new = alldatadf[num_data:]
    testdf_new['Id'] = test_ID
    testdf_new.set_index('Id', inplace=True)

    numdata = datadf_new.shape[0]
    traindf = datadf_new[:-numvalid]
    validdf = datadf_new[-numvalid:]
    trainy = ydata[:-numvalid].reshape(numdata-numvalid, 1)
    validy = ydata[-numvalid:].reshape(numvalid, 1)
    trainy = _np.log1p(trainy)
    validy = _np.log1p(validy)
    
#    numtrainsamp, numtrainfeat = traindf.shape
#    numvalidsamp, numvalidfeat = validdf.shape
#    numtestsamp, numtestfeat = testdf.shape
#    print("Number of train samples is {}.".format(numtrainsamp))
#    print("Number of valid samples is {}.".format(numvalidsamp))
#    print("Number of test samples is {}.".format(numtestsamp))
    
#    numfeatcols = len(list(dictcategorical.keys()))+len(listnumeric)
#    print("Number of feature columns is {}.".format(numtrainfeat))

#    listcolumns = traindf.keys().tolist()
#    listcategorical = list(dictcategorical.keys())
#    print('Checking listnumeric ...')
#    print([num in listcolumns for num in listnumeric])
#    print("Checking dictcategorical ...")
#    print([cat in listcolumns for cat in listcategorical])

#    traindf = _replace_nan(traindf, listnumeric, listcategorical)
#    validdf = _replace_nan(validdf, listnumeric, listcategorical)
#    testdf = _replace_nan(testdf, listnumeric, listcategorical)
    #traindf, validdf, testdf = normalize_numeric(traindf, validdf, testdf, listnumeric)
    return (traindf, trainy), (validdf, validy), testdf_new