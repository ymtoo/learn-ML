"""Load data."""
import os as _os
import numpy as _np
import pandas as _pd
import random as _random
from sklearn.preprocessing import LabelEncoder

def ames_housing(dirfolder, numvalid=100):
    """Load ames houseing dataset"""
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

    # fillna
    alldatadf.loc[:, 'Functional'].fillna('Typ', inplace=True)
    cols_fillna_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
                        'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'BsmtQual', 'BsmtCond',
                        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', "MasVnrType", 'MSSubClass']
    cols_fillna_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
                        'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', "MasVnrArea"]
    list_cols_name = alldatadf.columns.values[1:]
    for cols_name in list_cols_name:
        if cols_name in cols_fillna_none:
            alldatadf.loc[:, cols_name].fillna('None', inplace=True)
        elif cols_name in cols_fillna_zero:
            alldatadf.loc[:, cols_name].fillna(0, inplace=True)
        else:
            if cols_name not in ['SalePrice']:
                alldatadf.loc[:, cols_name].fillna(datadf.loc[:, cols_name].mode()[0], inplace=True)
                
    # Adding total sqfootage feature 
    alldatadf['TotalSF'] = alldatadf['TotalBsmtSF'] + alldatadf['1stFlrSF'] + alldatadf['2ndFlrSF']

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
    return (traindf, trainy), (validdf, validy), testdf_new

def credit_card_fraud(path):
    """Load credit card fraud dataset"""
    _random.seed(1)
    n_train_notfraud = 89800
    n_train_fraud = 200
    n_valid_notfraud = 9977
    n_valid_fraud = 23
    
    df = _pd.read_csv(path, compression='zip', sep=',')
    labelkey = 'Class'
    yidx_notfraud = df.index[df[labelkey] == 0].tolist()
    yidx_fraud = df.index[df[labelkey] == 1].tolist()
    trainyidx_notfraud = _random.sample(yidx_notfraud, n_train_notfraud)
    trainyidx_fraud = _random.sample(yidx_fraud, n_train_fraud)
    
    trainyidx = trainyidx_notfraud + trainyidx_fraud

    traindf = df.loc[trainyidx]
    validdf = df.loc[~df.index.isin(trainyidx)]
    trainy = traindf[labelkey].values
    traindf.drop(labels=[labelkey], axis=1, inplace=True)
    validy = validdf[labelkey].values
    validdf.drop(labels=[labelkey], axis=1, inplace=True)
    return (traindf, trainy), (validdf, validy)