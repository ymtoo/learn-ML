"""Load data."""
import os
import numpy as np
import pandas as pd

def _preprocess_numeric(df, listnumeric):
    """Preprocess data to avoid numeric data columns with string 'NA' """
    listcolumns = df.keys().tolist()
    for i, col in enumerate(listcolumns):
        if col in listnumeric:
            df.loc[:, col].replace('NA', -1, inplace=True)
            df.loc[:, col] = df.loc[:, col].apply(pd.to_numeric)
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
               '3SsnPorch', 'ScreenPorch', 'PoolArea',
               'MiscVal', 'YrSold']
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
                   'MoSold': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                   'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD',
                                'Con', 'ConLw', 'ConLI', 'ConLD',
                                'Oth'],
                   'SaleCondition': ['Normal', 'Abnorml', 'AdjLand',
                                     'Alloca', 'Family', 'Partial']
                  }
    labelkey = 'SalePrice'
    
    datadf = pd.read_csv(os.path.join(dirfolder, 'train.csv'), keep_default_na=False)
    testdf = pd.read_csv(os.path.join(dirfolder, 'test.csv'), keep_default_na=False)    
    datadf.set_index('Id', inplace=True)
    testdf.set_index('Id', inplace=True)

    ydata = np.array(datadf[labelkey]) 
    datadf.drop(columns=[labelkey], inplace=True)
    traindf = datadf[:-numvalid]
    validdf = datadf[-numvalid:]
    trainy = ydata[:-numvalid]
    validy = ydata[-numvalid:]
    numtrainsamp, numtrainfeat = traindf.shape
    numvalidsamp, numvalidfeat = validdf.shape
    numtestsamp, numtestfeat = testdf.shape
    print("Number of train samples is {}.".format(numtrainsamp))
    print("Number of valid samples is {}.".format(numvalidsamp))
    print("Number of test samples is {}.".format(numtestsamp))
    
    numfeatcols = len(list(dictcategorical.keys()))+len(listnumeric)
    print("Number of feature columns is {}.".format(numfeatcols))

    listcolumns = traindf.keys().tolist()
    listcategorical = list(dictcategorical.keys())
    print('Checking listnumeric ...')
    print([num in listcolumns for num in listnumeric])
    print("Checking dictcategorical ...")
    print([cat in listcolumns for cat in listcategorical])

    traindf = _preprocess_numeric(traindf, listnumeric)
    validdf = _preprocess_numeric(validdf, listnumeric)
    testdf = _preprocess_numeric(testdf, listnumeric)
    #traindf, validdf, testdf = normalize_numeric(traindf, validdf, testdf, listnumeric)
    return (traindf, trainy), (validdf, validy), testdf, listnumeric, dictcategorical