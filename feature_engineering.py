from feature_engineer.eda import *
from feature_engineer.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

import pandas as pd
df = pd.read_csv("feature_engineer/titanic.csv")
df.columns = [col.upper() for col in df.columns]
df.shape
df.head()

def titanic_data_prep(df):
    #feature engineering
    df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int') #cabin varmı yok mu ?
    df["NEW_NAME_COUNT"] = df["NAME"].str.len() #name count
    df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" "))) #name word count
    df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")])) #doktor ünvanı taşıyor mu
    df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False) #name title
    df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1 #family size
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"] #age pclass
    df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO" #yalnız
    df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES" #yalnız değil
    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young' #genc
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature' #yetiskin
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior' #yaslı
    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale' #genc erkek
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale' #yetiskin erkek
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale' #yaslı erkek
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale' #genc kadın
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale' #yetiskin kadın
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale' #yaslı kadın
    df.loc[(df['SURVIVED'] == 0) & (df['SEX'] == 'male'), 'SURVIVED_MALE'] = 0
    df.loc[(df['SURVIVED'] == 1) & (df['SEX'] == 'male'), 'SURVIVED_MALE'] = 1
    df.loc[(df['SURVIVED'] == 0) & (df['SEX'] == 'female'), 'SURVIVED_FEMALE'] = 0
    df.loc[(df['SURVIVED'] == 1) & (df['SEX'] == 'female'), 'SURVIVED_FEMALE'] = 0

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    #outlier (aykırı değer)
    for col in num_cols:
        replace_with_thresholds(df, col)

    #missing values (eksik değer)
    df.drop(["CABIN","TICKET", "NAME"], inplace=True, axis=1) #cabin,ticket ve name sütununu kaldırdık
    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    #label encoding
    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]

    for col in binary_cols:
        df = label_encoder(df, col)

    #rare encoding
    df = rare_encoder(df, 0.01)

    #one-hot encoding

    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

    df = one_hot_encoder(df, ohe_cols)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]
    useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                    (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
    df.drop(useless_cols, axis=1, inplace=True)
    df

    #standart scaler

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

prep = titanic_data_prep(df)

df.head()
df.shape

import pickle
prep.to_pickle("./prep.pkl")