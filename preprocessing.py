# preprocessing.py
import pandas as pd

def clean_and_transform_data(dataset):
    #Checking if there is any null data
    dataset.isnull().sum()

    # Renaming columns
    dataset = dataset.rename(columns={'default.payment.next.month': 'def_pay', 'PAY_0': 'PAY_1'})

    # Analysing the correlation of features with  default_pay
    X = dataset.drop(['def_pay'],axis=1)
    X.corrwith(dataset['def_pay']).plot.bar(figsize = (10, 5), title = "Correlation with Default", fontsize = 10,rot = 90, grid = True)

    # Handling education and marriage categories
    fil_education = (dataset['EDUCATION'] == 5) | (dataset['EDUCATION'] == 6) | (dataset['EDUCATION'] == 0)
    dataset.loc[fil_education, 'EDUCATION'] = 4

    dataset.loc[dataset['MARRIAGE'] == 0, 'MARRIAGE'] = 3

    # Handling PAY columns
    for i in range(1, 7):
        fil_pay = (dataset[f'PAY_{i}'] == -1) | (dataset[f'PAY_{i}'] == -2)
        dataset.loc[fil_pay, f'PAY_{i}'] = 0

    # One-Hot Encoding
    for att in ['SEX', 'EDUCATION', 'MARRIAGE']:
        dataset[att] = dataset[att].astype('category')

    dataset = pd.concat([
        pd.get_dummies(dataset['SEX'], prefix='SEX'),
        pd.get_dummies(dataset['EDUCATION'], prefix='EDUCATION'),
        pd.get_dummies(dataset['MARRIAGE'], prefix='MARRIAGE'),
        dataset
    ], axis=1)

    dataset.drop(['EDUCATION', 'SEX', 'MARRIAGE'], axis=1, inplace=True)

    return dataset
