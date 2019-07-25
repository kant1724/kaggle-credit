import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
import numpy as np

train_df = pd.read_csv('./application_train.csv')
test_df = pd.read_csv("./application_test.csv")
test2_df = pd.read_csv("./application_test.csv")
target = train_df['TARGET']
test_df.drop('SK_ID_CURR', axis=1)
train_df.drop('SK_ID_CURR', axis=1)
train_df.drop('TARGET', axis=1)

all = pd.concat([train_df, test_df], ignore_index=True)

all = all.loc[:, ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'NAME_TYPE_SUITE',
                  'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OWN_CAR_AGE', 'OCCUPATION_TYPE',
                  'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                  ]]

'''
all['NAME_CONTRACT_TYPE'] = all['NAME_CONTRACT_TYPE'].astype('category').cat.codes
all['CODE_GENDER'] = all['CODE_GENDER'].astype('category').cat.codes
all['FLAG_OWN_CAR'] = all['FLAG_OWN_CAR'].astype('category').cat.codes
all['FLAG_OWN_REALTY'] = all['FLAG_OWN_REALTY'].astype('category').cat.codes
all['NAME_TYPE_SUITE'] = all['NAME_TYPE_SUITE'].astype('category').cat.codes
all['NAME_INCOME_TYPE'] = all['NAME_INCOME_TYPE'].astype('category').cat.codes
all['NAME_EDUCATION_TYPE'] = all['NAME_EDUCATION_TYPE'].astype('category').cat.codes
all['NAME_FAMILY_STATUS'] = all['NAME_FAMILY_STATUS'].astype('category').cat.codes
all['NAME_HOUSING_TYPE'] = all['NAME_HOUSING_TYPE'].astype('category').cat.codes
all['OCCUPATION_TYPE'] = all['OCCUPATION_TYPE'].astype('category').cat.codes
all['WEEKDAY_APPR_PROCESS_START'] = all['WEEKDAY_APPR_PROCESS_START'].astype('category').cat.codes
all['ORGANIZATION_TYPE'] = all['ORGANIZATION_TYPE'].astype('category').cat.codes
all['FONDKAPREMONT_MODE'] = all['FONDKAPREMONT_MODE'].astype('category').cat.codes
all['HOUSETYPE_MODE'] = all['HOUSETYPE_MODE'].astype('category').cat.codes
all['WALLSMATERIAL_MODE'] = all['WALLSMATERIAL_MODE'].astype('category').cat.codes
all['EMERGENCYSTATE_MODE'] = all['EMERGENCYSTATE_MODE'].astype('category').cat.codes
'''

all = pd.get_dummies(all)

train_len = train_df.shape[0]
train_df = all[:train_len]
test_df = all[train_len:]

print(train_df)

x, x_test, y, y_test = train_test_split(train_df, target, test_size=0.2, random_state=42, stratify=target)
train_data = lightgbm.Dataset(x, label=y)
test_data = lightgbm.Dataset(x_test, label=y_test)

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 20,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=50000,
                       early_stopping_rounds=100)


pred = model.predict(test_df)

p = np.squeeze(pred)
output = pd.DataFrame({'SK_ID_CURR': test2_df.SK_ID_CURR, 'TARGET': p})
output.to_csv('submission.csv', index=False)
