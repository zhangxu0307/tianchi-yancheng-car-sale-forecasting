import pandas as pd

datapath = '../data/'

train_df = pd.read_csv(datapath + '[new] yancheng_train_20171226.csv')

test_df = pd.read_csv(datapath + 'yancheng_testA_20171225.csv')

train_sum10 = train_df[(train_df.sale_date == 201710)].groupby(['class_id']).sale_quantity.sum().round()

predicted = train_sum10.reset_index()

result = pd.merge(test_df[['predict_date', 'class_id']], predicted, how='left', on=['class_id'])

result.fillna(0)

result.columns = ['predict_date', 'class_id', 'predict_quantity']

result.to_csv('../result/result_201710.csv', index=False, header=True)