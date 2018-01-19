import pandas as pd

datapath = '../data/'

train_df = pd.read_csv(datapath + '[new] yancheng_train_20171226.csv')

test_df = pd.read_csv(datapath + 'yancheng_testA_20171225.csv')

train_df["year"] = train_df['sale_date'].apply(lambda x: int(str(x)[:4]))
train_df["month"] = train_df['sale_date'].apply(lambda x: int(str(x)[4:]))

# 提取动态特征，包括时间戳和销量，聚合方式
sumData =train_df.groupby(by=["class_id", "month", "year"]).sum()
sumData.reset_index(inplace=True)
dynamicData = sumData[["class_id", "month", "year", 'sale_quantity']]

train_mean = dynamicData[(dynamicData.month == 10)].groupby(['class_id']).sale_quantity.mean().round()

predicted = train_mean.reset_index()

result = pd.merge(test_df[['predict_date', 'class_id']], predicted, how='left', on=['class_id'])

result.fillna(0)

result.columns = ['predict_date', 'class_id', 'predict_quantity']

result.to_csv('../result/result_mean_11.csv', index=False, header=True)