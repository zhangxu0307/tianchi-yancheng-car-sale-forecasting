import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import grid_search
import xgboost as xgb
import copy

# 去除"-"
def replaceNULL(datastr):

    if datastr == "-":
        return 0
    else:
        return float(datastr)

# TR字段有";"
def operateTR(TRstr):

    if ";" in TRstr:
        return int(TRstr[0])
    else:
        return int(TRstr)

# 处理斜杠
def operateSlash(data):

    if not isinstance(data, float):
        if "-" in data:
            return data
        else:
            return float(data.split('/')[0])
    else:
        return data

# 处理载客限制，此字段比较混乱
def operateRatedPassenger(data):
    if data[-1] == "日":
        return -1
    if "-" in data:
        return int(data[0])
    else:
        return int(data)

# 解析价格区间字段
def parsePrictLevel(prictLevelStr):

    if prictLevelStr[-1] == "W":
        priceList = prictLevelStr[:-1].split("-")
        lowPrice = int(priceList[0])
        highPrice = int(priceList[1])
        lowPrice = lowPrice*10000
        highPrice = highPrice*10000
    else:
        lowPrice = 5000
        highPrice = 5000
    return lowPrice, highPrice

# 获得低价
def getLowPrice(prictLevelStr):
    return parsePrictLevel(prictLevelStr)[0]

# 获得高价
def getHighPrice(prictLevelStr):
    return parsePrictLevel(prictLevelStr)[1]

# 读取训练数据
data = pd.read_csv("../data/[new] yancheng_train_20171226.csv")
print("oringinal data features:")
print(data.columns)

# 处理时间
data["year"] = data['sale_date'].apply(lambda x: int(str(x)[:4]))
data["month"] = data['sale_date'].apply(lambda x: int(str(x)[4:]))

# 处理脏乱数据
data["TR"] = data["TR"].apply(operateTR)
data["power"] = data["power"].apply(operateSlash)
data["engine_torque"] = data["engine_torque"].apply(operateSlash)
data["rated_passenger"] = data["rated_passenger"].apply(operateRatedPassenger)

# 去除"-"，数据中代表空值
data["level_id"] = data["level_id"].apply(replaceNULL)
data["price"] = data["price"].apply(replaceNULL)
data["fuel_type_id"] = data["fuel_type_id"].apply(replaceNULL)
data["engine_torque"] = data["engine_torque"].apply(replaceNULL)

# 离散化特征
gearBoxEncoder = LabelEncoder()
gearBoxEncoder.fit(data["gearbox_type"])
data["gearbox_type"] = gearBoxEncoder.transform(data["gearbox_type"])

if_chargingEncoder = LabelEncoder()
if_chargingEncoder.fit(data["if_charging"])
data["if_charging"] = if_chargingEncoder.transform(data["if_charging"])

# 高低价提取
data["low_price"] = data["price_level"].apply(getLowPrice)
data["high_price"] = data["price_level"].apply(getHighPrice)


featureName = ['class_id', 'brand_id', 'compartment',
       'type_id', 'level_id', 'department_id', 'TR', 'gearbox_type',
       'displacement', 'if_charging', 'price', 'driven_type_id',
       'fuel_type_id', 'newenergy_type_id', 'emission_standards_id',
       'if_MPV_id', 'if_luxurious_id', 'power', 'cylinder_number',
       'engine_torque', 'car_length', 'car_width', 'car_height',
       'total_quality', 'equipment_quality', 'rated_passenger', 'wheelbase',
       'front_track', 'rear_track',
        'low_price', 'high_price',
               'month', 'year',
               ]

# 提取汽车静态特征，此类特征与时间无关
staticFeatureName = featureName[:-2]
carData = data[staticFeatureName].drop_duplicates("class_id")

# 提取动态特征，包括时间戳和销量，聚合方式
sumData = data.groupby(by=["class_id", "month", "year"]).sum()
sumData.reset_index(inplace=True)
dynamicData = sumData[["class_id", "month", "year", 'sale_quantity']]

# 训练集生成
newData = pd.merge(carData, dynamicData, on="class_id", how="inner")
trainx = newData[featureName]
trainy = newData['sale_quantity']

print("train data features:")
print(trainx.columns)
print("train data num:", len(trainx))

submit = pd.read_csv('../data/yancheng_testA_20171225.csv')
testx = copy.deepcopy(carData)
testx["month"] = 11
testx["year"] = 2017
del submit["predict_quantity"]

print("test data features:")
print(testx.columns)
print("test data num:", len(testx))


#网格搜索
# print("网格搜索")
# model = xgb.XGBRegressor(objective="reg:linear")
# params = {"learning_rate": [0.5, 0.1, 0.01], "max_depth": [10, 20, 30], "n_estimators":[50, 100, 200, 250]}
# gsearch = grid_search.GridSearchCV(estimator=model, param_grid=params, scoring="neg_mean_squared_error", cv=3)
#
# gsearch.fit(trainx, trainy)
# print("best param:", gsearch.best_params_)
# print("best score:", gsearch.best_score_)


lr = 0.5 #gsearch.best_params_['learning_rate']
maxDepth = 10 #gsearch.best_params_['max_depth']
treeNum = 50 # gsearch.best_params_['n_estimators'],
scores = []
print("验证")
k = 10
for i in range(k):
       train_x, val_x, train_y, val_y = train_test_split(trainx, trainy, test_size=1/k)
       model = xgb.XGBRegressor(objective="reg:linear",
                                 learning_rate= lr,
                                 max_depth= maxDepth,
                                 n_estimators= treeNum,
                                 silent=True,
                                 colsample_bytree=0.9,
                                 )
       model.fit(train_x, train_y)
       pre = model.predict(val_x)
       dispDF = pd.DataFrame()
       # dispDF["y"] = val_y
       # dispDF["pred"] = pre
       # print(dispDF)
       score = np.sqrt(mean_squared_error(val_y, pre))
       scores.append(score)
print("valid scores:", scores)
print("mean score:", np.mean(scores))

# 测试
print("训练预测")
model = xgb.XGBRegressor(objective="reg:linear",
                         learning_rate=lr,
                         max_depth=maxDepth,
                         n_estimators=treeNum,
                         silent=True,
                         colsample_bytree=0.9,
                         )
trainx.sort_index(axis=1, inplace=True)
testx.sort_index(axis=1,inplace=True)
model.fit(trainx, trainy)
pred = model.predict(testx)
testx["predict_quantity"] = pred
testx = testx.groupby(by=["class_id"]).max().round()
testx = testx.reset_index()
submit = pd.merge(submit, testx[["class_id", "predict_quantity"]], on="class_id", how="left")
submit.to_csv("../result/result1.csv", index=False)

print("predicting finished!")


