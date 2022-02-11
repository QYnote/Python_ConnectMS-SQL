import clsDatabase as db
import ClsModel as mo
import pandas as pd
#pip install pandas 필요
import numpy as np
#pip install numpy

#테스트용
import random

#delete 필요
from sklearn import utils
#pip install scikit-learn, scipy 필요



#Database에서 자료 가져오기
query = 'SELECT candle_date_time_utc, candle_date_time_kst, opening_price, high_price, low_price, '
query +=      ' trade_price, timestamp, candle_acc_trade_price, candle_acc_trade_volume, unit,'
query +=      ' gap, [check] '
query += ' FROM TradingTest'

#table type : List
table = db.SelectQuery(query)

db.ConnectionClose()



#데이터 전처리
trainRate = 0.8 #학습용 데이터 80%, 테스트용 20%
resultColName = 'check'

trainLearningData, trainResultData, testLearningData, testResultData = mo.PreTrainData(table, 'check', 0.8)
model = mo.CreateModel(trainLearningData)
model = mo.TrainModel(model, trainLearningData, trainResultData, testLearningData, testResultData)

model.summary()



#입력데이터 전처리
randRowIdx = random.randrange(0, testLearningData.shape[0])

#InputData 전처리
testInputAry = testLearningData[randRowIdx:randRowIdx+1, :]
testInputData = np.asarray(testInputAry).astype(np.float32)


#데이터 예측
result = model.predict(testInputData)
maxcount = 0
while int(result) == 1 :
    result = model.predict(testInputData)
    maxcount = maxcount +1
    if maxcount > 1000: break
print(result)