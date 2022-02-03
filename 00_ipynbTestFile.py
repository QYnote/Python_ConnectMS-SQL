import clsDatabase as db
import pandas as pd
#pip install pandas 필요
import datetime
from sklearn import utils
#install scikit-learn, scipy pip 필요
import tensorflow as tf

#Database에서 자료 가져오기
query = 'SELECT * FROM TradingTest'

table = db.SelectQuery(query)

db.ConnectionClose()

#데이터 형변환
for rows in table :
     for colFieldName in rows :
         if type(rows[colFieldName]) is datetime.datetime :
            rows[colFieldName] = rows[colFieldName].strftime('%Y%m%d%H%M')
            
data = pd.DataFrame(table)

trainRate = 0.8 #학습용 데이터 80%, 테스트용 20%
resultColName = 'check'
data = utils.shuffle(data)

trainSize = int(data.shape[0] * trainRate)

trainData = data.iloc[:trainSize, :]
trainLearningData = trainData.loc[:, trainData.columns != resultColName]
trainResultData = trainData.loc[:, resultColName]

testData = data.iloc[trainSize:, :]
testLearningData = testData.loc[:, testData.columns != resultColName]
testResultData = testData.loc[:, resultColName]


Input = tf.keras.layers.Input(shape=trainLearningData.shape[0], name="Input")
#Flatten = tf.keras.layers.Flatten(input_shape=int(trainLearningData.shape[0]), name="Flatten")
Dense1 = tf.keras.layers.Dense(64,  activation='relu', name="Dense1")
Dense2 = tf.keras.layers.Dense(256, activation='relu', name="Dense2")
Dense3 = tf.keras.layers.Dense(64,  activation='relu', name="Dense3")
Dense4 = tf.keras.layers.Dense(16,  activation='relu', name="Dense4")
Output = tf.keras.layers.Dense(1,   activation='sigmoid', name="Output")

model = tf.keras.models.Sequential([Input, Dense1, Dense2, Dense3, Dense4, Output])