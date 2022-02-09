import tensorflow as tf
import pandas as pd
import numpy as np
#pip install numpy
from sklearn import utils
#데이터 셔플용

import datetime

#모델 생성
def CreateModel(Inputdata) :
    Input = tf.keras.layers.Input(shape=Inputdata.shape[1], name="Input")
    #Flatten = tf.keras.layers.Flatten(input_shape=int(trainLearningData.shape[0]), name="Flatten")
    Dense1 = tf.keras.layers.Dense(64,  activation='relu', name="Dense1")(Input)
    Dense2 = tf.keras.layers.Dense(256, activation='relu', name="Dense2")(Dense1)
    Dense3 = tf.keras.layers.Dense(64,  activation='relu', name="Dense3")(Dense2)
    Dense4 = tf.keras.layers.Dense(16,  activation='relu', name="Dense4")(Dense3)
    Output = tf.keras.layers.Dense(1,   activation='softmax', name="Output")(Dense4)

    model = tf.keras.Model(inputs=Input, outputs=Output)

    return model

#모델 학습
def TrainModel(model, trainLearningData, trainResultData, testLearningData, testResultData) :
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics='accuracy')

    model.fit(trainLearningData, trainResultData, 
              epochs=3, validation_data=(testLearningData, testResultData))


#데이터 전처리
def PreTrainData(table, resultColName, trainRate) :
    #데이터 형변환
    for rows in table :
         for colFieldName in rows :
             if type(rows[colFieldName]) == datetime.datetime :
                rows[colFieldName] = rows[colFieldName].strftime('%Y%m%d%H%M')
             elif rows[colFieldName] == 'Up':
                 rows[colFieldName] = 1
             elif rows[colFieldName] == 'Down' : 
                 rows[colFieldName] = 2
    
    data = pd.DataFrame(table)
    data = data.convert_dtypes()
    
    #학습데이터, 테스트 데이터 구분
    data = utils.shuffle(data)

    trainSize = int(data.shape[0] * trainRate)

    trainData = data.iloc[:trainSize, :]
    trainLearningData = trainData.loc[:, trainData.columns != resultColName]
    trainResultData = trainData.loc[:, resultColName]

    testData = data.iloc[trainSize:, :]
    testLearningData = testData.loc[:, testData.columns != resultColName]
    testResultData = testData.loc[:, resultColName]

    trainLearningData = np.asarray(trainLearningData).astype(np.float32)
    trainResultData = np.asarray(trainResultData).astype(np.float32)
    testLearningData = np.asarray(testLearningData).astype(np.float32)
    testResultData = np.asarray(testResultData).astype(np.float32)
    
    return trainLearningData, trainResultData, testLearningData, testResultData