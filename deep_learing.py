import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #최소 경고수준으로 설정

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt # 시각화
import pandas as pd


pd.set_option('display.max_rows', 1000) 
#디스플레이 옵션 함수: set_option -> 디스플레이 옵션 중 행 수를 조절 n:None을 입력하면 무제한
#원하는 숫자 넣으면 그 숫자 만큼 최대 행수를 조절 가능
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 1000)

titanic_df = pd.read_csv("./2025.10.30/titanic_passengers.csv")
#csv파일 읽어와서 Dataframe 객체 생성
print(titanic_df.head()) #.head() 데이터 프레임에서 앞 부분 5개만 출력하겠다
print(type(titanic_df)) #class 'pandas.core.frame.DataFrame

train_target=titanic_df['Survived'] #survived라는 행만 불러옴 
# train_target2=titanic_df["Survived"].map({1:'Survival', 0:'fail'}) #survival의 1를 Survival로 0을 fail로 변경하겠다 
# print(train_target[:5])
# print(train_target2[:5])

#gender 컬럼 데이터를 수치 데이터로 변환
titanic_df['gender']=titanic_df['gender'].map({'male':1, 'female':0})

print(titanic_df.sample(10))

#Dataframe 객체의 정보를 확인 ==> info

titanic_df.info()

#print(titanic_df['Age']) age컬럼에 결측지 데이터가 존재 ==> NAN
#결측지 제거가 필요

print(titanic_df['Age'].isnull()) #데이커가 없으면 True 데이터가 있으면 FLase로 반환
#false 반환 ==> 불린배열
#불린배열을 이용해서 True인 위치만 추출하는 문법 ==> 불린색인 
#boolean array==> 0, 1
#불린 색인 ==> 

print(len(titanic_df.loc[titanic_df['Age'].isnull(),['Age']])) #117개의 결측치 존재 
#loc[]를 통해 행 또는 열의 데이터를 조회 가능 
#df.loc[[시작열:끝열], [시작행:끝행]]

#결측치가 있는 행을 제거: dropna()
#how ='any': 결측치가 하나라도 있으면 해당 행 삭제
#how ='all': 해당행에 모든 데이터 각 결측치 일때 삭제 

titanic_df.dropna(subset='Age', how = 'any', inplace=True)
titanic_df.info()

print(titanic_df['Pclass']) #1등석과 2등석 정보만 중요한 역할 
#Pclass데이터를 원핫 인코딩으로 변환해서 1등석과 2등석 컬럼 데이터만 사용
#해당 컬럼 데이터의 수치 데이터를 원핫인코딩 형태로 변환 ==> pd.get_dymmies()
onehot_pclass = pd.get_dummies(titanic_df['Pclass'], prefix='Class')
print(onehot_pclass.head())

#넘파이 배열의 병합 ==> concatenate()
#pandas의 병합 ==> concat()
print('before concat:', titanic_df.head())
titanic_df = pd.concat([titanic_df, onehot_pclass], axis = 1)
print('after concat:', titanic_df.head()) #위아래 합치기 axis=0, 양옆으로 합치기 axis=1
print(titanic_df.head())

titanic_train_input=titanic_df[['gender', 'Age', 'Class_1', 'Class_2']]
titanic_train_target=titanic_df['Survived']

print(titanic_train_input.head())
print(titanic_train_target.head())

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target =\
    train_test_split(titanic_train_input, titanic_train_target, random_state=42, shuffle=True )
print(len(test_input))
print(len(test_input))
print(train_input[:30])

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

train_scaled = scalar.fit_transform(train_input)
test_scaled = scalar.transform(test_input)

print(train_scaled[:5])
print(test_scaled[:5])

model=Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, batch_size=16, epochs=800, verbose=1 )

print('test acc:', model.evaluate(test_scaled, test_target)[1])

model.save('titanicbestmodel.h5')