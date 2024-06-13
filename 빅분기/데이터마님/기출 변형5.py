# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p1_1.csv
# # 종량제 봉투 가격 데이터 : https://www.data.go.kr/data/15025538/standard.do
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p1_1_.csv')
#
# # 20L가격과 5L가격이 모두 0원이 아닌 데이터만 필터를 한 후,
# # 각 row별로 20L가격과 5L가격의 차이를 ‘차이가격’ 이라 부른다고 하자.
# # 시도명 별 차이가격의 평균가격을 비교할때 그 값이 가장 큰 금액을 반올림하여
# # 소숫점 이하 1자리까지 구하여라
#
# print(df.info())
#
# df1 = df[(df['20L가격'] != 0) & (df['5L가격'] != 0)]
#
# df1['차이가격'] = df1['20L가격'] - df1['5L가격']
#
# print(round(df1.groupby('시도명').mean()['차이가격'].sort_values(ascending = False).values[0],1))










# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p1_2.csv
# # 성인 체중 및 키 데이터 : 자체 제작
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p1_2_.csv')
#
#
# # BMI는 몸무게(kg) / (키(M) * 키(M)) 로 정의 된다.
# # 초고도 비만은 BMI 25이상 , 고도 비반은 BMI 25미만 - 23이상 ,
# # 정상은 23미만 - 18.5이상 저체중은 18.5미만으로 정의 된다.
# # 주어진 데이터에서 초고도비만 인원 + 저체중 인원 의 숫자는?
#
# print(df.info())
#
# df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100)**2)
#
# print(df['BMI'])
#
# def bmi(x):
#     if x >= 25:
#         return '초고도비만'
#     elif x < 25 and x >= 23:
#         return '고도비만'
#     elif x < 23 and x >= 18.5:
#         return '정상'
#     elif x < 18.5:
#         return '저체중'
#
# df['등급'] = df['BMI'].apply(bmi)
#
# print(df['등급'].unique())
# print(df['등급'].value_counts())
#
# print(len(df[df['등급'] == '초고도비만']) + len(df[df['등급'] == '저체중']))









# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p1_2.csv
# # 년도별 서울 각 구의 초,중,고 전출 전입 인원 : https://data.seoul.go.kr/dataList/10729/S/2/datasetView.do
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p1_3.csv')
#
# # 순유입인원은 초중고 도내,도외 전입인원에서 초중고 도내, 도외 전출인원을 뺀값이다.
# # 각년도별로 가장 큰 순유입인원을 가진 지역구의 순유입인원을 구하고
# # 전체 기간의 해당 순유입인원들의 합을 구하여라
#
# print(df.info())
# df['순유입인원'] = (df['초등학교_전입_도내'] + df['초등학교_전입_도외'] + df['중학교_전입_도내']+ df['중학교_전입_도외']+ df['고등학교_전입_도내']+ df['고등학교_전입_도외']) - (df['초등학교_전출_도내']+ df['초등학교_전출_도외']+ df['중학교_전출_도내']+ df['중학교_전출_도외']+ df['고등학교_전출_도내']+ df['고등학교_전출_도외'])
#
# print(df[['년도','순유입인원']].sort_values(['년도','순유입인원']).groupby('년도').tail(1)['순유입인원'].sum())









# # 벤츠 차량 가격 예측 : https://www.kaggle.com/datasets/mysarahmadbhat/mercedes-used-car-listing
# # train = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p2_train_.csv
# # test = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p2_test_.csv
# # 예측 변수 price
# # test.csv에 대해 ID별로 price 값을 예측하여 제출
# # 제출 데이터 컬럼은 ID와 price 두개만 존재해야함
# # 평가지표는 rmse
#
# import pandas as pd
# train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p2_train_.csv')
# test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p2_test_.csv')
#
#
# print(train.info())
# print()
# print(train.isnull().sum())
# print()
# print(test.isnull().sum())
# print()
# print(train.nunique())
# print()
# print(test.nunique())
#
# x = train.drop(columns=['ID','price'])
# test_x = test.drop(columns=['ID'])
# x_dummies = pd.get_dummies(x)
# x_test_dummies = pd.get_dummies(test_x)
# x_test_dummies = x_test_dummies.reindex(columns = x_dummies.columns, fill_value=0)
# y = train['price']
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
#
# X_train, X_valid, Y_train, Y_valid = train_test_split(x_dummies,y,random_state = 42)
# rf = RandomForestRegressor(random_state=42)
# rf.fit(X_train, Y_train)
# predict_valid = rf.predict(X_valid)
#
# from sklearn.metrics import mean_squared_error
# import numpy as np
# print(np.sqrt(mean_squared_error(Y_valid,predict_valid)))
#
# predict_test = rf.predict(x_test_dummies)
#
# print(pd.DataFrame({'ID':test['ID'],'price':predict_test}))








# # 어느 학교에서 추출한 55명 학생들의 키 정보이다.
# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p3_1.csv
# # 이 학생들의 키의 95% 신뢰구간을 구하고자 한다.
#
# import pandas as pd
# df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p3_1.csv')
#
# # 55명 학생들의 키에 대한 표본 평균을 구하여라(반올림하여 소숫점 3째자리까지
#
# print(df.info())
#
# import numpy as np
#
# mean = np.mean(df['height'])
# print(mean)
#
#
# # t분포 양쪽 꼬리에서의 t 값을 구하여라 (반올림하여 소수4째자리까지)
#
# # t값 : t.ppf((1+신뢰수준)/2,ddof)  ddof는 n-1
# # ppf : 확률을 넣어 값계산
# # cdf : 값을 넣어 확률계산
# from scipy.stats import t
#
# t = round(t.ppf((1+0.95)/2,len(df)-1),4)
# print(t)
#
# # 95% 신뢰구간을 구하여라(print(lower,upper) 방식으로 출력, 각각의 값은 소숫점 이하 3째자리까지)
#
# # 신뢰구간 : 표본평균 +- t *(표준편차/루트(표본 수))
#
# std = np.std(df['height'],ddof=1)
# # ddof = 0이 디폴트 n으로 나눈다
# # ddof = 1 n-1으로 나눈다
#
# lower = round(mean - t*(std/np.sqrt(len(df))),3)
# uper = round(mean + t*(std/np.sqrt(len(df))),3)
# print(lower,uper)











# # A,B,C 세 공장에서 생산한 동일한 제품의 길이 데이터 이다.
# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p3_2.csv
# # 공장간의 제품 길이 차이가 유의미한지 확인 하려한다.
#
# import pandas as pd
# df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e5_p3_2.csv')
#
# # 3 그룹의 데이터에 대해 크루스칼-왈리스 검정을 사용하여
# # 검정 통계량을 반올림하여 소숫점 이하 3자리까지 구하여라¶
#
# print(df.info())
# print(df)
#
# from scipy.stats import kruskal
#
# A = df[df['ID'] == 'A']['value']
# B = df[df['ID'] == 'B']['value']
# C = df[df['ID'] == 'C']['value']
#
# print(kruskal(A,B,C))
#
# print(round(kruskal(A,B,C).statistic,3))
#
# # 3 그룹의 데이터에 대해 크루스칼-왈리스 검정을 사용하여
# # p-value를 반올림하여 소숫점 이하 3자리까지 구하여라.
# # 귀무가설과 대립가설중 0.05 유의수준에서 유의한 가설을 출력하라¶
#
# p = round(kruskal(A,B,C).pvalue,3)
# print(p)
#
# if p >= 0.05:
#     print('귀무')
# else:
#     print('대립')