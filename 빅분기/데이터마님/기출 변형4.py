# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p1_1.csv
# # 월마트 판매량 데이터 : https://www.kaggle.com/datasets/asahu40/walmart-data-analysis-and-forcasting
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p1_1.csv')
#
# # Temperature컬럼에서 숫자가 아닌 문자들을 제거후 숫자 타입으로 바꾸고
# # 3분위수에서 1분위수의 차이를 소숫점 이하 2자리까지 구하여라
#
# print(df.info())
#
# # df['Temperature'] = df['Temperature'].astype('float')
# # ValueError: could not convert string to float: '*77.22'
# # *와 같은 정규표현식 문자가 있음을 확인
#
# # df['Temperature'] = df['Temperature'].str.replace('*','').astype('float')
# # error발생 : error문구 중 regex=True.를 사용하라고 함.
#
# # *와 같은 정규표현식 변경 시 regex=True를 해야함.
#
# df['Temperature'] = df['Temperature'].str.replace('*','',regex=True).astype('float')
#
# print(df.info())
#
# print(round(df['Temperature'].quantile(0.75) - df['Temperature'].quantile(0.25),2))










# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p1_2.csv
# # 유튜브 영상 통계량 : https://www.kaggle.com/datasets/advaypatil/youtube-statistics?select=videos-stats.csv
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p1_2.csv')
#
# # Likes를 Comments로 나눈 비율이 20이상이면서
# # Keyword값이 minecraft인 영상들의 Views값의 평균을 정수로 구하여라
#
# print(df.info())
#
# print(int(df[((df['Likes'] / df['Comments']) >= 20) & (df['Keyword'] == 'minecraft')]['Views'].mean()))











# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p1_3.csv
# # 넷플릭스 영상 메타정보 : https://www.kaggle.com/datasets/akashguna/netflix-prize-shows-information
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p1_3.csv')
#
# # date_added가 2018년 1월 이면서 country가 United Kingdom 단독 제작인 데이터의 갯수
#
# print(df.info())
#
# df['date_added'] = pd.to_datetime(df['date_added'])
#
# print(df[(df['date_added'].dt.strftime('%Y-%m') == '2018-01') & (df['country'] == 'United Kingdom')].shape[0])










# # 유저 분류 : https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation
# # train = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p2_train.csv
# # test = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p2_test.csv
# # 예측 변수 Segmentation
# # test.csv에 대해 ID별로 Segmentation의 클래스를 예측해서 저장후 제출
# # 제출 데이터 컬럼은 ID와 Segmentation 두개만 존재해야함.
# # 평가지표는 macro f1 score
#
#
# import pandas as pd
# train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p2_train.csv')
# test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p2_test.csv')
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
#
# train['Ever_Married'] = train['Ever_Married'].fillna(train['Ever_Married'].value_counts().index[0])
# test['Ever_Married'] = test['Ever_Married'].fillna(test['Ever_Married'].value_counts().index[0])
#
# train['Graduated'] = train['Graduated'].fillna(train['Graduated'].value_counts().index[0])
# test['Graduated'] = test['Graduated'].fillna(test['Graduated'].value_counts().index[0])
#
# train['Profession'] = train['Profession'].fillna(train['Profession'].value_counts().index[0])
# test['Profession'] = test['Profession'].fillna(test['Profession'].value_counts().index[0])
#
# train['Work_Experience'] = train['Work_Experience'].fillna(train['Work_Experience'].mean())
# test['Work_Experience'] = test['Work_Experience'].fillna(test['Work_Experience'].mean())
#
# train['Family_Size'] = train['Family_Size'].fillna(train['Family_Size'].value_counts().index[0])
# test['Family_Size'] = test['Family_Size'].fillna(test['Family_Size'].value_counts().index[0])
#
# train['Var_1'] = train['Var_1'].fillna(train['Var_1'].value_counts().index[0])
# test['Var_1'] = test['Var_1'].fillna(test['Var_1'].value_counts().index[0])
#
# print(train.isnull().sum())
# print()
# print(test.isnull().sum())
# print()
#
# x = train.drop(columns = ['ID','Segmentation'])
# test_x = test.drop(columns = ['ID'])
#
# x_dummies = pd.get_dummies(x)
# x_test_dummies = pd.get_dummies(test_x)
# x_test_dummies = x_test_dummies[x_dummies.columns]
#
# y = train['Segmentation']
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
#
# X_train, X_valid, Y_train, Y_valid = train_test_split(x_dummies, y ,stratify = y,random_state=42)
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, Y_train)
# predict_valid_label = rf.predict(X_valid)
#
# from sklearn.metrics import f1_score
#
# print(f1_score(Y_valid,predict_valid_label,average = 'macro'))
#
# predict_test_label = rf.predict(x_test_dummies)
#
# print(pd.DataFrame({'ID':test['ID'],'Segmentation':predict_test_label}))









# # 어느 대학교의 신입생의 학과별 성별에 대한 데이터이다.
# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p3_1_.csv
# # 이 데이터를 바탕으로, 학생들의 학과와 성별이 서로 독립적인지 여부를 확인하기 위해 카이제곱 독립성 검정을 실시 하려한다.
#
# import pandas as pd
# df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e4_p3_1_.csv')
#
# #학과 평균 인원에 대한 값을 소숫점 이하 3자리까지 구하여라
#
# print(df.info())
#
# print(round(df.groupby('학과').size().mean(),3))
#
# #카이제곱검정 독립성 검정 통계량을 소숫점 이하 3자리까지 구하여라
#
# from scipy.stats import chi2_contingency
#
# print(pd.crosstab(df['성별'],df['학과']))
#
# data = pd.crosstab(df['성별'],df['학과'])
#
# chi2, p, dof, exp = chi2_contingency(data)
#
# print(round(chi2,3))
#
#
# # 카이제곱검정 독립성 검정의 pvalue를 소숫점 이하 3자리까지 구하여라.
# # 유의수준 0.05하에서 귀무가설과 대립가설중 유의한 것을 출력하라
#
# p =round(p,3)
# print(p)
#
# if p >= 0.05:
#     print('귀무')
# elif p < 0.05:
#     print('대립')









# # 어느 학교에서 수학 시험을 본 학생 100명 중 60명이 60점 이상을 받았다.
# # 이 학교의 수학 시험의 평균 점수가 50점 이상인지 95%의 신뢰 수준에서 검정하려한다.
#
#
# # 검정 통계량을 소숫점 이하 3자리에서 구하시오
#
# # 모비율에 대한 가설검정에서 검정통계량 : (표본비율 - 검정비율) / (루트((검정비율 * (1-검정비율)) / 표본수))
#
# 표본비율 = 0.6
# 검정비율 = 0.5
# 표본수 = 100
#
# import numpy as np
# 검정통계량 = (표본비율 - 검정비율) / np.sqrt((검정비율 * (1-검정비율)) / 표본수)
#
# print(round(검정통계량,3))
#
#
# # pvalue를 소숫점 이하 3자리까지 구하고 귀무가설과 대립가설중 유의한 것을 출력하라
#
# # pvalue는 1-검정통계량에서 확률로 구한다
#
# from scipy.stats import norm #정규분포
#
# p_value = 1-norm.cdf(검정통계량)
# print(round(p_value,3))
#
# if p_value >= 0.5:
#     print('귀무')
# elif p_value < 0.5:
#     print('대립')