# # DataUrl = 'https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p1.csv'
# # 데이터 출처 : 자체 제작
# # 데이터 설명 : 학생 15명의 국어,수학,영어,과학 시험 점수이다. 각 학생은 4과목 중 3과목을 선택해서 시험봤다.
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p1.csv')
#
# # 국어,수학,영어,과학 과목 중 가장 많은 학생들이 응시한 시험을 선택하고
# # 해당과목의 점수를 표준화 했을 때 가장 큰 표준화 점수를 구하여라
#
# print(df.info())
# 시험 = df.iloc[:,1:].count().sort_values(ascending=False).index[0]
# print(시험)
#
# # 표준화 : (x-mean) / std
# # zsore 또는 StandardScaler 사용
#
# from scipy.stats import zscore
# from sklearn.preprocessing import StandardScaler
#
# # 결측치 존재시 에러 발생하기에 제거후 사용
# print(zscore(df[시험].dropna()).sort_values().values[-1])
#
# # StandardScaler 사용시 차원 변환 후 사용
#
# import numpy as np
# SS = StandardScaler()
# print(SS.fit_transform(np.array(df[시험].dropna()).reshape(-1,1)).max())










# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p2.csv
# # 데이터 출처 : 자체제작 32개 변수의 수치값
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p2_.csv')
#
# # 32개의 변수간 상관관계를 확인 했을 때, var_11 컬럼과 상관계수의 절댓값이
# # 가장 큰 변수를 찾아 해당 변수의 평균값을 구하여라
#
# print(df.info())
# print(abs(df.corr()['var_11']).sort_values().index[-2])
#
# 변수 = abs(df.corr()['var_11']).sort_values().index[-2]
#
# print(df[변수].mean())










# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p3.csv
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p3.csv')
#
# # *var_6 컬럼의 1,3사분위수 각각 IQR의 1.5배 벗어난 이상치의 숫자를 구하라 *
#
# print(df.info())
# IQR = df['var_6'].quantile(0.75) - df['var_6'].quantile(0.25)
#
# min = df['var_6'].quantile(0.25) - (1.5 * IQR)
# max = df['var_6'].quantile(0.75) + (1.5 * IQR)
#
# print(IQR,min,max)
#
# print(df[(df['var_6'] < min) | (df['var_6'] > max)].shape[0])











# # 제주업종별카드이용정보 : https://www.jejudatahub.net/data/view/data/1048
# # train = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p2_train2.csv
# # test = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p2_test2.csv
# # 종속변수 :이용금액 , 평가지표 : rmse
#
# import pandas as pd
# train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p2_train2.csv')
# test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p2_test2.csv')
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
# x = train.drop(columns=['ID','이용금액'])
# test_x = test.drop(columns=['ID'])
#
# x_dummies = pd.get_dummies(x)
# x_test_dummies = pd.get_dummies(test_x)
# x_test_dummies = x_test_dummies[x_dummies.columns]
#
# y = train['이용금액']
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
#
# X_train, X_valid, Y_train, Y_valid = train_test_split(x_dummies,y,random_state=42)
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
# print(pd.DataFrame({'ID':test['ID'],'이용금액':predict_test}))
#
# ## pd.DataFrame({'ID':test['ID'],'이용금액':predict_test}).to_csv('result.csv',index = False)
# ## result  = pd.read_csv('result.csv')
# ## print(result)










# # 데이터url = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p3_1.csv
# # 종속변수 : Target
#
# import pandas as pd
# df=pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p3_1.csv')
#
# # 선형관계 가장 큰 변수 찾아 상관계수를 구하여라
#
# print(df.info())
# print(df.corr()['Target'].sort_values().values[-2])
#
#
# # Target 변수를 종속변수로 하여 다중선형회귀모델링을 진행했을 때 v2 컬럼의 회귀 계수는?
#
# # 다중 선형 회귀 : sm.OLS 사용
#
# import statsmodels.api as sm
#
# X = sm.add_constant(df.drop(columns='Target'))
# y = df['Target']
# s = sm.OLS(y,X)
# ss = s.fit()
# print(ss.params['v2'])
#
#
# # 회귀 계수들이 가지는 p값들 중 최대 값은?
#
# print(ss.pvalues.sort_values().values[-1])








# # 심장병 발병 예측 / 종속변수 target 1: 발병
# # 데이터 출처 : uci머신러닝
# # 데이터 url : https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p3_t.csv
# # train 데이터는 앞의 210개 행을, test데이터는 나머지 부분을 사용한다
#
# import pandas as pd
# df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e7_p3_t.csv')
#
# # train 데이터로 target을 종속변수로 로지스틱 회귀를 진행할 때 age 컬럼의 오즈비를 구하여라
#
# print(df.info())
#
# train = df.iloc[:210,].reset_index(drop = True)
# test = df.iloc[210:,].reset_index(drop = True)
#
# import statsmodels.api as sm
#
# X = sm.add_constant(train.drop(columns = ['target']))
# y = train['target']
#
# s = sm.Logit(y, X)
# ss = s.fit()
# print(ss.summary())
# import numpy as np
# print(np.exp(ss.params['age']))
#
#
# # train으로 로지스틱 회귀 진행했을 경우 잔차 이탈도 (residual deviance)를 계산하라
#
# m = sm.GLM(y, X ,family = sm.families.Binomial())
# mm = m.fit()
# print(mm.summary())
# print(mm.deviance)
#
#
# # train으로 로지스틱 회귀 진행했을 경우 로짓 우도값을 도출하라
# # Log-Likelihood:                -67.250
#
#
# # test 데이터의 독립변수로 target 예측 후 오류율을 구하여라
#
# X = sm.add_constant(test.drop(columns = ['target']))
# print((ss.predict(X) > 0.5) * 1)
#
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(test['target'],(ss.predict(X) > 0.5) * 1))
#
# 오류율 = 1 - accuracy_score(test['target'],(ss.predict(X) > 0.5) * 1)
#
# print(오류율)