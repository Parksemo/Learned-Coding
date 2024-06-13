# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p1_1.csv
# # 캘리포니아 집값 일부 변형 : https://www.kaggle.com/datasets/harrywang/housing?select=housing.csv)
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p1_1.csv')
#
# # 결측치가 하나라도 존재하는 행의 경우 경우 해당 행을 삭제하라.
# # 그후 남은 데이터의 상위 70%에 해당하는 데이터만 남겨둔 후
# # median_income 컬럼의 1분위수를 반올림하여 소숫점이하 2째자리까지 구하여라
#
# print(len(df))
# print(df.isnull().sum())
#
# df = df.dropna().reset_index(drop = True)
# print(len(df))
# print(df.isnull().sum())
#
# df = df.iloc[:int(len(df) * 0.7),]
# print(len(df))
# print(df)
#
# print(round(df['median_income'].quantile(0.25),2))










# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p1_2.csv
# # 년도별 국가의 GDP : https://www.kaggle.com/datasets/tunguz/country-regional-and-world-gdp
#
# import pandas as pd
# df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p1_2.csv')
#
# # 1990년도는 해당년도 평균 이하 GDP를 가지지만,
# # 2010년도에는 해당년도 평균 이상 GDP를 가지는 국가의 숫자를 구하여라
#
# print(df.info())
#
# df90 = df[df['Year'] == 1990]
# Country_90 = set(df90[df90['Value'] <= df90['Value'].mean()]['Country Name'].values)
#
# df10 = df[df['Year'] == 2010]
# Country_10 = set(df10[df10['Value'] >= df10['Value'].mean()]['Country Name'].values)
#
# print(len(Country_90))
# print(len(Country_10))
# print(len(Country_90 & Country_10))









# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p1_3.csv
# # 타이타닉 탑승자 생존 데이터 : https://www.kaggle.com/competitions/titanic
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p1_3.csv')
#
# # 데이터에서 결측치가 가장 많은 컬럼을 출력하라
#
# print(df.isnull().sum().sort_values().index[-1])










# # 여행자 보험 가입여부 분류 : https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data
# # DataUrl(train) = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p2_train_.csv
# # DataUrl(test) = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p2_test_.csv
# # 종속 변수 : TravelInsurance , TravelInsurance가 1일 확률을 구해서 제출하라.
# # 평가지표 : auc
# # 제출 파일의 컬럼은 ID, proba 두개만 존재해야한다.
#
# import pandas as pd
# train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p2_train_.csv')
# test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p2_test_.csv')
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
# x = train.drop(columns=['ID','TravelInsurance'])
# test_x = test.drop(columns = ['ID'])
#
# x_dummies = pd.get_dummies(x)
# x_test_dummies = pd.get_dummies(test_x)
# x_test_dummies = x_test_dummies[x_dummies.columns]
#
# y = train['TravelInsurance']
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
#
# X_train, X_valid, Y_train, Y_valid = train_test_split(x_dummies,y,stratify=y,random_state=42)
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, Y_train)
# predict_valid_proba = rf.predict_proba(X_valid)[:,1]
#
# from sklearn.metrics import roc_auc_score
#
# print(roc_auc_score(Y_valid,predict_valid_proba))
#
# predict_test_proba = rf.predict_proba(x_test_dummies)[:,1]
#
# print(pd.DataFrame({'ID':test['ID'],'TravelInsurance':predict_test_proba}))










# # 다이어트약의 전후 체중 변화 기록이다.
# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p3_1.csv
# # 투약 후 체중에서 투약 전 체중을 뺏을 때 값은 일반 적으로 세가지 등급으로 나눈다.
# # -3이하 : A등급
# # -3초과 0이하 : B등급
# # 0 초과 : C등급.
# # 약 실험에서 A,B,C 그룹간의 인원 수 비율은 2:1:1로 알려져 있다.
# # 위 데이터 표본은 각 범주의 비율에 적합한지 카이제곱 검정하려한다.
#
# import pandas as pd
# df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p3_1.csv')
#
# # A등급에 해당하는 유저는 몇명인지 확인하라
# print(df.info())
# df['투약'] = df['투약후'] - df['투약전']
#
# def 등급(x):
#     if x <= -3:
#         return 'A'
#     elif x > -3 and x <= 0:
#         return 'B'
#     elif x > 0:
#         return 'C'
#
# df['등급'] = df['투약'].apply(등급)
#
# print(df)
#
# print(df['등급'].unique())
#
# print(df[df['등급'] == 'A'].shape[0])
#
#
# # 카이제곱검정 통계량을 반올림하여 소숫점 이하 3째자리까지 구하여라
#
# 측정빈도 = list(df.groupby('등급').size().sort_index().values)
# print(측정빈도)
#
# n = len(df)
#
# 기대빈도 = [n*2/4, n*1/4, n*1/4]
#
# print(기대빈도)
#
# from scipy.stats import chisquare
#
# print(round(chisquare(측정빈도,기대빈도).statistic,3))
#
#
# # 카이제곱 검정 p값을 반올림하여 소숫점 이하 3자리까지 구하고,
# # 유의수준 0.05하에서 귀무가설과 대립가설중 유의한 가설을 하나를 선택하시오(귀무/대립)
#
#
# p = round(chisquare(측정빈도,기대빈도).pvalue,3)
# print(p)
#
# if p >= 0.05:
#     print('귀무')
# elif p < 0.05:
#     print('대립')











# # 다이어트약의 전후 체중 변화 기록이다.
# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p3_2.csv
# # A,B 공장에서 생산한 기계들의 rpm 값들을 기록한 데이터이다.
# # 대응 표본 t 검정을 통해 B공장 제품들이 A 공장 제품들보다 rpm이 높다고 말할 수 있는지 검정하려한다
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e3_p3_2_.csv')
#
# # A,B 공장 각각 정규성을 가지는지 샤피로 검정을 통해 확인하라. (각 공장의 pvalue 출력할 것)
#
# print(df.info())
# from scipy.stats import shapiro
#
#
# a = df[df['group'] == 'A']['rpm']
# b = df[df['group'] == 'B']['rpm']
#
# print(shapiro(a).pvalue)
# print(shapiro(b).pvalue)
#
#
# # A,B 공장 생산 제품의 rpm은 등분산성을 가지는지 바틀렛 검정을 통해 확인하라.
# # (각 공장의 pvalue 출력할 것)
#
# from scipy.stats import bartlett
#
# print(bartlett(a,b).pvalue)
#
#
# # 대응 표본 t 검정을 통해 B공장 제품들의 rpm이 A 공장 제품의 rpm보다 크다고 말할 수 있는지 검정하라.
# # pvalue를 소숫점 이하 3자리까지 출력하고 귀무가설, 대립가설 중 하나를 출력하라*
#
# from scipy.stats import ttest_rel
#
# p = round(ttest_rel(b,a,alternative = 'greater').pvalue,3)
# print(p)
#
# if p >= 0.05:
#     print('귀무')
# elif p < 0.05:
#     print('대립')