# # 데이터설명 : 보스턴집값, 각 행은 지역구별 집값관련된 메타정보 : https://www.kaggle.com/datasets/arunjathari/bostonhousepricedata
# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e2_p1_1.csv
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e2_p1_1.csv')
#
# # 주어진 Dataset에서 CRIM값이 가장 큰 10개의 지역을 구하고
# # 10개의 지역의 CRIM값을 그 중 가장 작은 값으로 대체하라.
# # 그리고 AGE 컬럼 값이 80이상인 대체 된 CRIM 평균값을 구하라
#
# print(df.info())
# df = df.sort_values('CRIM',ascending=False).reset_index(drop = True)
#
# df.loc[:9,'CRIM'] = df.loc[:9,'CRIM'].min()
#
# print(df.head(10))
#
# print(df[df['AGE'] >= 80]['CRIM'].mean())
#
# # 1-1에서 사용한 데이터에서 RM 중앙값으로 해당 컬럼의 결측치를 대체하라.
# # 그리고 해당 컬럼의 결측치 대치 전후의 표준편차 차이의 절댓값을 소숫점 이하 3째자리 까지 구하라
#
# std_before = df['RM'].std()
#
# df['RM'] = df['RM'].fillna(df['RM'].median())
#
# std_after = df['RM'].std()
#
# print(round(abs(std_before - std_after),3))
#
# # 주어진 Dataset의 DIS 평균으로 부터 1.5 * 표준편차를 벗어나는 영역을
# # 이상치라고 판단하고 DIS 컬럼의 이상치들의 합을 구하여라.
#
# lower = df['DIS'].mean() - (1.5 * df['DIS'].std())
# upper = df['DIS'].mean() + (1.5 * df['DIS'].std())
#
# print(df[(df['DIS'] < lower) | (df['DIS'] > upper)]['DIS'].sum())








# # 데이터 설명 : e-commerce 배송의 정시 도착여부 (1: 정시배송 0 : 정시미배송)
# # x_train: https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_train.csv
# # y_train: https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_train.csv
# # x_test: https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_test.csv
# # x_label(평가용) : https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_test.csv
# # 데이터 출처 :https://www.kaggle.com/datasets/prachi13/customer-analytics (참고, 데이터 수정)
# # x_train 데이터로 학습한 모델을 x_test에 적용하여 예측한 결과를 제출하라
# # 평가 지표는 f1_score이다.
#
# import pandas as pd
# #데이터 로드
# x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_train.csv")
# y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_train.csv")
# x_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_test.csv")
#
#
# print(x_train.info())
# print()
# print(x_train.isnull().sum())
# print()
# print(x_test.isnull().sum())
# print()
# print(x_train.nunique())
# print()
# print(x_test.nunique())
#
#
# x = x_train.drop(columns=['ID'])
# test_x = x_test.drop(columns=['ID'])
#
# x_dummies = pd.get_dummies(x)
# x_test_dummies = pd.get_dummies(test_x)
# x_test_dummies = x_test_dummies[x_dummies.columns]
# y = y_train['Reached.on.Time_Y.N']
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
#
# X_train, X_valid, Y_train, Y_valid = train_test_split(x_dummies, y ,stratify = y ,random_state=42)
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, Y_train)
# predict_valid_label = rf.predict(X_valid)
#
# from sklearn.metrics import f1_score
#
# print(f1_score(Y_valid,predict_valid_label))
#
# predict_test_label = rf.predict(x_test_dummies)
#
# print(pd.DataFrame({'ID':x_test['ID'],'Reached.on.Time_Y.N':predict_test_label}))







# # 어느 호수에서 잡은 물고기 122마리 길이 데이터(자체제작)
# # DataUrl = https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e2_p3_1.csv
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e2_p3_1.csv')
#
# # 122마리의 height 평균값을 m(미터) 단위로 소숫점 이하 5자리까지 실수 값만 출력하라
#
# print(df.info())
#
# df['height(m)'] = df['height'].str.replace('cm','').astype('float')
# df['height(m)'] = df['height(m)'] / 100
#
# print(df)
# print(df.info())
# print(round(df['height(m)'].mean(),5))
#
#
# # 모집단의 평균 길이가 30cm 인지 확인하려 일표본 t 검정을 시행하여 확인하려한다.
# # 검정통계량을 소숫점 이하 3째자리까지 구하여라
#
# from scipy.stats import ttest_1samp
#
# df['height'] = df['height'].str.replace('cm','').astype('float')
#
# print(round(ttest_1samp(df['height'],30).statistic,3))
#
# # 위의 통계량에 대한 p-값을 구하고 (반올림하여 소숫점 이하 3째자리)
# # 유의수준 0.05하에서 귀무가설과 대립가설중 유의한 가설을 하나를 선택하시오(귀무/대립)
#
# p = round(ttest_1samp(df['height'],30).pvalue,3)
#
# print(p)
#
# if p >= 0.05:
#     print('귀무')
# elif p < 0.05:
#     print('대립')








# # 조사결과 70%의 성인 남성이 3년 동안에 적어도 1번 치과를 찾는다고 할때,
# # 21명의 성인 남성이 임의로 추출되었다고 하자.
#
# # 21명 중 16명 미만이 치과를 찾았을 확률(반올림하여 소숫점 이하 3자리)
#
# # 이항분포문제
#
# # 이항 누적 확률 값을 구해야 되기 때문에 binom.cdf(k,n,p) 사용
# from scipy.stats import binom
#
# # P(X < k) 계산
# # cdf가 k까지의 누적확률을 계산하기 16미만 확률은 k = 15 까지 누적확률을 구한다.
# print(round(binom.cdf(15,21,0.7),3))
#
#
#
# # 적어도 19명이 치과를 찾았을 확률(반올림하여 소숫점 이하 3자리)
#
# # P(X >= k) 계산
# # 적어도 19명 : 19 20 21 3개 누적 확률 -> 1에다가 18까지의 누적확률을 뺀다
#
# print(round(1- binom.cdf(18,21,0.7),3))