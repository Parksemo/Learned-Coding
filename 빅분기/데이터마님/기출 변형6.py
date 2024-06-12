# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e6_p1_1.csv')
# print(df)
#
# print(df.info())
#
# print(df['출동일자'].astype('str') + df['출동시각'].astype('str').str.zfill(6))
#
# print(pd.to_datetime(df['출동일자'].astype('str') + df['출동시각'].astype('str').str.zfill(6)) - pd.to_datetime(df['신고일자'].astype('str') + df['신고시각'].astype('str').str.zfill(6)))
# a = pd.to_datetime(df['출동일자'].astype('str') + df['출동시각'].astype('str').str.zfill(6)) - pd.to_datetime(df['신고일자'].astype('str') + df['신고시각'].astype('str').str.zfill(6))
#
#
# df['소요시간'] = a.dt.total_seconds()
#
# print(df['소요시간'])
#
# print(df.groupby('소방서명').mean()['소요시간'].sort_values().reset_index().iloc[2,:])
#





# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e6_p1_2.csv')
# df
#
# df1 = df[df['학교세부유형'] == '일반중학교']
# print(df1['시도'].value_counts().index[1])
#
# df2 = df1[df1['시도'] == df1['시도'].value_counts().index[1]]
# print(df2.isnull().sum())
# print(df2)
# df2['비율'] = df2['일반학급_학생수_계'] / df2['교원수_총계_계']
# print(df2.dropna().sort_values('비율',ascending = False)['교원수_총계_계'].values[0])






# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e6_p1_3.csv')
# print(df.info())
# df['총범죄수'] = df['절도']+df['사기']+df['배임']+df['방화']+df['폭행']
# print(df)
#
# lst = []
# for x in ['2018년_','2019년_']:
#     for y in range(1,5):
#         for z in range(3):
#             lst.append(x + str(y) + '분기')
# print(lst)
# print(len(lst))
#
# df['분기'] = lst
#
# print(df)
#
# print(df.groupby('분기').mean()['총범죄수'].sort_values(ascending=False).index[0])
#
# print(df[df['분기'] == df.groupby('분기').mean()['총범죄수'].sort_values(ascending=False).index[0]].sort_values('사기',ascending = False)['사기'].values[0])




# import pandas as pd
# train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/ep6_p2_train.csv')
# test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/ep6_p2_test.csv')
# # 건강 상태 분류문제
# # 예측 변수 General_Health, test.csv에 대해 ID별로 General_Health 값을 예측하여 제출
# # 제출 데이터 컬럼은 ID와 General_Health 두개만 존재해야함.
# # 평가지표는 f1score
#
# print(train.head(),test.head())
#
# print(train['General_Health'].unique())
# print(train.isnull().sum(),test.isnull().sum())
# print(train.nunique(),test.nunique())
#
# x = train.drop(columns = ['ID','General_Health'])
# test_x = test.drop(columns = ['ID'])
#
# x_dummies = pd.get_dummies(x)
# x_test_dummies = pd.get_dummies(test_x)
# x_test_dummies = x_test_dummies.reindex(columns = x_dummies.columns, fill_value = 0)
# y = train['General_Health']
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
#
# X_train, X_valid, Y_train, Y_valid = train_test_split(x_dummies,y,stratify=y,random_state=42)
# rf = RandomForestClassifier(random_state = 42)
# rf.fit(X_train, Y_train)
# predict_valid_label = rf.predict(X_valid)
#
# from sklearn.metrics import f1_score
#
# print(f1_score(Y_valid,predict_valid_label,average='macro'))
#
# predict_test_label = rf.predict(x_test_dummies)
#
# print(pd.DataFrame({'ID':test['ID'],'General_Health':predict_test_label}))
#
# # pd.DataFrame({'ID':test['ID'],'General_Health':predict_test_label}).to_csv('result.csv',index = False)
# # result = pd.read_csv('result.csv')
# # print(result)






# A 도시의 남성 600명과 여성 550명이 있다.
# 남성들 중 흡연자 비율은 0.2이며 여성들 중 흡연자 비율은 0.26이다.
# 남성과 여성 간에 흡연 여부에 따른 인구 비율이 다른지 확인하고 싶다.
# 유의 수준 0.05하 귀무가설에 대해 기각 / 채택 여부와 p-value값을 각각 출력하라

# smk_male = 600 * 0.2
# n_smk_male = 600 * 0.8
# smk_female = 550 * 0.26
# n_smk_female = 550 * 0.74
#
# import pandas as pd
#
# data = pd.DataFrame({'흡연':[smk_male,smk_female],'비흡연':[n_smk_male,n_smk_female]},index = ['남성','여성'])
#
# from scipy.stats import chi2_contingency
#
# chi2, p, df, ext = chi2_contingency(data)
#
# print(p)
#
# if p >= 0.05:
#     print('귀무(독립)')
# else :
#     print('기각(종속)')






# # 연령 몸무게 콜레스테롤 수치 데이터
# import pandas as pd
# df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/adp/28/p7.csv')
#
# # age와 Cholesterol을 가지고 weight를 예측하는 선형 회귀 모델을 만들려고한다.
# # age의 회귀 계수를 구하여라
#
# import statsmodels.api as sm
#
# X = sm.add_constant(df[['age','Cholesterol']])
#
# model = sm.OLS(df['weight'], X)
#
# result = model.fit()
#
# print(result.summary())
#
# print(result.params.age)
#
# # age가 고정일 때 Cholesterol와 weight가 선형관계에 있다는 가설을 유의수준 0.05하에 검정하라
#
# p = result.pvalues['Cholesterol']
#
# if p >= 0.05:
#     print('귀무(선형x)')
# else :
#     print('기각(선형o)')
#
# # age가 55, Cholesterol가 72.6일때 위 모델을 기반으로 weight값을 예측하라.
#
# print(result.predict([1,55,72.6])[0])
#
