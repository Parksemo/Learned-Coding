import pandas as pd
# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/youtube.csv",index_col=0)
#
# # 3번 채널명을 바꾼 케이스가 있는지 확인하고 싶다.
# # channelId의 경우 고유값이므로 이를 통해 채널명을 한번이라도 바꾼 채널의 갯수를 구하여라
# print(df.info())
# print(df[['channelTitle','channelId']].drop_duplicates().value_counts('channelId'))
# data = df[['channelTitle','channelId']].drop_duplicates().value_counts('channelId')
# print(((data > 1)*1).sum())
#
#
# # 4번 일요일에 인기있었던 영상들중 가장많은 영상 종류(categoryId)는 무엇인가?
# df['trending_date2'] = pd.to_datetime(df['trending_date2'])
# print(df[df['trending_date2'].dt.day_name() == 'Sunday']['categoryId'].value_counts().index[0])
# group = df.groupby([df['trending_date2'].dt.day_name(),'categoryId'],as_index=False).size()
# #pivot(index=None, columns=None, values=None)
# print(group.pivot(index = 'categoryId', columns = 'trending_date2'))


# # 12번 수집된 각 video의 가장 최신화 된 날짜의 viewcount값을 출력하라
# channel = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/channelInfo.csv')
# video = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/videoInfo.csv')
# channel['ct'] = pd.to_datetime(channel['ct'])
# video['ct'] = pd.to_datetime(video['ct'])
# print(video.sort_values(['videoname','ct']).drop_duplicates('videoname', keep = 'last').reset_index(drop = True)[['videoname','ct','viewcnt']])
#
#
# # 13번 Channel 데이터중 2021-10-03일 이후 각 채널의 처음 기록 됐던 구독자 수(subcnt)를 출력하라
# print(channel[channel['ct'] >= '2021-10-03'].sort_values(['channelname','ct']).drop_duplicates('channelname')[['channelname','subcnt']].reset_index(drop =True))
#
# # 14번 각채널의 2021-10-03 03:00:00 ~ 2021-11-01 15:00:00 까지 구독자수 (subcnt) 의 증가량을 구하여라
# channel1 = channel[channel['ct'].dt.strftime('%Y-%m-%d %H') == '2021-10-03 03'][['channelname','subcnt']]
# channel1.columns = ['channelname','start_subcnt']
# channel2 = channel[channel['ct'].dt.strftime('%Y-%m-%d %H') == '2021-11-01 15'][['channelname','subcnt']]
# channel2.columns = ['channelname','end_subcnt']
# merge1 = pd.merge(channel1,channel2)
# merge1['증가량'] = merge1['end_subcnt'] - merge1['start_subcnt']
# print(merge1[['channelname','증가량']])
#
#
#
# # 15번 각 비디오는 10분 간격으로 구독자수, 좋아요, 싫어요수, 댓글수가 수집된것으로 알려졌다.
# # 공범 EP1의 비디오정보 데이터중 수집간격이 5분 이하, 20분이상인
# # 데이터 구간( 해당 시점 전,후) 의 시각을 모두 출력하라
# df1 = video[video['videoname'] == ' 공범 EP1'].sort_values('ct').reset_index(drop =True)
# import datetime
# print(df1[(df1['ct'].diff() >= datetime.timedelta(minutes = 20)) | (df1['ct'].diff() <= datetime.timedelta(minutes = 5))])
# # index : 721, 722, 1636
# print(df1[df1.index.isin([720,721,722,723,1635,1636,1637])])
#
#
#
# # 16번 각 에피소드의 시작날짜(년-월-일)를 에피소드 이름과 묶어 데이터 프레임으로 만들고 출력하라
# df1 = video[['videoname','ct']].sort_values(['videoname','ct']).drop_duplicates('videoname').reset_index(drop = True)
# df1['date'] = df1['ct'].dt.date
# print(df1[['date','videoname']])
#
#
#
# # 17번 “공범” 컨텐츠의 경우 19:00시에 공개 되는것으로 알려져있다.
# # 공개된 날의 21시의 viewcnt, ct, videoname 으로 구성된 데이터 프레임을
# # viewcnt를 내림차순으로 정렬하여 출력하라
# print(video[video['ct'].dt.hour == 21][['videoname','viewcnt', 'ct']].sort_values(['videoname','ct']).drop_duplicates('videoname').reset_index(drop = True).sort_values('viewcnt',ascending = False))
#
#
#
# # 19번 2021-11-01 00:00:00 ~ 15:00:00까지 각 에피소드별 viewcnt의 증가량을 데이터 프레임으로 만드시오
# data = video[(video['ct'].dt.strftime('%Y-%m-%d %H:%M:%S') >= '2021-11-01 00:00:00') & (video['ct'].dt.strftime('%Y-%m-%d %H:%M:%S') <= '2021-11-01 15:00:00')]
# def 증가량(x):
#     return max(x) - min(x)
# print(data[['videoname','viewcnt']].groupby('videoname').agg(증가량))
#
#
# # 20번 video 데이터 중에서 중복되는 데이터가 존재한다.
# # 중복되는 각 데이터의 시간대와 videoname 을 구하여라
# print(video[video.index.isin(set(video.index) - set(video.drop_duplicates().index))][['videoname','ct']])




# df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/worldcup/worldcupgoals.csv')
# # 23번 Years 컬럼은 년도 -년도 형식으로 구성되어있고, 각 년도는 4자리 숫자이다.
# # 년도 표기가 4자리 숫자로 안된 케이스가 존재한다. 해당 건은 몇건인지 출력하라
# print(df['Years'].str.split('-'))
# def four(x):
#     for i in x:
#         if len(i) != 4:
#             return False
#         else:
#             return True
# print(df[df['Years'].str.split('-').apply(four) == False].shape[0])
#
#
#
# # 25번 월드컵 출전횟수를 나타내는 ‘LenCup’ 컬럼을 추가하고 4회 출전한 선수의 숫자를 구하여라
# def num(x):
#     return len(x)
# df['LenCup'] = df['Years'].str.split('-').apply(num)
# print(df[df['LenCup']  == 4].shape[0])
#
#
# # 28번 이름에 ‘carlos’ 단어가 들어가는 선수의 숫자는 몇 명인가? (대, 소문자 구분 x)
#
# print(df[df['Player'].str.lower().str.contains('carlos')].shape[0])





# # 33번 각 요일별 가장 많이 이용한 대여소의 이용횟수와 대여소 번호를 데이터 프레임으로 출력하라
# import pandas as pd
# df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bicycle/seoul_bi.csv')
# print(df.info())
# df['대여일자'] = pd.to_datetime(df['대여일자'])
# df['day_name'] = df['대여일자'].dt.day_name()
# print(df[['day_name','대여소번호']].groupby(['day_name','대여소번호'],as_index = False).size().sort_values(['day_name','size'],ascending=False).drop_duplicates('day_name').reset_index(drop=True))
#
#
#
# # 34번 나이대별 대여구분 코드의 (일일권/전체횟수) 비율을 구한 후
# # 가장 높은 비율을 가지는 나이대를 확인하라.
# # 일일권의 경우 일일권 과 일일권(비회원)을 모두 포함하라
# print(df['대여구분코드'].unique())
# print(df[df['대여구분코드'].isin(['일일권','일일권(비회원)'])].value_counts('연령대코드').sort_index())
# print(df.value_counts('연령대코드').sort_index())
# print((df[df['대여구분코드'].isin(['일일권','일일권(비회원)'])].value_counts('연령대코드').sort_index()) / (df.value_counts('연령대코드').sort_index()))
# data = (df[df['대여구분코드'].isin(['일일권','일일권(비회원)'])].value_counts('연령대코드').sort_index()) / (df.value_counts('연령대코드').sort_index())
# print(data.sort_values().index[-1])
#
#
#
#
# # 38번 평일 (월~금) 출근 시간대(오전 6,7,8시)의 대여소별 이용 횟수를 구해서
# # 데이터 프레임 형태로 표현한 후 각 대여시간별 이용 횟수의 상위 3개 대여소와 이용횟수를 출력하라
# df1 = df[(df['day_name'].isin(['Monday','Tuesday','Wednesday','Thursday','Friday'])) & (df['대여시간'].isin([6,7,8]))]
# print(df1.groupby(['대여시간','대여소번호']).size().to_frame('이용횟수').sort_values(['대여시간','이용횟수'],ascending = False).groupby('대여시간').head(3))
#
#
#
#
# # 40번 남성(‘M’ or ‘m’)과 여성(‘F’ or ‘f’)의 이동거리값의 평균값을 구하여라
# df['sex'] = df['성별'].map(lambda x : '남' if x in ['M','m'] else '여')
# print(df[['sex','이동거리']].groupby('sex').mean())







df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/happy2/happiness.csv',encoding='utf-8')
#
# # 44번 2018년도와 2019년도의 행복랭킹이 변화하지 않은 나라명의 수를 구하여라
#
# # 변화하지 않은 나라는 중복으로 제거가 되기에
# # 기존 data개수에서 중복제거한 data개수를 빼면 중복제거된 수를 구할 수있다.
# print(len(df[['행복랭킹','나라명']]) - len(df[['행복랭킹','나라명']].drop_duplicates()))
#
#
#
#
# # 45번 2019년도 데이터들만 추출하여 각변수간 상관계수를 구하고
# # 내림차순으로 정렬한 후 상위 5개를 데이터 프레임으로 출력하라. 컬럼명은 v1,v2,corr으로 표시하라
#
#
# df1 = df[df['년도'] == 2019].drop(columns = ['년도']).corr().unstack().to_frame().reset_index()
# df2 = df1[df1[0] != 1].sort_values(0,ascending = False).drop_duplicates(0).head().reset_index(drop =True)
# df2.columns = ['v1', 'v2', 'corr']
# print(df2)
#
#
#
# # 49번 2018년도 행복랭킹 50위 이내에 포함됐다가 2019년 50위 밖으로 밀려난 국가의 숫자를 구하여라
# print(len(set(df[(df['년도'] == 2018) & (df['행복랭킹'] <= 50)]['나라명']) - set(df[(df['년도'] == 2019) & (df['행복랭킹'] <= 50)]['나라명'])))
#
#
#
# # 50번 2018년,2019년 모두 기록이 있는 나라들 중 년도별 행복점수가 가장 증가한 나라와 그 증가 수치는?
# set_l = set(df[df['년도'] == 2018]['나라명']) & set(df[df['년도'] == 2019]['나라명'])
# df1 = df[df['나라명'].isin(set_l)]
# # 년도별 증가 수치를 보기위해 2018년 수치를 음수로 변환후 서로 더하면 증감수치를 알수 있다
# df1.loc[df1['년도'] == 2018,'점수'] = df1[df1['년도'] == 2018]['점수'] *(-1)
# print(df1.groupby('나라명').sum()[['점수']].sort_values('점수',ascending = False).head(1))






# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/consum/Tetuan%20City%20power%20consumption.csv')
# # 51번 DateTime컬럼을 통해 각 월별로 몇개의 데이터가 있는지 데이터 프레임으로 구하여라
# df['DateTime'] = pd.to_datetime(df['DateTime'])
# print(df['DateTime'].dt.month.value_counts().sort_index().to_frame())





# df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/pok/Pokemon.csv')
# # 65번 ‘HP’, ‘Attack’, ‘Defense’, ‘Sp. Atk’, ‘Sp. Def’, ‘Speed’
# # 간의 상관 계수중 가장 절댓값이 큰 두 변수와 그 값을 구하여라
# df1 = df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].corr().unstack().reset_index().rename(columns = {0:'corr'})
# print(df1[df1['corr'] != 1].sort_values('corr',ascending = False).drop_duplicates('corr').reset_index(drop = True).iloc[0,:])
#
#
#
#
# # 70번 한번씩만 존재하는 (Type1 , Type2)의 쌍을 각 세대(Generation)은 각각 몇개씩 가지고 있는가?
# df1 = df[['Type 1','Type 2']].value_counts()
# df2 = df1[df1 == 1].index
# lst = []
# for i in df2:
#     t1 = i[0]
#     t2 = i[1]
#     tt = df[(df['Type 1'] == t1) & (df['Type 2'] == t2)]
#     lst.append(tt)
# df3 = pd.concat(lst)
# print(df3.value_counts('Generation').sort_index())




# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/train.csv")
# # 86번 수축기혈압과 이완기 혈압기 수치의 차이를 새로운 컬럼(‘혈압차’) 으로 생성하고,
# # 연령대 코드별 각 그룹 중 ‘혈압차’ 의 분산이 5번째로 큰 연령대 코드를 구하여라
# print(df.info())
# df['혈압차'] = df['수축기혈압'] - df['이완기혈압']
# print(df.groupby('연령대코드(5세단위)').agg('var')['혈압차'].sort_values(ascending = False).index[4])
#
#
#
#
# # 87번 비만도를 나타내는 지표인 WHtR는 허리둘레 / 키로 표현한다.
# # 일반적으로 0.58이상이면 비만으로 분류한다.
# # 데이터중 WHtR 지표상 비만인 인원의 남/여 비율을 구하여라
#
# df['WHtR'] = df['허리둘레'] / df['신장(5Cm단위)']
# df['비만판별'] = df['WHtR'].map(lambda x : '비만' if x >= 0.58 else '정상')
# df1 = df[df['비만판별'] == '비만']
# print((df1[df1['성별코드'] == 'M'].shape[0]) / (df1[df1['성별코드'] == 'F'].shape[0]))







# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/insurance/train.csv")
# # 89번 vehicle_age에 따른 각 성별(gender)그룹의 Annual_Premium값의 평균을 구하여 아래 테이블과 동일하게 구현하라
# print(df.info())
# data = df[['Vehicle_Age','Gender','Annual_Premium']].groupby(['Vehicle_Age','Gender'],as_index = False).mean()
# print(data.pivot(index='Vehicle_Age',columns = 'Gender',values = 'Annual_Premium'))






# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/mobile/train.csv")
# # 90번 price_range 의 각 value를 그룹핑하여
# # 각 그룹의 n_cores 의 빈도가 가장높은 value와 그 빈도수를 구하여라
# print(df.info())
# print(df[['price_range','n_cores']].groupby(['price_range','n_cores']).size().sort_values(ascending = False).groupby('price_range').head(1))







# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/airline/train.csv")
# # 92번 Arrival Delay in Minutes 컬럼이 결측치인 데이터들 중
# # ‘neutral or dissatisfied’ 보다 ‘satisfied’의 수가 더 높은 Class는 어디 인가?
# print(df.info())
# df1 = df[df['Arrival Delay in Minutes'].isnull()].reset_index(drop=True)
# df2 = df1[['Class','satisfaction']].groupby(['Class','satisfaction'],as_index = False).size()
# df3 = df2.pivot(index = 'Class',columns = 'satisfaction')
# print(df3[df3['size']['neutral or dissatisfied'] < df3['size']['satisfied']])





# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/admission/train.csv")
# # 96번 Serial No. 컬럼을 제외하고 ‘Chance of Admit’을 종속변수,
# # 나머지 변수를 독립변수라 할때, 랜덤포레스트를 통해 회귀 예측을 할 떄
# # 변수중요도 값을 출력하라 (시드값에 따라 순서는 달라질수 있음)
# from sklearn.ensemble import RandomForestRegressor
# print(df.info())
# df1 = df.drop(columns= ['Serial No.'])
# x = df1.drop(columns = ['Chance of Admit'])
# y = df1['Chance of Admit']
# rf = RandomForestRegressor()
# rf.fit(x, y)
# print(rf.feature_importances_)
# print(pd.DataFrame({'importance':rf.feature_importances_},x.columns).sort_values('importance',ascending = False))





# df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/nba/nba.csv",encoding='latin',sep=';')
# # 106번 선수들의 이름은 first_name+ 공백 + last_name으로 이루어져 있다.
# # 가장 많은 first_name은 무엇이며 몇 회 발생하는지 확인하라
# print(df['Player'].str.split(' ').str[0].str.lower().value_counts(ascending = False).index[0])