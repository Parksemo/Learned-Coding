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





# 33번 각 요일별 가장 많이 이용한 대여소의 이용횟수와 대여소 번호를 데이터 프레임으로 출력하라
df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bicycle/seoul_bi.csv')
print(df.info())
df['대여일자'] = pd.to_datetime(df['대여일자'])
df['day_name'] = df['대여일자'].dt.day_name()
print(df[['day_name','대여소번호']].groupby(['day_name','대여소번호'],as_index = False).size().sort_values(['day_name','size'],ascending=False).drop_duplicates('day_name').reset_index(drop=True))












