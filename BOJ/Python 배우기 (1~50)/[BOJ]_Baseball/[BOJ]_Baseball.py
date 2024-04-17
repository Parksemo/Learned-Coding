T = int(input()) #테스트 케이스 수 입력

for i in range (0,T): #T개의 테스트 케이스
    YT = 0 #연세대 합산 점수
    KT = 0 #고려대 합산 점수
    for j in range (0,9): #9회까지
        Y, K = map(int,input().split()) #회차별 양 팀 획득 점수
        YT += T #연세대 누적 점수
        KT += K #고려대 누적 점수
    if YT > KT : #경기 결과 출력
        print('Yonsei')
    elif YT < KT :
        print('Korea')
    else :
        print('Draw')