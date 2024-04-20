T = int(input()) # 테스트 케이스 수 T

for i in range(0,T):
    N = int(input()) # 학교의 숫자 정수 N
    SL = {} # 학교(key), 술의 양(value)
    for j in range(0,N):
        S, L = map(str,input().split())
        L = int(L)
        SL[S] = L
    print(max(SL,key=SL.get)) # 가장 술 소비가 많은 학교 이름 출력