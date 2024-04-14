CT = 100
ST = 100 #기본점수

n = int(input()) # 라운드 지정

for i in range(n): # n라운드
    C, S = map(int,input().split()) # 주사위 굴리기
    if C > S: # 점수잃기
        ST -= C
    elif C < S:
        CT -= S

print(CT,ST, sep='\n')