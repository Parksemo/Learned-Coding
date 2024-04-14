T = int(input()) # 요리시간 입력

if T%10 == 0 : # 3개의 버튼으로 동작 가능 여부 확인
    A = T//300 # A버튼
    TA = T%300 # A버튼을 누르고 남은 시간
    B = TA//60 # B버튼
    TB = TA%60 # B버튼을 누르고 남은 시간
    C = TB//10 # C버튼
    print(A, B ,C)
else :
    print(-1)