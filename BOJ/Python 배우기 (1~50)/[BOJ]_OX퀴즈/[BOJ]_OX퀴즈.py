T = int(input())
for _ in range(T):
    총점수 = 0
    연속점수 = 0
    퀴즈 = input()
    if 퀴즈[0] == 'O':
        총점수 += 1
        연속점수 += 1
    for i in range(1, len(퀴즈)):
        if 퀴즈[i-1] == 'O' and 퀴즈[i] == 'O':
            연속점수 += 1
            총점수 += 연속점수
        elif 퀴즈[i-1] == 'O' and 퀴즈[i] == 'X':
            연속점수 = 0
        elif 퀴즈[i-1] == 'X' and 퀴즈[i] == 'O':
            연속점수 = 1
            총점수 += 연속점수
    print(총점수)