## OX퀴즈
[문제링크](https://www.acmicpc.net/problem/8958)

![OX퀴즈](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5DOX%ED%80%B4%EC%A6%88.JPG?raw=true)
<br>
```python
T = int(input())
for _ in range(T): # 첫째 줄에 테스트 케이스의 개수가 주어진다.
    총점수 = 0 # 퀴즈결과에 대한 최종 점수
    연속점수 = 0 # O 연속시 누적 점수
    퀴즈 = input()
    if 퀴즈[0] == 'O': # 젤 첫 번째 값은 이전 값이 없기 때문에 따로 설정
        총점수 += 1
        연속점수 += 1
    for i in range(1, len(퀴즈)): # 두 번째부터 시작
        if 퀴즈[i-1] == 'O' and 퀴즈[i] == 'O': # 연속점수 누적 및 총 점수에 부여
            연속점수 += 1
            총점수 += 연속점수
        elif 퀴즈[i-1] == 'O' and 퀴즈[i] == 'X': # 점수누적 초기화
            연속점수 = 0
        elif 퀴즈[i-1] == 'X' and 퀴즈[i] == 'O': # 점수획득 및 연속점수 활성화
            연속점수 = 1
            총점수 += 연속점수
    print(총점수)
```