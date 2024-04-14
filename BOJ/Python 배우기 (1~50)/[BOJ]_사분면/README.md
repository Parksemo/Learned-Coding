## 사분면
[문제링크](https://www.acmicpc.net/problem/9610)

![사분면](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%82%AC%EB%B6%84%EB%A9%B4.JPG?raw=true)
<br>
```python
n = int(input())
Q1 = 0
Q2 = 0
Q3 = 0
Q4 = 0
AXIS = 0
for _ in range(n):# 첫째 줄에 점의 개수 n이 주어진다.
    x, y = map(int,input().split()) # 다음 n개 줄에는 점의 좌표 (x, y)가 주어진다.
    if x == 0 or y == 0: # 축
        AXIS += 1
    elif x > 0 and y > 0: # 1사분면
        Q1 += 1
    elif x < 0 and y > 0: # 2사분면
        Q2 += 1
    elif x < 0 and y < 0: # 3사분면
        Q3 += 1
    elif x > 0 and y < 0: # 4사분면
        Q4 += 1
# 출력
print(f'''Q1: {Q1}
Q2: {Q2}
Q3: {Q3}
Q4: {Q4}
AXIS: {AXIS}
''')
```