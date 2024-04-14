## 네번째점
[문제링크](https://www.acmicpc.net/problem/3009)

![네번째점](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EB%84%A4%EB%B2%88%EC%A7%B8%EC%A0%90.JPG?raw=true)
<br>
![네번째점2](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EB%84%A4%EB%B2%88%EC%A7%B8%EC%A0%902.JPG?raw=true)
<br>
<예제입력1번의 좌표 그래프>
<br>
x좌표와 y좌표에서 각각 값의 개수가 1인 값을 출력하면 된다.
<br>
```python
# 각각의 좌표 값 저장
x좌표 = []
y좌표 = []
for _ in range(3): # 세 점의 좌표가 한 줄에 하나씩 주어진다.
  x, y = map(int,input().split())
  x좌표.append(x)
  y좌표.append(y)
# 개수 확인을 위해 정렬
x좌표.sort()
y좌표.sort()
# 첫 번째값과 두 번째값은 무조건 다른 값임을 활용
# x좌표값의 개수가 1인 값을 저장
if x좌표.count(x좌표[0]) == 1:
      네번째점x좌표 = x좌표[0]
else:
      네번째점x좌표 = x좌표[2]
if y좌표.count(y좌표[0]) == 1:
      네번째점y좌표 = y좌표[0]
else:
      네번째점y좌표 = y좌표[2]
print(f'{네번째점x좌표} {네번째점y좌표}') # 직사각형의 네 번째 점의 좌표를 출력한다.
```