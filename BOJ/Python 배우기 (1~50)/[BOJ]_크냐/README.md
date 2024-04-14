## 크냐
[문제링크](https://www.acmicpc.net/problem/4101)

![크냐](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%ED%81%AC%EB%83%90.JPG?raw=true)
<br>
```python
while True: # 여러개의 테스트 케이스가 이루어져 있다.
  A, B = map(int,input().split()) # 각 테스트 케이스는 한 줄로 이루어져 있으며, 두 정수가 주어진다.
  if A == 0 and B == 0: # 입력의 마지막 줄에는 0이 두 개 주어진다.
    break
  elif A > B: # 각 테스트 케이스마다, 첫 번째 수가 두 번째 수보다 크면 Yes를, 아니면 No를 한 줄에 하나씩 출력한다.
    print("Yes")
  else:
    print("No")
```