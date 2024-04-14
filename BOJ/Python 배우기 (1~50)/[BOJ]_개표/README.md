## 개표
[문제링크](https://www.acmicpc.net/problem/10102)

![개표](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EA%B0%9C%ED%91%9C.JPG?raw=true)
<br>
```python
V = int(input()) # 심사위원의 수 V
투표 = input() # 각 심사위원이 투표한 결과
if len(투표) == V:
    if 투표.count('A') < 투표.count('B'): # B가 받은 표가 A보다 많은 경우에는 B
        print('B')
    elif 투표.count('A') > 투표.count('B'): # A가 받은 표가 B보다 많은 경우에는 A
        print('A')
    elif 투표.count('A') == 투표.count('B'): # 같은 경우에는 Tie
        print('Tie')
```