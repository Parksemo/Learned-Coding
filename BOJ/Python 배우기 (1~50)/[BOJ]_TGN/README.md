## TGN
[문제링크](https://www.acmicpc.net/problem/5063)

![TGN](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5DTGN.JPG?raw=true)
<br>
```python
for _ in range(int(input())): # 첫째 줄에 테스트 케이스의 개수 N이 주어진다.
    # r은 광고를 하지 않았을 때 수익, e는 광고를 했을 때의 수익, c는 광고 비용
    r, e, c = map(int,input().split())
    #  광고를 해야 하면 "advertise", 하지 않아야 하면 "do not advertise", 광고를 해도 수익이 차이가 없다면 "does not matter"를 출력한다.
    if r < e-c:
        print('advertise')
    elif r == e-c:
        print('does not matter')
    elif r > e-c:
        print('do not advertise')
```