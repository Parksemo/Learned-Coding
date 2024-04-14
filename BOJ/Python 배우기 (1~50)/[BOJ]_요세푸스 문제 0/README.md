## 요세푸스 문제 0
[문제링크](https://www.acmicpc.net/problem/11866)

![요세푸스 문제 0](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%9A%94%EC%84%B8%ED%91%B8%EC%8A%A4%20%EB%AC%B8%EC%A0%9C%200.JPG?raw=true)

![요세푸스 문제 0_1](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%9A%94%EC%84%B8%ED%91%B8%EC%8A%A4%20%EB%AC%B8%EC%A0%9C%200_1.JPG?raw=true)

```python
import sys
N, K = map(int,sys.stdin.readline().split())
N_l = list(range(1,N+1))

if K == 1:
    print('<',end='')
    print(*N_l, sep=', ', end='')
    print('>')

elif K != 1:
    K = K-1
    KK = K

    N_O = []
    while True:
        if len(N_l) > KK:
            N_O.append(N_l.pop(KK))
        elif len(N_l) <= KK:
            KK = KK%len(N_l)
            N_O.append(N_l.pop(KK))
        KK = KK + K

        if len(N_l) == 1:
            N_O.append(N_l.pop())
            break
    print('<',end='')
    print(*N_O, sep=', ', end='')
    print('>')
```