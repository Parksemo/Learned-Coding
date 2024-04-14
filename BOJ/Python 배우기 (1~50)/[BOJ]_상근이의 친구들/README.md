## 상근이의 친구들
[문제링크](https://www.acmicpc.net/problem/5717)

![상근이의 친구들](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%83%81%EA%B7%BC%EC%9D%B4%EC%9D%98%20%EC%B9%9C%EA%B5%AC%EB%93%A4.JPG?raw=true)
<br>
```python
while True: # 입력은 여러 개의 테스트 케이스로 이루어져 있다. 
    M, F = map(int,input().split()) # 각 테스트 케이스는 두 정수 M과 F로 이루어져 있으며, 각각은 상근이의 남자 친구의 수와 여자 친구의 수이다.
    if M == 0 and F == 0: # 입력의 마지막 줄에는 0이 두 개 주어진다.
        break
    print(M+F) # 각 테스트 케이스마다 상근이의 친구의 수를 출력한다.
```