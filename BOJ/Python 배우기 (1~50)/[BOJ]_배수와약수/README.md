## 배수와 약수
[문제링크](https://www.acmicpc.net/problem/5086)

![배수와 약수](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EB%B0%B0%EC%88%98%EC%99%80%20%EC%95%BD%EC%88%98.JPG?raw=true)
<br>
```python
while True: # 입력은 여러 테스트 케이스로 이루어져 있다.
    A, B = map(int,input().split())
    if A == 0 and B == 0: # 마지막 줄에는 0이 2개 주어진다. 
        break
    elif A<B and B%A == 0: # 첫 번째 숫자가 두 번째 숫자의 약수이면 factor
        print('factor')
    elif A>B and A%B == 0: # 첫 번째 숫자가 두 번째 숫자의 배수이면 multiple
        print('multiple')
    else: # 첫 번째 숫자가 두 번째 숫자의 약수와 배수 모두 아니면 neither
        print('neither')
```