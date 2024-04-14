## 알람시계
[문제링크](https://www.acmicpc.net/problem/2884)

![알람시계](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%95%8C%EB%9E%8C%EC%8B%9C%EA%B3%84.JPG?raw=true)
<br>
```python
H, M = map(int,input().split()) # 첫째 줄에 두 정수 H와 M이 주어진다.
설정한알람시각총분 = H*60+M
설정해야하는알림시각총분 = 설정한알람시각총분-45 # 45분 일찍 알람 설정하기
출력_시 = 설정해야하는알림시각총분//60
if 출력_시 < 0: #0시에서 23시를 출력할 때 -1이 나오는 것을 방지
    출력_시 += 24
출력_분 = 설정해야하는알림시각총분%60
print(f'{출력_시} {출력_분}')
```