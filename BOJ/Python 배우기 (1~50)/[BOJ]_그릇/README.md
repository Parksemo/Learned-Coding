## 그릇
[문제링크](https://www.acmicpc.net/problem/7567)

![그릇](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EA%B7%B8%EB%A6%87.JPG?raw=true)
<br>
```python
그릇=input()
높이=10 # 그릇을 바닥에 놓았을 때 그 높이는 10cm 이다.
for i in range(1,len(그릇)):
    if 그릇[i] == 그릇[i-1]: # 두 개의 그릇을 같은 방향으로 포개면 그 높이는 5cm만 증가된다.
        높이+=5
    elif 그릇[i] != 그릇[i-1]: # 그릇이 서로 반대방향으로 쌓이면 높이는 그릇만큼, 즉 10cm 늘어난다.
        높이+=10
print(높이)
```