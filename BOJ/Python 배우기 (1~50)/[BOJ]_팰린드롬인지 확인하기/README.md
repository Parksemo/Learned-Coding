## 팰린드롬인지 확인하기
[문제링크](https://www.acmicpc.net/problem/10988)

![팰린드롬인지 확인하기](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%ED%8C%B0%EB%A6%B0%EB%93%9C%EB%A1%AC%EC%9D%B8%EC%A7%80%20%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0.JPG?raw=true)
<br>
```python
단어 = input() # 첫째 줄에 단어가 주어진다.
# 팰린드롬이란 앞으로 읽을 때와 거꾸로 읽을 때 똑같은 단어를 말한다. 
if 단어 == 단어[::-1]: #  팰린드롬이면 1, 아니면 0을 출력
    print(1)
else:
    print(0)
```