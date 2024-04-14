## 과자
[문제링크](https://www.acmicpc.net/problem/10156)

![과자](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EA%B3%BC%EC%9E%90.JPG?raw=true)
<br>
```python
K, N, M = map(int,input().split()) # 과자 한 개의 가격 K, 사려고 하는 과자의 개수 N, 현재 동수가 가진 돈 M
부모님께_받아야_하는_돈 = K*N-M
if 부모님께_받아야_하는_돈 >= 0:
  print(K*N-M) # 첫 줄에 동수가 부모님께 받아야 하는 돈의 액수를 출력한다. 
else:
  print(0)
```