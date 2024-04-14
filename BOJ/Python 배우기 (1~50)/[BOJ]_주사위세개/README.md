## 주사위세개
[문제링크](https://www.acmicpc.net/problem/2480)

![주사위세개](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%A3%BC%EC%82%AC%EC%9C%84%EC%84%B8%EA%B0%9C.JPG?raw=true)
<br>
```python
dice = list(map(int,input().split())) # 던져진 주사위 눈

# 1.같은 눈이 3개가 나오면 10,000원+(같은 눈)×1,000원의 상금을 받게 된다.
if dice[0] == dice[1] == dice[2]: 
  print(10000+dice[0]*1000)
# 2.같은 눈이 2개만 나오는 경우에는 1,000원+(같은 눈)×100원의 상금을 받게 된다. 
elif dice[0] == dice[1]:
  print(1000+dice[0]*100)
elif dice[0] == dice[2]:
  print(1000+dice[0]*100)
elif dice[1] == dice[2]:
  print(1000+dice[1]*100)
# 3.모두 다른 눈이 나오는 경우에는 (그 중 가장 큰 눈)×100원의 상금을 받게 된다.  
else:
  print(max(dice)*100)
```