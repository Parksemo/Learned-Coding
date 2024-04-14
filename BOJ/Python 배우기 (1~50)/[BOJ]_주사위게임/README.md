## 주사위게임
[문제링크](https://www.acmicpc.net/problem/2476)

![주사위게임](https://github.com/Parksemo/Parksemo/blob/master/image/%5BBOJ%5D%EC%A3%BC%EC%82%AC%EC%9C%84%EA%B2%8C%EC%9E%84.JPG?raw=true)
<br>
```python
N = int(input()) # 참여하는 사람 수 N
상금 = []
for _ in range(N):
      dice = list(map(int,input().split()))
      # 1.같은 눈이 3개가 나오면 10,000원+(같은 눈)×1,000원의 상금을 받게 된다. 
      if dice[0] == dice[1] == dice[2]:
        상금.append(10000+dice[0]*1000)
      # 2.같은 눈이 2개만 나오는 경우에는 1,000원+(같은 눈)×100원의 상금을 받게 된다. 
      elif dice[0] == dice[1]:
        상금.append(1000+dice[0]*100)
      elif dice[0] == dice[2]:
        상금.append(1000+dice[0]*100)
      elif dice[1] == dice[2]:
        상금.append(1000+dice[1]*100)
      # 3.모두 다른 눈이 나오는 경우에는 (그 중 가장 큰 눈)×100원의 상금을 받게 된다.  
      else:
        상금.append(max(dice)*100)
print(max(상금)) #  가장 많은 상금을 받은 사람의 상금을 출력한다.
```