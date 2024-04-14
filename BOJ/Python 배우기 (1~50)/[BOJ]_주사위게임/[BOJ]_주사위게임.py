N = int(input())
상금 = []
for _ in range(N):
      dice = list(map(int,input().split()))
      if dice[0] == dice[1] == dice[2]:
        상금.append(10000+dice[0]*1000)
      elif dice[0] == dice[1]:
        상금.append(1000+dice[0]*100)
      elif dice[0] == dice[2]:
        상금.append(1000+dice[0]*100)
      elif dice[1] == dice[2]:
        상금.append(1000+dice[1]*100)
      else:
        상금.append(max(dice)*100)
print(max(상금))