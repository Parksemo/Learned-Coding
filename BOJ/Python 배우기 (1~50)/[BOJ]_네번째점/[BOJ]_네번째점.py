x좌표 = []
y좌표 = []
for _ in range(3):
  x, y = map(int,input().split())
  x좌표.append(x)
  y좌표.append(y)
x좌표.sort()
y좌표.sort()
if x좌표.count(x좌표[0]) == 1:
      네번째점x좌표 = x좌표[0]
else:
      네번째점x좌표 = x좌표[2]
if y좌표.count(y좌표[0]) == 1:
      네번째점y좌표 = y좌표[0]
else:
      네번째점y좌표 = y좌표[2]
print(f'{네번째점x좌표} {네번째점y좌표}')