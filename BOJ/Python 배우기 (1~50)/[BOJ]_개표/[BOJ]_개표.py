V = int(input())
투표 = input()
if len(투표) == V:
    if 투표.count('A') < 투표.count('B'):
        print('B')
    elif 투표.count('A') > 투표.count('B'):
        print('A')
    elif 투표.count('A') == 투표.count('B'):
        print('Tie')