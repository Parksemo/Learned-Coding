H, M = map(int,input().split())
설정한알람시각총분 = H*60+M
설정해야하는알림시각총분 = 설정한알람시각총분-45
출력_시 = 설정해야하는알림시각총분//60
if 출력_시 < 0:
    출력_시 += 24
출력_분 = 설정해야하는알림시각총분%60
print(f'{출력_시} {출력_분}')