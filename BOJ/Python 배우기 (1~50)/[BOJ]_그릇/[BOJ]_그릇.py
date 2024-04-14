그릇=input()
높이=10
for i in range(1,len(그릇)):
    if 그릇[i] == 그릇[i-1]:
        높이+=5
    elif 그릇[i] != 그릇[i-1]:
        높이+=10
print(높이)