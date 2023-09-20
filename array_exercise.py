test1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test2 = ['a','b','c','d','e','f','g','h','i','j']
test3 = zip(test1, test2)

for i in test3:
    print(i)

print(test1[:3])
print(test1[3:])
print(test1[:-3])
print(test1[-3:])
print(test1[3:7])
print(test1[2:-3])
print(test3)
