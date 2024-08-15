import random as rand
import matplotlib.pyplot as plt

l = []
for _ in range(10000):
	l.append(rand.random())
for _ in range(5000):
	l.append(rand.random()*1.2)

r = []
for _ in range(15000):
	r.append(rand.random())
	
colors = ["blue" for _ in range(10000)] + ["red" for _ in range(5000)]
tup = list(zip(l,colors))
rand.shuffle(tup)

l = [i[0] for i in tup]
colors = [i[1] for i in tup]

plt.scatter(l,r,color=colors)
plt.show()
