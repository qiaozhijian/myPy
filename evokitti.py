import numpy as np
a=np.arange(9).reshape(3,3)
a=np.expand_dims(a,axis=0).repeat(axis=0,repeats=4)
print(a.shape)
# print(a)
b=np.arange(3).reshape(3,1)
b=np.expand_dims(b,axis=0).repeat(axis=0,repeats=4)
print(b.shape)
# print(b)
# b=np.arange(3).reshape(3,1)
# b=np.repeat(b,repeats=4,axis=0)
# c=np.hstack((a,b))
# print(a)
c=[]
if len(b.shape)>2:
    for batch in range(b.shape[0]):
        c.append(np.hstack((a[batch],b[batch])))
else:
    c.append(np.hstack((a, b)))
print(c)
c=np.asarray(c).reshape(-1,3,4)
# print(b)
print(c)
print(c.shape)