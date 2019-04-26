import torch
import numpy as np
import time

a_gpu= torch.randn(100,1000).cuda()
b_gpu= torch.randn(1000,100).cuda()
a= np.random.randn(100,1000)
b= np.random.randn(1000,100)

start= time.time()
r1= torch.mm(a_gpu, b_gpu)
end= time.time()
print('implement with gpu costs ', end-start, 'seconds')

start= time.time()
r1= a_gpu @ b_gpu
end= time.time()
print('implement with gpu vecotorization costs ', end-start, 'seconds')

start= time.time()
r2= np.dot(a, b)
end= time.time()
print('implement with cpu costs ', end-start, 'seconds')

print(r1.shape, r2.shape)


p1= torch.randn(100,1).cuda()
p2= torch.randn(100,1).cuda()
p3= torch.randn(100,1).cuda()
condition= (p1>0.5) & (p2>0.2) &(p3>0.1)
print(torch.nonzero(condition))