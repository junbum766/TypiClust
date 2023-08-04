import torch
import numpy as np

data = np.load("/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/pt4al_budget50_Atypicality_notUseMainLoss/lSet.npy", allow_pickle=True)

# print(len(data))
# # data.sort()
print(len(data))
# print(data[990:])
data1 = data[:50]
data6 = data[250:]
# print(data6)
# data10 = data[9000:10000]
np.save("/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/pt4al_budget50_Atypicality_notUseMainLoss/lSet_cycle1.npy", data1)
np.save("/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/pt4al_budget50_Atypicality_notUseMainLoss/lSet_cycle6.npy", data6)
# np.save("/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/pt4al_budget1000_balancedClass_to_c1_allbatch/lSet_activeSet_cycle10.npy", data10)


# a = [1]
# a.append(2)
# a.append(3)
# print(a)

# a = [1, 2, 3, 4, 5, 1]
# a = np.array(a)

# print(len(a))

# b = (a == 6).nonzero()[0]

# print(b)
# a = [1, 2, 3]
# cluster_u = [5, 3, 2, 9]
# u_ranks = [5, 1, 2, 3, 2, 4, 5, 10, 11, 1,9, 2]
# u_ranks = np.array(u_ranks)


# print(u_ranks)
# labeld_idx = []
# for l in data:
#     labeld_idx.append(int(l.split('/')[-1].split('.')[0]))

# np.save('lSet', labeld_idx)

# a = [0, 2, 3, 1]
# b = [1, 2, 3, 4]

# b = np.array(b)

# print(b[a])

# a = torch.zeros(10)
# for i in a:
#     print(i+1)
# a[4] += 1

# print(a)