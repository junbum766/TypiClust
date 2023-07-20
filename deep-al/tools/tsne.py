from sklearn.manifold import TSNE
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os, random

# random seed 고정 
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

SEED = 555
set_seeds(SEED)

with open('/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/resnet18_trainset_feature_2/feature.pkl', 'rb') as f : # resnet18 train dataset feature
    feature_data = pkl.load(f)
with open('/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/resnet18_trainset_feature_2/labels.pkl', 'rb') as f : # labels of features
    labels = pkl.load(f)
with open('/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/resnet18_trainset_feature_2/image_index.pkl', 'rb') as f : # image_index of features
    image_index = pkl.load(f)

lSet_path = "/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/pt4al_budget1000_cycle1" #####

labeled_dataset = np.load(lSet_path+"/lSet_5000.npy", allow_pickle=True) # labeled dataset path


dim = 2
model = TSNE(dim)
tsne_result = model.fit_transform(feature_data)

# print(tsne_result)
print('feature dimension : ' ,len(feature_data), len(feature_data[0])) 
print('t_SNE dimension : ', len(tsne_result), len(tsne_result[0]))
print('The number of labeled dataset : ', len(labeled_dataset))

class_num = 10

colors = plt.cm.rainbow(np.linspace(0, 1, class_num))
    
for idx, feature in enumerate(tsne_result) :
    total = len(tsne_result)
    print(f'{idx+1}/{total}', end='\r')
    if idx % 5 == 0: # 너무 오래 걸려서 10의 배수 번째만 그려보자..
        plt.scatter(feature[0], feature[1], c='grey', s=5, alpha=0.05)
    # plt.scatter(feature[0], feature[1], c=colors[labels[idx]].reshape(1, -1), s=5, alpha=0.1)

    if image_index[idx] in labeled_dataset :
        plt.scatter(feature[0], feature[1], c=colors[labels[idx]].reshape(1, -1), s=5, alpha=1)
        print(idx, 'labeled')

print('\n\n...saving... ')
plt.savefig(lSet_path+'/feature_5cycle.png',dpi=300) # save feature t-SNE graph
print('...complete!')