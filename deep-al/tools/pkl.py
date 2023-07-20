import pickle
import numpy as np

data = np.load("/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/simclr_budget1000_cycle1/episode_0/lSet.npy", allow_pickle=True)
labeled_dataset = np.load("/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/simclr_budget1000_cycle1_2/episode_0/lSet.npy", allow_pickle=True) # labeled dataset path


with open('/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/resnet18_trainset_feature/feature.pkl', 'rb') as f :
    feature = pickle.load(f)

with open('/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/resnet18_trainset_feature/labels.pkl', 'rb') as f :
    labels = pickle.load(f)

with open('/home/ubuntu/junbeom/repo/TypiClust/output/CIFAR10/resnet18/resnet18_trainset_feature/image_index.pkl', 'rb') as f :
    image_index = pickle.load(f)


print(len(data))
print('batch size :', len(feature))
print('feature size : ', len(feature[0]))
print('labels size : ', len(labels))
print('image_index size : ', len(image_index))

print(labeled_dataset)
# for i in range(len(image_index)) :
#     if image_index[i] in labeled_dataset:
#         print(image_index[i], labels[i])