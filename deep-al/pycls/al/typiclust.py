import numpy as np
import pandas as pd
import faiss
from sklearn.cluster import MiniBatchKMeans, KMeans
import pycls.datasets.utils as ds_utils
import torch
from tqdm import tqdm
import random

def get_nn(features, num_neighbors):
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(features)  # add vectors to the index
    distances, indices = gpu_index.search(features, num_neighbors + 1)
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_


class TypiClust:
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, cfg, lSet, uSet, budgetSize, is_scan=False, model=None, dataset=None, dataObj=None): ###
        self.model = model ###
        self.dataset = dataset ###
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.features = None
        self.clusters = None
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.dataObj = dataObj
        self.init_features_and_clusters(is_scan)

    def init_features_and_clusters(self, is_scan):
        num_clusters = min(len(self.lSet) + self.budgetSize, self.MAX_NUM_CLUSTERS)
        print(f'Clustering into {num_clusters} clustering. Scan clustering: {is_scan}')
        if is_scan:
            fname_dict = {'CIFAR10': f'../../scan/results/cifar-10/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'CIFAR100': f'../../scan/results/cifar-100/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'TINYIMAGENET': f'../../scan/results/tiny-imagenet/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          }
            fname = fname_dict[self.ds_name]
            self.features = np.load(fname)
            self.clusters = np.load(fname.replace('features', 'probs')).argmax(axis=-1)
        else:
            self.features = ds_utils.load_features(self.ds_name, self.seed) # 50,000개 데이터 각각의 feature vector
            self.clusters = kmeans(self.features, num_clusters=num_clusters) # 각 feature들의 수도 라벨들?
        print(f'Finished clustering into {num_clusters} clusters.')

    ### Original TypiClust ###
    def select_samples(self):
        # using only labeled+unlabeled indices, without validation set.
        relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        features = self.features[relevant_indices]
        labels = np.copy(self.clusters[relevant_indices])
        existing_indices = np.arange(len(self.lSet))
        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        selected = []
        for i in range(self.budgetSize):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id  
            indices = (labels == cluster).nonzero()[0] ### label이 특정 cluster와 같은 index들을 뽑아줌. ex. 1, 3 번째가 같으면 indices = [1, 3]

            rel_feats = features[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        assert len(selected) == self.budgetSize, 'added a different number of samples'
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        activeSet = relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet
        # ##########################


        ##### Random Sampling #####
    # def select_samples(self):
    #     np.random.seed(self.cfg.RNG_SEED) ### for randome sampling 

    #     # using only labeled+unlabeled indices, without validation set.
    #     relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
    #     labels = np.copy(self.clusters[relevant_indices])
    #     existing_indices = np.arange(len(self.lSet))
    #     # counting cluster sizes and number of labeled samples per cluster
    #     cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    #     cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
    #     clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
    #                                 'neg_cluster_size': -1 * cluster_sizes})
    #     # drop too small clusters
    #     clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
    #     # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
    #     clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
    #     labels[existing_indices] = -1

    #     selected = []
    #     for i in range(self.budgetSize):
    #         cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id  
    #         indices = (labels == cluster).nonzero()[0]
    #         # while len(indices) == 0 :
    #         #     ii = i+1
    #         #     cluster = clusters_df.iloc[ii % len(clusters_df)].cluster_id  
    #         #     indices = (labels == cluster).nonzero()[0]
    #         #     if ii > self.budgetSize*2 :
    #         #         break
    #         indices = indices.tolist()
    #         if len(indices) == 0:
    #             print('there are no indices')
    #             continue
    #         idx = random.sample(indices, 1)[0] # random selection
    #         selected.append(idx)
    #         labels[idx] = -1
        
    #     selected = np.array(selected)
    #     assert len(selected) == self.budgetSize, 'added a different number of samples'
    #     assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
    #     activeSet = relevant_indices[selected]
    #     remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

    #     print(f'Finished the selection of {len(activeSet)} samples.')
    #     print(f'Active set is {activeSet}')
    #     return activeSet, remainSet
        ########################


    ##### Uncertatinty Sampling #####
    # def select_samples(self):
    #     # using only labeled+unlabeled indices, without validation set.
    #     relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)

    #     num_classes = self.cfg.MODEL.NUM_CLASSES
    #     assert self.model.training == False, "Model expected in eval mode whereas currently it is in {}".format(self.model.training)
        
    #     clf = self.model.cuda()
        
    #     u_ranks = []
    #     # if self.cfg.TRAIN.DATASET == "IMAGENET":
    #     #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
    #     #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
    #     #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
    #     # else:
    #     uSetLoader = self.dataObj.getSequentialDataLoader(indexes=relevant_indices, batch_size=int(self.cfg.TRAIN.BATCH_SIZE),data=self.dataset)
    #     uSetLoader.dataset.no_aug = True

    #     n_uLoader = len(uSetLoader)
    #     print("len(uSetLoader): {}".format(n_uLoader))
    #     for i, (x_u, _, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
    #         with torch.no_grad():
    #             x_u = x_u.cuda(0)

    #             temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
    #             temp_u_rank, _ = torch.max(temp_u_rank, dim=1)
    #             temp_u_rank = 1 - temp_u_rank
    #             u_ranks.append(temp_u_rank.detach().cpu().numpy())

    #     u_ranks = np.concatenate(u_ranks, axis=0)
    #     # Now u_ranks has shape: [U_Size x 1]

    #     # index of u_ranks serve as key to refer in u_idx
    #     print(f"u_ranks.shape: {u_ranks.shape}")

    #     labels = np.copy(self.clusters[relevant_indices])
    #     existing_indices = np.arange(len(self.lSet))
    #     # counting cluster sizes and number of labeled samples per cluster
    #     cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    #     cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
    #     clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
    #                                 'neg_cluster_size': -1 * cluster_sizes})
    #     # drop too small clusters
    #     clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
    #     # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
    #     clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
    #     labels[existing_indices] = -1

    #     selected = []
    #     for i in range(self.budgetSize):
    #         cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id  
    #         indices = (labels == cluster).nonzero()[0] ### label이 특정 cluster와 같은 index들을 뽑아줌. ex. 1, 3 번째가 같으면 indices = [1, 3]
    #         cluster_u = u_ranks[indices]
    #         idx = (max(cluster_u) == np.array(u_ranks)).nonzero()[0][0]
    #         selected.append(idx)
    #         labels[idx] = -1

    #     selected = np.array(selected)
    #     assert len(selected) == self.budgetSize, 'added a different number of samples'
    #     assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
    #     activeSet = relevant_indices[selected]
    #     remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

    #     print(f'Finished the selection of {len(activeSet)} samples.')
    #     print(f'Active set is {activeSet}')
    #     return activeSet, remainSet
    ###################