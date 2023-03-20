import torch
from torch.utils.data import Dataset

import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class mi_dataset(Dataset):
    def __init__(self, feature_path, feature_name, label_mode="one-hot", label_list=[]):
        super().__init__()

        with open(feature_path, "rb") as f:
            list_of_dicts, labels = pickle.load(f)
        # The features is list of dicts for each sample
        features = resort_features(list_of_dicts, feature_name)
        self.features = np.squeeze(np.array(features))                                    # n_batch * n_dim
        if self.features.ndim > 2:
            self.features = np.reshape(self.features, (self.features.shape[0], -1))
        print("self.features", self.features.shape)
        self.labels = labels

        if label_mode == "one-hot":
            self.labels = one_hot(self.labels)

        # TODO normalization

    def __len__(self):

        return len(self.features)


    def __getitem__(self, index):

        return self.features[index], self.labels[index]


def one_hot(labels, num_classes=None):
   
    if num_classes == None:
        num_classes = len(np.unique(labels))

    labels_one_hot = []
    for label in labels:
        label_one_hot = np.zeros([num_classes])
        label_one_hot[int(label)] = 1.
        labels_one_hot.append(label_one_hot)

    return np.array(labels_one_hot)


def concat_dataset(features, labels):

    return np.concatenate((features, labels), axis=1)


def resort_features(list_of_dicts, feature_name):

    features = []
    for dict in list_of_dicts:
        feature = dict[feature_name]
        features.append(feature.numpy())

    return features


if __name__ == "__main__":

    featurePath = "D://projects//open_cross_entropy//osr_closed_set_all_you_need-main//features//cifar-10-10_classifier32_0"
    feature_name = "module.avgpool"

    with open(featurePath, "rb") as f:
        # features is list of dicts
        list_of_dicts, labels = pickle.load(f)

    dataset = mi_dataset(feature_path=featurePath, feature_name=feature_name)
    f, l = dataset[5]
    print(f.shape)

    
        
    