
import random
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
from joblib import dump, load
import pickle
from model import Model

SAVE_MODEL_PATH = "checkpoint/best_accuracy.pth"

def export_model(model, model_export_name):
    dump(model, model_export_name)

def train(opt):
    device = torch.device("cuda" if opt.use_gpu and torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = Model().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data preparation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size)
    testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size)

    X_train = trainset.data.numpy().reshape(-1, 784)
    y_train = trainset.targets.numpy()

    X_test = testset.data.numpy().reshape(-1, 784)
    y_test = testset.targets.numpy()

    print(X_train.shape, y_train.shape)
    # # print(X_train.shape)
    # # #KNN CLASSIFIER
    # from sklearn.neighbors import KNeighborsClassifier
    # KNN_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    # KNN_classifier.fit(X_train, y_train)

    # export_model(KNN_classifier, 'knn')

    #SVM CLASSIFIER
    from sklearn import svm
    clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))

    svm_classifier = svm.SVC(gamma=0.001, probability=True, n_jobs=-1)
    svm_classifier.fit(X_train, y_train)
    export_model(svm_classifier, 'svm')

    # from sklearn.ensemble import RandomForestClassifier
    # RF_classifier = RandomForestClassifier(max_depth=2, random_state=0)
    # RF_classifier.fit(X_train, y_train)
    # export_model(RF_classifier, 'rf')


    # print(trainset.data.numpy()[0])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_seed", type=int, default=1111, help="random seed setting")
    parser.add_argument("--batch_size", type=int, default=50, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=30, help="number of epochs to train")
    parser.add_argument("--valid_interval", type=int, default=1, help="interval between each validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--use_gpu", action="store_true", help="use gpu if availabe")
    opt = parser.parse_args()
    print("args", opt)

    # set seed
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

    # training
    train(opt)
