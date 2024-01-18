import numpy as np
import scipy.io
from load_data import *

# environment and parameters
checkpoint_path_basic = './checkpoint_basic_module'
checkpoint_path_lifelong = './checkpoint_lifelong_module'

SEMANTIC_EMBED = 512
MAX_ITER = 100
max_epoch = 60
batch_size = 32

images, tags, labels = loading_data()
dimTxt = tags.shape[1]
dimLab = labels.shape[1]

DATABASE_SIZE = 21000
TRAINING_SIZE = 21000
QUERY_SIZE = 2100
num_samples = 2000

X, Y, L = split_data(images, tags, labels, QUERY_SIZE, num_seen=19, seed=None)

seen_L = L['seen']
seen_x = X['seen']
seen_y = Y['seen']

unseen_L = L['unseen']
unseen_x = X['unseen']
unseen_y = Y['unseen']

query_L = L['query']
query_x = X['query']
query_y = Y['query']

retrieval_L = L['retrieval']
retrieval_x = X['retrieval']
retrieval_y = Y['retrieval']

num_train = seen_x.shape[0]
numClass = seen_L.shape[1]
num_unseen = unseen_x.shape[0]

Sim = (np.dot(seen_L, seen_L.transpose()) > 0).astype(int)*0.999

Epoch = 1
k_lab_net = 15
k_img_net = 15
k_txt_net = 15

bit = 64
# hyper here

# Learning rate
lr_lab = [np.power(0.1, x) for x in np.arange(2.0, MAX_ITER, 0.5)]
lr_img = [np.power(0.1, x) for x in np.arange(4.5, MAX_ITER, 0.5)]
lr_txt = [np.power(0.1, x) for x in np.arange(3.5, MAX_ITER, 0.5)]
lr_dis = [np.power(0.1, x) for x in np.arange(3.0, MAX_ITER, 0.5)]

