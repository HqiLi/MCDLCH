import torch
from tqdm import tqdm
import numpy as np
from utils.calc_hammingranking import mean_average_precision


def generate_image_code(img_model, X, bit, batch_size, device):
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if device:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1), position=0):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if device:
            image = image.cuda()
        _, cur_f, _, _, _, _ = img_model(image)
        B[ind, :] = cur_f.detach()
    B = torch.sign(B)
    return B


def generate_text_code(txt_model, Y, bit, batch_size, device):
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if device:
        B = B.cuda()
    for i in tqdm(range(num_data // batch_size + 1), position=0):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if device:
            text = text.cuda()
        _, cur_g, _ = txt_model(text)
        B[ind, :] = cur_g.detach()
    B = torch.sign(B)
    return B


def valid(img_model, txt_model, query_x, rBX, query_y, rBY, query_L, retrieval_L, code_length,
          batch_size, device):
    qBX = generate_image_code(img_model, query_x, code_length, batch_size, device)
    qBY = generate_text_code(txt_model, query_y, code_length, batch_size, device)

    mapi2t = mean_average_precision(qBX, rBY, query_L, retrieval_L, device)
    mapt2i = mean_average_precision(qBY, rBX, query_L, retrieval_L, device)
    return mapi2t, mapt2i