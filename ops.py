from setting import *
import torch

def calc_neighbor(label_1, label_2):
    Sim = (np.dot(label_1, label_2.transpose()) > 0).astype(np.float32)*0.999
    return Sim


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def multi_similar(labels_batchsize, labels_train, device):
    labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
    labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
    if device:
        labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t()).type(torch.cuda.FloatTensor)
    else:
        labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())
    return labelsSimilarity


def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    hash_loss = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    quantization_loss = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    correlation_loss = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = (hash_loss + gamma * quantization_loss + eta * correlation_loss) / (F.shape[0] * B.shape[0])
    return loss


def solve_dcc(B, Hq, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    P = code_length * S.t() @ Hq + gamma * expand_U

    for bit in range(code_length):
        p = P[:, bit]
        u = Hq[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((Hq[:, :bit], Hq[:, bit+1:]), dim=1)

        B[:, bit] = (p.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_increment_loss(Ux, Uy, B, S, code_length, omega, gamma, mu):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - Ux @ B.t()) ** 2).sum() + ((code_length * S - Uy @ B.t()) ** 2).sum()
    quantization_loss = ((Ux - B[omega, :]) ** 2).sum() + ((Uy - B[omega, :]) ** 2).sum()
    correlation_loss = (Ux @ torch.ones(Ux.shape[1], 1, device=Ux.device)).sum() + (Uy @ torch.ones(Uy.shape[1], 1, device=Uy.device)).sum()
    loss = (hash_loss + gamma * quantization_loss + mu * correlation_loss) / (B.shape[0] * Ux.shape[0])

    return loss