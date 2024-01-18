from setting import *
from tnet import *
from load_data import *
from ops import *
from utils.calc_hammingranking import *
import os
import time
import scipy.io as sio
from tqdm import tqdm
import torch
import torch.nn as nn

from generatecode import *

from torch.autograd import Variable


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
            
def multi_loss(labels_batchsize, labels_train, hashrepresentations_batchsize, hashrepresentations__train):
        batch_size = labels_batchsize.shape[0]
        num_train = labels_train.shape[0]
        labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
        labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
        hashrepresentations_batchsize = hashrepresentations_batchsize / torch.sqrt(
            torch.sum(torch.pow(hashrepresentations_batchsize, 2), 1)).unsqueeze(1)
        hashrepresentations__train = hashrepresentations__train / torch.sqrt(
            torch.sum(torch.pow(hashrepresentations__train, 2), 1)).unsqueeze(1)
        labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())  # [0,1]
        hashrepresentationsSimilarity = torch.relu(
            torch.matmul(hashrepresentations_batchsize, hashrepresentations__train.t()))  # [0,1]
        MSEloss = torch.sum(torch.pow(hashrepresentationsSimilarity - labelsSimilarity, 2)) / (num_train * batch_size)
        return MSEloss
    
    
class MICH(object):
    def __init__(self):

        # inter_yl项参数
        self.alpha = 10
        # 模态内一致性保持项参数
        self.beta = 1
        # 标签预测项参数
        self.delta = 1000
        #  varphi 量化误差项参数
        self.varphi_l = 1
        self.varphi_v = 1
        self.varphi_t = 1
        # 模态对齐项参数
        self.lamda = 100  # loss_align
        self.hyper_sigma = 0.01
        
        # lifelong模块
        # 可塑性损失项参数
        self.varepsilon = 0.1
        # 量化误差项参数
        self.gamma = 0.1
        # 标签预测项参数
        self.eta = 0.1

        self.seen_L = seen_L
        self.seen_X = seen_x
        self.seen_Y = seen_y

        self.unseen_L = unseen_L
        self.unseen_X = unseen_x
        self.unseen_Y = unseen_y

        self.query_L = query_L
        self.query_X = query_x
        self.query_Y = query_y

        self.retrieval_L = retrieval_L
        self.retrieval_X = retrieval_x
        self.retrieval_Y = retrieval_y

        self.lr_lab = lr_lab
        self.lr_img = lr_img
        self.lr_txt = lr_txt
        self.Sim = Sim

        self.lr_for_lifelong = 0.000001

        self.lnet = LabelNet().cuda()
        self.inet = ImageNet().cuda()
        self.tnet = TextNet().cuda()

        self.checkpoint_dir_basic = checkpoint_path_basic
        self.checkpoint_dir_lifelong = checkpoint_path_lifelong
        self.bit = bit
        self.num_train = num_train
        self.num_unseen = num_unseen
        self.num_samples = num_samples
        self.SEMANTIC_EMBED = SEMANTIC_EMBED

    def train(self):
        # 定义优化器
        self.lnet_opt = torch.optim.Adam(self.lnet.parameters(), lr=self.lr_lab[0])
        self.inet_opt = torch.optim.Adam(self.inet.parameters(), lr=self.lr_img[0])
        self.tnet_opt = torch.optim.Adam(self.tnet.parameters(), lr=self.lr_txt[0])

        var = {}
        var['lr_lab'] = self.lr_lab
        var['lr_img'] = self.lr_img
        var['lr_txt'] = self.lr_txt

        var['batch_size'] = batch_size
        var['F'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['G'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['H'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['FG'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED).astype(np.float32)
        var['B'] = np.sign(self.varphi_v * var['F'] + self.varphi_t * var['G'] + self.varphi_l * var['H'])
        total_time_o = time.time()
        # Iterations for basic module
        for epoch in range(Epoch):
            results = {}
            results['loss_labNet'] = []
            results['loss_imgNet'] = []
            results['loss_txtNet'] = []
            results['Loss_D'] = []
            results['mapl2l'] = []
            results['mapi2i'] = []
            results['mapt2t'] = []

            print('++++++++Start train lab_net++++++++')
            for idx in range(2):
                lr_lab_Up = var['lr_lab'][epoch:]
                lr_lab = lr_lab_Up[idx]
                for train_labNet_k in range(k_lab_net // (idx + 1)):
                    adjust_learning_rate(self.lnet_opt, lr_lab)
                    train_labNet_loss = self.train_lab_net(var)
                    var['B'] = np.sign(self.varphi_v * var['F'] + self.varphi_t * var['G'] + self.varphi_l * var['H'])
                    results['loss_labNet'].append(train_labNet_loss)
                    print('---------------------------------------------------------------')
                    print('...epoch: %3d, loss_labNet: %3.3f' % (epoch, train_labNet_loss))
                    print('---------------------------------------------------------------')
                    if train_labNet_k > 1 and (results['loss_labNet'][-1] - results['loss_labNet'][-2]) >= 0:
                        break

            print('++++++++Starting Train txt_net++++++++')
            for idx in range(3):
                lr_txt_Up = var['lr_txt'][epoch:]
                lr_txt = lr_txt_Up[idx]
                for train_txtNet_k in range(k_txt_net // (idx + 1)):
                    adjust_learning_rate(self.tnet_opt, lr_txt)
                    train_txtNet_loss = self.train_txt_net(var)
                    var['B'] = np.sign(self.varphi_v * var['F'] + self.varphi_t * var['G'] + self.varphi_l * var['H'])
                    if train_txtNet_k % 2 == 0:
                        results['loss_txtNet'].append(train_txtNet_loss)
                        print('---------------------------------------------------------------')
                        print('...epoch: %3d, Loss_txtNet: %s' % (epoch, train_txtNet_loss))
                        print('---------------------------------------------------------------')
                    if train_txtNet_k > 2 and (results['loss_txtNet'][-1] - results['loss_txtNet'][-2]) >= 0:
                        break

            print('++++++++Starting Train img_net++++++++')
            for idx in range(3):
                lr_img_Up = var['lr_img'][epoch:]
                lr_img = lr_img_Up[idx]
                for train_imgNet_k in range(k_img_net // (idx + 1)):
                    adjust_learning_rate(self.inet_opt, lr_img)
                    train_imgNet_loss = self.train_img_net(var)
                    var['B'] = np.sign(self.varphi_v * var['F'] + self.varphi_t * var['G'] + self.varphi_l * var['H'])
                    if train_imgNet_k % 2 == 0:
                        results['loss_imgNet'].append(train_imgNet_loss)
                        print('---------------------------------------------------------------')
                        print('...epoch: %3d, loss_imgNet: %3.3f' % (epoch, train_imgNet_loss))
                        print('---------------------------------------------------------------')
                    if train_imgNet_k > 2 and (results['loss_imgNet'][-1] - results['loss_imgNet'][-2]) >= 0:
                        break

            '''
            evaluation after each epoch
            '''
            with torch.no_grad():
                qBY = self.generate_code(self.query_Y, "text")
                rBY = self.generate_code(self.retrieval_Y, "text")
                qBX = self.generate_code(self.query_X, "image")
                rBX = self.generate_code(self.retrieval_X, "image")

                mapi2t = calc_map(qBX, rBY, self.query_L, self.retrieval_L)
                mapt2i = calc_map(qBY, rBX, self.query_L, self.retrieval_L)
                mapi2i = calc_map(qBX, rBX, self.query_L, self.retrieval_L)
                mapt2t = calc_map(qBY, rBY, self.query_L, self.retrieval_L)

                o_time = time.time() - total_time_o

                condition_dir = './Flickr-MCDLCH-mi-%f-sigma-%f' % (self.lamda, self.hyper_sigma)
                if not os.path.exists(condition_dir):
                    os.mkdir(condition_dir)

                save_dir_name = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
                cur_dir_path = os.path.join(condition_dir, save_dir_name)
                os.mkdir(cur_dir_path)

                scipy.io.savemat(os.path.join(cur_dir_path, 'B_all.mat'), {
                    'BxTest': qBX,
                    'BxTrain': rBX,
                    'ByTest': qBY,
                    'ByTrain': rBY,
                    'LTest': self.query_L,
                    'LTrain': self.retrieval_L
                })

                with open(os.path.join(cur_dir_path, 'map.txt'), 'a') as f:
                    f.write('==================================================\n')
                    f.write('...details of dataset: seen_data: %f, unseen_data: %f\n' % (num_train, num_unseen))
                    f.write('hash bit: %.1f\n' % (self.bit))
                    f.write('...test map of basic module: map(i->t): %3.4f, map(t->i): %3.4f\n' % (mapi2t, mapt2i))
                    f.write('...test map of basic module: map(t->t): %3.4f, map(i->i): %3.4f\n' % (mapt2t, mapi2i))
                    f.write('...training basic phase finish time : %3.3f\n' % o_time)
                    f.write('==================================================\n')

                '''
                save checkpoint
                '''
                state = {
                    'lnet': self.lnet.state_dict(),
                    'inet': self.inet.state_dict(),
                    'tnet': self.tnet.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, os.path.join(cur_dir_path, checkpoint_path_basic))
                
                #   基础模型训练结束
                print("generate basic codes")
                old_F = self.generate_code(self.seen_X, "image")
                old_G = self.generate_code(self.seen_Y, "text")

                torch.save(old_F, os.path.join(cur_dir_path, 'old_F.t'))
                torch.save(old_G, os.path.join(cur_dir_path, 'old_G.t'))

            print("start for lifelong module")
            # 重新定义优化器和学习率衰减策略
            self.optimizer_img2 = torch.optim.Adam(self.inet.parameters(), lr=self.lr_for_lifelong)
            self.optimizer_txt2 = torch.optim.Adam(self.tnet.parameters(), lr=self.lr_for_lifelong)

            self.lr_scheduler_img2 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_img2, 0.95)
            self.lr_scheduler_txt2 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_txt2, 0.95)

            Ux_new = torch.zeros(self.num_samples, self.bit).cuda()
            Uy_new = torch.zeros(self.num_samples, self.bit).cuda()

            new_F_buffer = torch.randn(self.num_samples, self.bit).cuda()
            new_G_buffer = torch.randn(self.num_samples, self.bit).cuda()

            new_F = torch.randn(self.num_unseen, self.bit).sign().cuda()
            new_G = torch.randn(self.num_unseen, self.bit).sign().cuda()

            old_F = torch.from_numpy(old_F).cuda()
            old_G = torch.from_numpy(old_G).cuda()

            F2 = torch.cat((old_F, new_F), dim=0)
            G2 = torch.cat((old_G, new_G), dim=0)

            device = True

            train_L, train_x, train_y, train_index, unseen_in_unseen_index, unseen_in_sample_index = sample_data(
                    retrieval_x, retrieval_y, retrieval_L, num_train, num_samples)
            train_L = torch.from_numpy(train_L)
            train_x = torch.from_numpy(train_x)
            train_y = torch.from_numpy(train_y)

            max_mapi2t = max_mapt2i = 0.
            total_time_i = time.time()

            # Iterations for lifelong module
            for epoch in range(max_epoch):
                seen_L = torch.from_numpy(self.seen_L)
                unseen_L = torch.from_numpy(self.unseen_L)
                old_S = multi_similar(train_L, seen_L, device)
                new_S = multi_similar(train_L, unseen_L, device)
                S = torch.cat((old_S, new_S), dim=1)
                    # train image net
                print("...start to train image net")
                for i in tqdm(range(num_samples // batch_size)):
                    index = np.random.permutation(num_samples)
                    ind = index[0: batch_size]
                    # print("ind shape", ind.shape)
                    unupdated_ind = np.setdiff1d(range(num_samples), ind)

                    sample_L = Variable(train_L[ind, :])
                    image = train_x[ind].type(torch.float)
                    image = Variable(image)
                    if device:
                        image = image.cuda()
                        sample_L = sample_L.cuda()
       
                    _, cur_f, lab_I, _, _, _ = self.inet(image) # inet的输入应该为np.array格式，cur_f是tensor格式
                    new_F_buffer[ind, :] = cur_f.detach()
                    Ux_new = Variable(torch.tanh(new_F_buffer))

                    hashloss_x1 = ((self.bit * old_S[ind, :] - cur_f @ old_F.t()) ** 2).sum()
                    hashloss_x2 = ((self.bit * new_S[ind, :] - cur_f @ new_F.t()) ** 2).sum()
                    quantization_x2 = torch.sum(torch.pow(F2[ind, :] - cur_f, 2))
                    label_loss_x = nn.functional.mse_loss(sample_L, lab_I, reduction='sum')
                    loss_x2 = hashloss_x1 + self.varepsilon * hashloss_x2 + self.gamma * quantization_x2 + self.eta * label_loss_x
                    loss_x2 /= (batch_size * F2.shape[0])

                    self.optimizer_img2.zero_grad()
                    loss_x2.backward()
                    self.optimizer_img2.step()
                # update F
                expand_Ux_new = torch.zeros(num_unseen, self.bit).cuda()
                expand_Ux_new[unseen_in_unseen_index, :] = Ux_new[unseen_in_sample_index, :]
                
                new_F = solve_dcc(new_F, Ux_new, expand_Ux_new, S[:, num_train:], self.bit, self.gamma)
                F2 = torch.cat((old_F, new_F), dim=0).cuda()
                # train txt net
                print("...start to train txt net")
                for i in tqdm(range(num_samples // batch_size)):
                    index = np.random.permutation(num_samples)
                    ind = index[0: batch_size]
                    unupdated_ind = np.setdiff1d(range(num_samples), ind)

                    sample_L = Variable(train_L[ind, :])
                    text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
                    text = Variable(text)
                    if device:
                        text = text.cuda()
                        sample_L = sample_L.cuda()

                    _, cur_g, lab_T  = self.tnet(text)
                    new_G_buffer[ind, :] = cur_g.detach()
                    Uy_new = Variable(torch.tanh(new_G_buffer))

                    hashloss_y1 = ((self.bit * old_S[ind, :] - cur_g @ old_G.t()) ** 2).sum()
                    hashloss_y2 = ((self.bit * new_S[ind, :] - cur_g @ new_G.t()) ** 2).sum()
                    quantization_y2 = torch.sum(torch.pow(G2[ind, :] - cur_g, 2))
                    label_loss_y = nn.functional.mse_loss(sample_L, lab_T, reduction='sum')
                    loss_y2 = hashloss_y1 + self.varepsilon * hashloss_y2 + self.gamma * quantization_y2 + self.eta * label_loss_y
                    loss_y2 = loss_y2 / (batch_size * G2.shape[0])

                    self.optimizer_txt2.zero_grad()
                    loss_y2.backward()
                    self.optimizer_txt2.step()

                # update G
                expand_Uy_new = torch.zeros(num_unseen, self.bit).cuda()
                expand_Uy_new[unseen_in_unseen_index, :] = Uy_new[unseen_in_sample_index, :]
                new_G = solve_dcc(new_G, Uy_new, expand_Uy_new, S[:, num_train:], self.bit, self.gamma)
                G2 = torch.cat((old_G, new_G), dim=0).cuda()

                update_B_new = torch.sign(F2 + G2)

                # calculate total loss
                loss2 = calc_increment_loss(Ux_new, Uy_new, update_B_new, S, self.bit, train_index, self.gamma, self.eta)

                print('...epoch: %3d, loss: %3.3f, lr: %.10f' % (epoch + 1, loss2.data, self.optimizer_img2.param_groups[0]['lr']))

                database_L = torch.cat((seen_L, unseen_L), dim=0)
                query_L = torch.from_numpy(self.query_L).cuda()
                database_L = database_L.cuda()
                query_X = torch.from_numpy(self.query_X)
                query_Y = torch.from_numpy(self.query_Y)
                qBX2 = generate_image_code(self.inet, query_X, self.bit, batch_size, device)
                qBY2 = generate_text_code(self.tnet, query_Y, self.bit, batch_size, device)

                mapi2t = mean_average_precision(qBX2, update_B_new, query_L, database_L, device, topk=None)
                mapt2i = mean_average_precision(qBY2, update_B_new, query_L, database_L, device, topk=None)
                
                print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))


                if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                    max_mapi2t = mapi2t
                    max_mapt2i = mapt2i

                self.lr_scheduler_img2.step()
                self.lr_scheduler_txt2.step()
                i_time = time.time() - total_time_i

            print('...training basic phase finish and time : %3.2f' % o_time)
            print('...training incremental phase finish and time : %3.2f' % i_time)
            print(' DLCH max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, MAP(i->t)@500: %3.4f, MAP(t->i)@500: %3.4f' % (max_mapi2t, max_mapt2i, mapi2t500, mapt2i500))

            with open(os.path.join(cur_dir_path, 'map.txt'), 'a') as f:
                f.write('==================================================\n')
                f.write('...test map of lifelong module: map(i->t): %3.4f, map(t->i): %3.4f\n' % (max_mapi2t, max_mapt2i))
                f.write('...training incremental phase finish time : %3.4f\n' % i_time)
                f.write('==================================================\n')
            
            '''
            save checkpoint
            '''
            state = {
                'lnet': self.lnet.state_dict(),
                'inet': self.inet.state_dict(),
                'tnet': self.tnet.state_dict(),
                'epoch': epoch
            }
            torch.save(state, os.path.join(cur_dir_path, checkpoint_path_lifelong))

    def train_lab_net(self, var):
        print('update label_net for basic module')
        F = var['F']
        G = var['G']
        H = var['H']
        B = var['B']
        loss_total = 0.0
        num_train = self.seen_L.shape[0]
        for iter in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = self.seen_L[ind, :]
            label = self.seen_L[ind, :].astype(np.float32)
            label = label.reshape([label.shape[0], 1, 1, label.shape[1]])
            S_cuda = multi_similar(torch.from_numpy(self.seen_L).cuda(), torch.from_numpy(sample_L).cuda(), device=True)
            hsh_L, lab_L = self.lnet(torch.from_numpy(label).cuda())

            H[ind, :] = hsh_L.detach().cpu().numpy()
            # S_cuda = torch.from_numpy(S).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_FL = 1.0 / 2 * torch.from_numpy(F).cuda().mm(hsh_L.transpose(1, 0))
            Loss_pair_Hsh_FL = nn.functional.mse_loss(S_cuda.mul(theta_FL), nn.functional.softplus(theta_FL),
                                                      reduction='sum')
            theta_GL = 1.0 / 2 * torch.from_numpy(G).cuda().mm(hsh_L.transpose(1, 0))
            Loss_pair_Hsh_GL = nn.functional.mse_loss(S_cuda.mul(theta_GL), nn.functional.softplus(theta_GL),
                                                      reduction='sum')
            Loss_quant_L = nn.functional.mse_loss(B_cuda, hsh_L, reduction='sum')
            Loss_label_L = nn.functional.mse_loss(torch.from_numpy(self.seen_L[ind, :]).cuda(), lab_L, reduction='sum')
            loss_l = (self.alpha * Loss_pair_Hsh_FL + Loss_pair_Hsh_GL) + self.varphi_l * Loss_quant_L + self.delta * Loss_label_L

            loss_total += float(loss_l.detach().cpu().numpy())

            self.lnet_opt.zero_grad()
            loss_l.backward()
            self.lnet_opt.step()
        return loss_total

    def train_img_net(self, var):
        print('update image_net for basic module')
        F = var['F']
        H = var['H']
        FG = var['FG']
        B = var['B']
        loss_total = 0.0
        num_train = self.seen_X.shape[0]
        for iter in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = seen_L[ind, :]
            image = self.seen_X[ind, :, :, :].astype(np.float32)
            S = calc_neighbor(self.seen_L, sample_L)
            S_cuda = torch.from_numpy(S).cuda()

            fea_I, hsh_I, lab_I, fea_T_pred, mu_I, log_sigma_I = self.inet(torch.from_numpy(image).cuda())
            F[ind, :] = hsh_I.detach().cpu().numpy()
            fea_T_real = torch.from_numpy(FG[ind, :]).cuda()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_MH = 1.0 / 2 * torch.from_numpy(H).cuda().mm(hsh_I.transpose(1, 0))
            Loss_pair_Hsh_MH = nn.functional.mse_loss(S_cuda.mul(theta_MH).float(), nn.functional.softplus(theta_MH).float(),
                                                      reduction='sum')
            theta_MM = 1.0 / 2 * torch.from_numpy(F).cuda().mm(hsh_I.transpose(1, 0))
            Loss_pair_Hsh_MM = nn.functional.mse_loss(S_cuda.mul(theta_MM).float(), nn.functional.softplus(theta_MM).float(),
                                                      reduction='sum')
            Loss_quant_I = nn.functional.mse_loss(B_cuda, hsh_I, reduction='sum')
            Loss_label_I = nn.functional.mse_loss(torch.from_numpy(self.seen_L[ind, :]).cuda(), lab_I, reduction='sum')
            Loss_prior_kl = torch.sum(mu_I.pow(2).add_(log_sigma_I.exp()).mul_(-1).add_(1).add_(log_sigma_I)).mul_(-0.5)
            Loss_cross_hash_MI = nn.functional.binary_cross_entropy_with_logits(fea_T_pred, torch.sigmoid(fea_T_real), reduction='sum') \
                                 + self.hyper_sigma * Loss_prior_kl

            loss_i = (Loss_pair_Hsh_MH + self.beta * Loss_pair_Hsh_MM) \
                + self.varphi_v * Loss_quant_I \
                + self.delta * Loss_label_I \
                + self.lamda * Loss_cross_hash_MI
            loss_total += float(loss_i.detach().cpu().numpy())

            self.inet_opt.zero_grad()
            loss_i.backward()
            self.inet_opt.step()
        return loss_total

    def train_txt_net(self, var):
        print('update text_net for basic module')
        G = var['G']
        H = var['H']
        FG = var['FG']
        B = var['B']
        loss_total = 0.0
        num_train = self.seen_Y.shape[0]
        for iter in tqdm(range(num_train // batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = seen_L[ind, :]
            text = self.seen_Y[ind, :].astype(np.float32)
            text = text.reshape([text.shape[0], 1, text.shape[1], 1])
            S = calc_neighbor(self.seen_L, sample_L)
            S_cuda = torch.from_numpy(S).cuda()

            fea_T, hsh_T, lab_T = self.tnet(torch.from_numpy(text).cuda())
            G[ind, :] = hsh_T.detach().cpu().numpy()
            FG[ind, :] = fea_T.detach().cpu().numpy()
            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            theta_MH = 1.0 / 2 * torch.from_numpy(H).cuda().mm(hsh_T.transpose(1, 0))
            Loss_pair_Hsh_MH = nn.functional.mse_loss(S_cuda.mul(theta_MH).float(), nn.functional.softplus(theta_MH).float(),
                                                      reduction='sum')
            theta_MM = 1.0 / 2 * torch.from_numpy(G).cuda().mm(hsh_T.transpose(1, 0))
            Loss_pair_Hsh_MM = nn.functional.mse_loss(S_cuda.mul(theta_MM).float(), nn.functional.softplus(theta_MM).float(),
                                                      reduction='sum')
            Loss_quant_T = nn.functional.mse_loss(B_cuda, hsh_T, reduction='sum')
            Loss_label_T = nn.functional.mse_loss(torch.from_numpy(self.seen_L[ind, :]).cuda(), lab_T, reduction='sum')
            loss_t = (self.alpha * Loss_pair_Hsh_MH + self.beta * Loss_pair_Hsh_MM) \
                + self.varphi_t * Loss_quant_T \
                + self.delta * Loss_label_T
            loss_total += float(loss_t.detach().cpu().numpy())
            
            self.tnet_opt.zero_grad()
            loss_t.backward()
            self.tnet_opt.step()
        return loss_total


    def multi_similar(self, labels_batchsize, labels_train, device):
        labels_batchsize = labels_batchsize / torch.sqrt(torch.sum(torch.pow(labels_batchsize, 2), 1)).unsqueeze(1)
        labels_train = labels_train / torch.sqrt(torch.sum(torch.pow(labels_train, 2), 1)).unsqueeze(1)
        if device:
            labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t()).type(torch.cuda.FloatTensor)
        else:
            labelsSimilarity = torch.matmul(labels_batchsize, labels_train.t())
        return labelsSimilarity

    def calc_isfrom_acc(self, train_isfrom_, Train_ISFROM):
        erro = Train_ISFROM.shape[0] - np.sum(
            np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype(int))
        acc = np.divide(np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype('float32')),
                        Train_ISFROM.shape[0])
        return erro, acc

    def generate_code(self, modal, generate):
        batch_size = 128
        if generate == "label":
            num_data = modal.shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(range(num_data // batch_size + 1)):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                label = modal[ind, :].astype(np.float32)
                Fea_L, Hsh_L, Lab_L = self.lnet(torch.from_numpy(label).cuda())
                B[ind, :] = Hsh_L.detach().cpu().numpy()
        elif generate == "image":
            num_data = len(modal)
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(range(num_data // batch_size + 1)):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                image = modal[ind, :, :, :].astype(np.float32)
                Fea_I, Hsh_I, Lab_I, fea_T_pred, eta_I, log_sigma_I = self.inet(torch.from_numpy(image).cuda())
                B[ind, :] = Hsh_I.detach().cpu().numpy()
        else:
            num_data = modal.shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in tqdm(range(num_data // batch_size + 1)):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                text = modal[ind, :].astype(np.float32)
                Fea_T, Hsh_T, Lab_T = self.tnet(torch.from_numpy(text).cuda())
                B[ind, :] = Hsh_T.detach().cpu().numpy()
        B = np.sign(B)
        return B

    def calc_labnet_loss(self, H, F, G, B, Sim):
        theta_fh = np.matmul(F, np.transpose(H)) / 2
        term_fh = np.sum(nn.functional.softplus(torch.from_numpy(theta_fh)).numpy() - Sim * theta_fh)
        theta_gh = np.matmul(G, np.transpose(H)) / 2
        term_gh = np.sum(nn.functional.softplus(torch.from_numpy(theta_gh)).numpy() - Sim * theta_gh)
        term_quant = np.sum(np.power(B - H, 2))
        loss = (term_fh + term_gh) + self.eta * term_quant
        print('pairwise_hash_FH:', term_fh)
        print('pairwise_hash_GH:', term_gh)
        print('quant loss:', term_quant)
        return loss

    def calc_loss(self, alpha, M, H, B, Sim):
        theta_mh = np.matmul(M, np.transpose(H)) / 2
        term_mh = np.sum(nn.functional.softplus(torch.from_numpy(theta_mh)).numpy() - Sim * theta_mh)
        theta_mm = np.matmul(M, np.transpose(M)) / 2
        term_mm = np.sum(nn.functional.softplus(torch.from_numpy(theta_mm)).numpy() - Sim * theta_mm)
        term_quant = np.sum(np.power(B - M, 2))  # 量化损失
        loss = (term_mh + term_mm) + alpha * term_quant
        print('pairwise_hash_MH:', term_mh)
        print('pairwise_hash_MM:', term_mm)
        print('quant loss:', term_quant)
        return loss
    
