import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils import constant, torch_utils
from model.rnn import PositionAwareRNN

class SSVAE(object):
    ''' A model for RE based on SSVAE paper'''
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt

        self.classifier = PositionAwareRNN(opt, emb_matrix)
        self.encoder = VAEEncoder(opt, emb_matrix)
        self.decoder = VAEDecoder(opt)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss(reduction='none')

        if opt['cuda']:
            self.classifier.cuda()
            self.encoder.cuda()
            self.decoder.cuda()
            self.criterion.cuda()

        self.parameters = [p for p in self.classifier.parameters() if p.requires_grad] + \
                          [p for p in self.encoder.parameters() if p.requires_grad] + \
                          [p for p in self.decoder.parameters() if p.requires_grad]

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def labeled_update(self, batch):
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        self.classifier.train()
        self.optimizer.zero_grad()
        logits, _ = self.classifier(inputs)
        loss1 = self.criterion(logits, labels) # classification loss

        self.encoder.train()
        self.decoder.train()
        btn, mu, logvar = self.encoder(inputs, labels)
        rec = self.decoder(btn, inputs)
        m, n = inputs[0].shape
        loss2 = self.criterion(rec.view((m*n, -1)), inputs[0].view(m*n)) # rec loss


        loss3 = self._LossKL(mu, logvar).sum()

        # backward
        loss = loss1 + (loss2 + loss3) / self.opt['alpha']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()

        return loss_val

    def unlabeled_update(self, batch):
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
        else:
            inputs = [b for b in batch[:7]]

        self.classifier.train()
        self.optimizer.zero_grad()
        logits, _ = self.classifier(inputs)

        #preds = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        pred = F.softmax(logits)
        sampled_preds, probs = self._SampleOneCategory(pred)

        self.encoder.train()
        self.decoder.train()

        m, n = inputs[0].shape

        btn, mu, logvar = self.encoder(inputs, sampled_preds)
        rec = self.decoder(btn, inputs)

        temp = self.criterion2(rec.view((-1, self.opt['vocab_size'])), inputs[0].view(-1))
        temp = temp.view((m, n)).mean(dim=1)

        loss = torch.matmul(probs, temp)
        loss += torch.matmul(probs, self._LossKL(mu, logvar))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()

        return loss_val

    def predict(self, batch, unsort=True):
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        orig_idx = batch[8]

        self.classifier.eval()
        logits, _ = self.classifier(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx, \
                                                                      predictions, probs)))]
        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        print('This function pretended to be saving :)')
        params = {
            'classifier': self.classifier.state_dict(),
            #'encoder': self.encoder.state_dict(),
            #'decoder': self.decoder.state_dict(),
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def _GaussianMarginalLogDensity(self, mu, logvar, normal_priori=False):
        if normal_priori:
            density = -0.5 * (np.log(2 * np.pi) + (torch.pow(mu, 2) + torch.exp(1e-8 + logvar)))
        else:
            density = -0.5 * (np.log(2 * np.pi) + 1 + logvar)
        return density.sum(-1)

    def _LossKL(self, mu, logvar):
        loss_kl = - self._GaussianMarginalLogDensity(mu, logvar, normal_priori=True) \
                  + self._GaussianMarginalLogDensity(mu, logvar, normal_priori=False)
        return loss_kl

    def _SampleOneCategory(self, preds):
        sampler = Categorical(preds)
        sampled_label = sampler.sample()
        probs = [preds[i][sampled_label[i]] for i in range(self.opt['batch_size'])]
        return sampled_label, torch.Tensor(probs).cuda()

class VAEEncoder(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super(VAEEncoder, self).__init__()

        opt['label_hidden_dim'] = 30
        opt['inter_dim'] = 100
        opt['bottleneck_dim'] = 20

        self.SentEncoder = PositionAwareRNN(opt, emb_matrix)
        self.LabelEmb = nn.Embedding(opt['num_class'], opt['label_hidden_dim'])

        self.l10 = nn.Linear(opt['hidden_dim'] + opt['label_hidden_dim'], opt['inter_dim'])

        self.l20 = nn.Linear(opt['inter_dim'], opt['bottleneck_dim'])
        self.l21 = nn.Linear(opt['inter_dim'], opt['bottleneck_dim'])

        self.emb_matrix = emb_matrix
        self.init_weights()

    def init_weights(self):
        self.LabelEmb.weight.data.uniform_(-1.0, 1.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, y):
        _, h01 = self.SentEncoder(x)
        h02 = self.LabelEmb(y)
        input = torch.cat((h01, h02), dim=1)
        h1 = F.relu(self.l10(input))
        mu = self.l20(h1)
        logvar = self.l21(h1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, opt):
        super(VAEDecoder, self).__init__()

        opt['seq_len'] = 10

        self.linear0 = nn.Linear(opt['bottleneck_dim'], opt['emb_dim'])
        self.rnn = nn.GRU(1, opt['emb_dim'], opt['num_layers'], batch_first=True, dropout=opt['dropout'])
        self.linear = nn.Linear(opt['emb_dim'], opt['vocab_size'])
        # 照着NMT写一个？

    def forward(self, z, x):
        m, n = x[1].shape
        input0 = x[1].reshape(m, n, 1).float()
        input1 = F.relu(self.linear0(z))
        input1 = torch.stack((input1, input1), 0)
        h1, _ = self.rnn(input0, input1)
        return F.softmax(self.linear(h1))