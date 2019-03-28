import argparse
import torch
import random
import numpy as np

from model.ssvae import SSVAE
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
from data.loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)


# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)

# load model
model = SSVAE(opt)
model.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load batch
labeled_batch = DataLoader(opt['data_dir'] + '/train_labeled_%.2f.json' % opt['ratio'], opt['batch_size'], opt, vocab, evaluation=False)
batch = labeled_batch[0]

# generate sentence
# idx_list = model.decoder.generate(100,y=21)
# sent = vocab.unmap(idx_list)
# print(' '.join(sent))

# test batch
if opt['cuda']:
    inputs = [b.cuda() for b in batch[:7]]
    labels = batch[7].cuda()
else:
    inputs = [b for b in batch[:7]]
    labels = batch[7]

btn, mu, logvar = model.encoder(inputs, labels)
rec = model.decoder.generate(40, z=btn, y=labels)
sent_list = np.array(rec).transpose()
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
labels = labels.cpu().numpy()
inputs = inputs[0].cpu()
for i, item in enumerate(sent_list):
    print(id2label[labels[i]], '\t', ' '.join(vocab.unmap(item)))
    print(' '.join(vocab.unmap(inputs[i])))
    print(' ')