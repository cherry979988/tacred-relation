import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, default='dataset/tacred')
parser.add_argument('--out_dir', type=str, default='dataset/tacred')
parser.add_argument('--ratio', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()
opt = vars(args)

random.seed(args.seed)

filename = opt['in_dir'] + '/train.json'
json_data = json.load(open(filename))

labeled_filename = opt['out_dir'] + '/train_labeled_%.2f.json' % opt['ratio']
unlabeled_filename = opt['out_dir'] + '/train_unlabeled_%.2f.json' % opt['ratio']

random.shuffle(json_data)
n_all = len(json_data)
n_labeled = int(n_all * opt['ratio'])

labeled_data = json_data[:n_labeled]
unlabeled_data = json_data[n_labeled:]

with open(labeled_filename, 'w') as outfile:
    json.dump(labeled_data, outfile)
with open(unlabeled_filename, 'w') as outfile:
    json.dump(unlabeled_data, outfile)

print('Ratio: ', opt['ratio'])
print('Total Instances: %d, Labeled Instances: %d' % (n_all, n_labeled))