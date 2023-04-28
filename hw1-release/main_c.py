import utils
import consts
import models
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

# GPU available?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load model and dataset
model = utils.load_pretrained_cnn(1).to(device)
model.eval()
dataset = utils.TMLDataset(transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=consts.BATCH_SIZE)

# model accuracy
acc_orig = utils.compute_accuracy(model, data_loader, device)
print(f'Model accuracy before flipping: {acc_orig:0.4f}')

# layers whose weights will be flipped
layers = {'conv1': model.conv1,
          'conv2': model.conv2,
          'fc1':   model.fc1,
          'fc2':   model.fc2,
          'fc3':   model.fc3}

# flip bits at random and measure impact on accuracy (via RAD)
RADs_bf_idx = dict([(bf_idx, []) for bf_idx in range(32)]) # will contain a list of RADs for each index of bit flipped
RADs_all = [] # will eventually contain all consts.BF_PER_LAYER*len(layers) RADs
for layer_name in layers:
    layer = layers[layer_name]
    with torch.no_grad():
        W = layer.weight
        W.requires_grad = False
        for _ in range(consts.BF_PER_LAYER):
            # Pick a random weight element to flip
            i = random.randint(0, W.numel() - 1)
            w_orig = W.view(-1)[i].item()

            # Flip a random bit in the weight element
            w_flipped, bf_idx = utils.random_bit_flip(w_orig)
            W.view(-1)[i] = w_flipped

            # Measure RAD
            acc_bf = utils.compute_accuracy(model, data_loader, device)
            rad = (acc_orig - acc_bf) / acc_orig

            # Restore original weight
            W.view(-1)[i] = w_orig

            # Save RAD for this bit-flip index
            RADs_bf_idx[bf_idx].append(rad)

            # Save RAD in the all-RADs list
            RADs_all.append(rad)

# Max and % RAD>10%
RADs_all = np.array(RADs_all)
print(f'Total # weights flipped: {len(RADs_all)}')
print(f'Max RAD: {np.max(RADs_all):0.4f}')
print(f'RAD>15%: {np.sum(RADs_all>0.15)/RADs_all.size:0.4f}')
# boxplots: bit-flip index vs. RAD
plt.figure()
RADs_by_bf_idx = [RADs_bf_idx[bf_idx] for bf_idx in range(32)]
# boxplots: bit-flip index vs. RAD
plt.figure()
plt.boxplot(RADs_by_bf_idx)
plt.xticks(np.arange(1, 33), np.arange(32))
plt.xlabel('Bit-flip index')
plt.ylabel('RAD')
plt.title('Impact of flipping a single bit on model accuracy')
plt.savefig('bf_idx-vs-RAD.jpg')
