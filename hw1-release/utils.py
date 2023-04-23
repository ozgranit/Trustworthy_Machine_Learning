import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id<0 or cnn_id>2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model

class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """
    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            num_correct += (predicted == labels).sum().item()
            num_total += labels.size(0)
    accuracy = num_correct / num_total

    return accuracy


def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    model = attack.model
    model.eval()
    adv_images = []
    true_labels = []
    target_labels = []
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if targeted:
            t = (y + torch.randint(1, n_classes, size=y.shape)) % n_classes
        else:
            t = y
        adv_x = attack.execute(x, t, targeted)
        assert adv_x.min() >= 0, 'Adversarial images lie outside of the lower bound'
        assert adv_x.max() <= 1, 'Adversarial images lie outside of the upper bound'
        assert torch.all(torch.abs(adv_x - x) <= attack.eps + 1e-7), 'Adversarial images lie outside of the epsilon-ball'
        adv_images.append(adv_x)
        true_labels.append(y)
        target_labels.append(t)
    adv_images = torch.cat(adv_images, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    target_labels = torch.cat(target_labels, dim=0)

    return adv_images, (target_labels if targeted else true_labels)


def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    model = attack.model
    model.eval()
    adv_images = []
    true_labels = []
    target_labels = []
    n_queries_list = []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if targeted:
            t = (y + torch.randint(1, n_classes, size=y.shape)) % n_classes
        else:
            t = y
        adv_x, num_queries = attack.execute(x, t, targeted)
        assert adv_x.min() >= 0, 'Adversarial images lie outside of the lower bound'
        assert adv_x.max() <= 1, 'Adversarial images lie outside of the upper bound'
        assert torch.all(
            torch.abs(adv_x - x) <= attack.eps + 1e-7), 'Adversarial images lie outside of the epsilon-ball'
        adv_images.append(adv_x)
        true_labels.append(y)
        target_labels.append(t)
        n_queries_list.append(num_queries)
    adv_images = torch.cat(adv_images, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    target_labels = torch.cat(target_labels, dim=0)
    n_queries_list = torch.cat(n_queries_list, dim=0)

    return adv_images, (target_labels if targeted else true_labels), n_queries_list


def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    from torch.utils.data import TensorDataset, DataLoader
    n_samples = len(x_adv)
    n_success = 0

    # Create data loader
    dataset = TensorDataset(x_adv, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluate model on adversarial examples
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        if targeted:
            n_success += (torch.argmax(logits, dim=1) == y_batch).sum().item()
        else:
            n_success += (torch.argmax(logits, dim=1) != y_batch).sum().item()

    success_rate = n_success / n_samples
    return success_rate


def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    pass # FILL ME

def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    pass # FILL ME

def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    pass # FILL ME
