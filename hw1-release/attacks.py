import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        adv_x = x.clone().detach()
        if self.rand_init:
            # Starting at a uniformly random point
            adv_x = adv_x + torch.empty_like(adv_x).uniform_(-self.eps, self.eps)
            adv_x = torch.clamp(adv_x, min=0, max=1).detach()

        for i in range(self.n):
            adv_x.requires_grad = True
            outputs = self.model(adv_x)
            if self.early_stop:
                if not targeted and not (torch.argmax(outputs, dim=1) == y).any():
                    break
                elif targeted and (torch.argmax(outputs, dim=1) == y).all():
                    break

            # Calculate loss
            loss = self.loss_func(outputs, y)
            if targeted:
                loss = -loss
            # Update adversarial images
            grad = torch.autograd.grad(loss.sum(), adv_x, retain_graph=False, create_graph=False)[0]

            adv_x = adv_x.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_x - x, min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

        assert torch.all(adv_x >= 0.) and torch.all(adv_x <= 1.)
        assert torch.all(torch.abs(adv_x - x) <= self.eps + 1e-7)

        return adv_x


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255., momentum=0.,
                 k=200, sigma=1/255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    import torch

    def NES_gradient_estimate(self, x, y):
        """
        Estimates the gradient of the model at x with respect to y using the
        Natural Evolutionary Strategies (NES) algorithm.

        Inputs:
            - model: PyTorch model to attack
            - x: input image tensor of shape (batch_size, channels, height, width)
            - y: tensor of true class labels of shape (batch_size,)
            - sigma: the standard deviation of the Gaussian noise used for querying
            - n: number of samples used to estimate the gradient

        Returns:
            - grad: gradient estimate tensor of shape (batch_size, channels, height, width)
        """
        batch_size = x.shape[0]
        N = x.shape[1] * x.shape[2] * x.shape[3]  # image dimensionality
        grad = torch.zeros((batch_size, N))  # initialize the gradient estimate

        for i in range(self.k):
            ui = torch.normal(mean=0, std=self.sigma, size=(batch_size, N))  # sample from N(0, I_NxN)
            x_plus_ui = x + self.sigma * ui.reshape(x.shape)  # add ui to the image
            x_minus_ui = x - self.sigma * ui.reshape(x.shape)  # subtract ui from the image

            # compute the probabilities of the class y for the perturbed images
            prob_plus = self.model(x_plus_ui)
            prob_plus = prob_plus[torch.arange(batch_size), y]
            prob_minus = self.model(x_minus_ui)
            prob_minus = prob_minus[torch.arange(batch_size), y]

            # update the gradient estimate
            grad += prob_plus[:, None] * ui - prob_minus[:, None] * ui

        # return the normalized gradient estimate
        return grad / (2 * self.k * self.sigma)

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        x_perturbed = x.clone()
        num_queries = torch.zeros(x.shape[0], dtype=torch.long)

        if self.rand_init:
            # Starting at a uniformly random point
            x_perturbed = x_perturbed + torch.empty_like(x_perturbed).uniform_(-self.eps, self.eps)
            x_perturbed = torch.clamp(x_perturbed, min=0, max=1).detach()

        delta = torch.zeros_like(x_perturbed)

        for i in range(self.n):
            # Estimate gradients using NES
            gradients = self.NES_gradient_estimate(x_perturbed + delta, y)

            # Update the perturbation
            delta = self.momentum * delta + (1 - self.momentum) * self.alpha * gradients.reshape(delta.shape)
            delta = torch.clamp(delta, -self.eps, self.eps)

            # Update the adversarial samples
            x_perturbed = torch.clamp(x + delta, 0, 1)

            # Count the number of queries
            num_queries += self.k * 2

            # Check if all samples are successfully perturbed and early stop if enabled
            outputs = self.model(x_perturbed)
            if self.early_stop:
                if not targeted and not (torch.argmax(outputs, dim=1) == y).any():
                    break
                elif targeted and (torch.argmax(outputs, dim=1) == y).all():
                    break

        assert torch.all(x_perturbed >= 0.) and torch.all(x_perturbed <= 1.)
        assert torch.all(torch.abs(x_perturbed - x) <= self.eps + 1e-7)

        return x_perturbed, num_queries


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """
    def __init__(self, models, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        pass # FILL ME
