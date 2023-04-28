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

    def nes_gradient_estimate(self, adv_x, x, y, targeted):
        grad = torch.zeros_like(adv_x)
        query_count = 0

        for j in range(self.n):
            adv_x.requires_grad = True
            outputs = self.model(adv_x)
            if self.early_stop:
                if not targeted and not (torch.argmax(outputs, dim=1) == y).any():
                    break
                elif targeted and (torch.argmax(outputs, dim=1) == y).all():
                    break

            grad_estimates = torch.zeros_like(adv_x)
            for i in range(self.k):
                delta_i = torch.normal(mean=0, std=1, size=adv_x.shape)  # sample from N(0, I_NxN)
                theta_i = adv_x.detach() + self.sigma * delta_i
                minus_theta_i = adv_x.detach() - self.sigma * delta_i
                plus_output = self.model(theta_i)
                minus_output = self.model(minus_theta_i)

                # Calculate loss
                plus_loss = self.loss_func(plus_output, y)
                minus_loss = self.loss_func(minus_output, y)
                if targeted:
                    plus_loss = -plus_loss
                    minus_loss = -minus_loss

                grad_estimates += delta_i * plus_loss.sum()
                grad_estimates -= delta_i * minus_loss.sum()

            # Normalize the gradient estimate.
            grad_estimates /= (2 * self.k * self.sigma)
            # Update the historical gradient.
            grad = self.momentum * grad + (1 - self.momentum) * grad_estimates

            query_count += 2 * self.k
            adv_x = adv_x.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_x - x, min=-self.eps, max=self.eps).detach()
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

        return adv_x, query_count

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
        adv_x = x.clone().detach()
        queries_per_sample = torch.zeros(len(x), dtype=torch.int32)

        if self.rand_init:
            # Starting at a uniformly random point
            adv_x = adv_x + torch.empty_like(adv_x).uniform_(-self.eps, self.eps)
            adv_x = torch.clamp(adv_x, min=0, max=1).detach()

        for i in range(len(x)):
            xi = x[i, ...][None, ...]
            adv_xi = adv_x[i, ...][None, ...]
            yi = y[i, ...][None, ...]

            adv_xi, query_count = self.nes_gradient_estimate(adv_xi, xi, yi, targeted)

            queries_per_sample[i, ...][None, ...] = query_count
            adv_x[i, ...][None, ...] = adv_xi.detach()

        assert torch.all(adv_x >= 0.) and torch.all(adv_x <= 1.)
        assert torch.all(torch.abs(adv_x - x) <= self.eps + 1e-7)

        return adv_x, queries_per_sample


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
        adv_x = x.clone().detach()
        if self.rand_init:
            # Starting at a uniformly random point
            adv_x = adv_x + torch.empty_like(adv_x).uniform_(-self.eps, self.eps)
            adv_x = torch.clamp(adv_x, min=0, max=1).detach()

        for i in range(self.n):
            adv_x.requires_grad = True

            # Calculate loss
            loss = torch.tensor(0.)
            for model in self.models:
                outputs = model(adv_x)
                loss += self.loss_func(outputs, y)
            if targeted:
                loss = -loss
            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_x, retain_graph=False, create_graph=False)[0]

            adv_x = adv_x.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_x - x, min=-self.eps, max=self.eps)
            adv_x = torch.clamp(x + delta, min=0, max=1).detach()

            # Check if attack goal is met
            with torch.no_grad():
                all_correct = True
                for model in self.models:
                    output = model(adv_x)
                    preds = output.argmax(dim=1)
                    if targeted:
                        all_correct = all_correct and (preds == y).all().item()
                    else:
                        all_correct = all_correct and (preds != y).all().item()
                if all_correct and self.early_stop:
                    break

        assert torch.all(adv_x >= 0.) and torch.all(adv_x <= 1.)
        assert torch.all(torch.abs(adv_x - x) <= self.eps + 1e-7)

        return adv_x
