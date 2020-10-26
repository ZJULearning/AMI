import torch
from torch import nn


# class Noise(nn.Module):
#     def __init__(self, hidden_size):
#         super(Noise, self).__init__()
#         self.mu_pjt = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size)
#         )
#         self.sigma_pjt = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size)
#         )
#         self.softplus = nn.Softplus()
#
#     def forward(self, h, n_sample=5):
#         mu = self.mu_pjt(h)
#         sigma = self.sigma_pjt(h)
#         sigma = self.softplus(sigma)
#
#         vals = []
#         for _ in range(n_sample):
#             epsiron = torch.randn(*mu.size()).type_as(mu)
#             val = mu + sigma * epsiron
#             vals.append(val)
#         return vals
class Noise(nn.Module):
    def __init__(self, hidden_size):
        super(Noise, self).__init__()
        self.mu_pjt = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.softplus = nn.Softplus()

    def forward(self, h, n_sample=5):
        mu = self.mu_pjt(h)
        sigma = self.softplus(mu)

        vals = []
        for _ in range(n_sample):
            epsiron = torch.randn(*mu.size()).type_as(mu)
            val = mu + sigma * epsiron
            vals.append(val)
        return vals