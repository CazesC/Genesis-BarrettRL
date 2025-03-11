import torch.nn as nn

class PPOCon(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PPOCon, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        features = self.base(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(-2, 2)  # Log standard deviation bounds
        return mean, log_std
