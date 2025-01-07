import torch
import torch.nn.functional as F

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        self.norm = torch.nn.BatchNorm1d(embed_size)
        self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        self.mlp2 = torch.nn.Linear(embed_size, embed_size)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.mlp3 = torch.nn.Linear(embed_size, embed_size)
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        hidden_state = self.norm(hidden_state)
        # hidden_state = F.relu((self.mlp1(hidden_state)))
        # hidden_state = F.relu((self.mlp2(hidden_state)))
        # hidden_state = self.dropout(hidden_state)
        hidden_state = F.relu((self.mlp3(hidden_state)))
        logits = self.mlp(hidden_state)

        return logits
