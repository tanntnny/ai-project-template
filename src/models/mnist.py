import torch.nn as nn

# ---------------- MNSIT Classifier ----------------
class MNISTClassifier(nn.Module):
    def __init__(self, cfg):
        super(MNISTClassifier, self).__init__()
        self.cfg = cfg

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x, labels):
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, labels)
        return {
            "logits": logits,
            "loss": loss,
        }