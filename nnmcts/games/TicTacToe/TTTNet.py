import torch

# Input Shape: [batch_size, 9]
# Flat array of value representing games state: 1 for X, -1 for O, 0 for blanks
class TTTNet(torch.nn.Module):
  def __init__(self):
    super(TTTNet, self).__init__()
    self.conv = torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=3, padding=0)
    self.bn = torch.nn.BatchNorm2d(256)

    self.policy1 = torch.nn.Linear(in_features=256, out_features=128)
    self.policy2 = torch.nn.Linear(in_features=128, out_features=9)

    self.value1 = torch.nn.Linear(in_features=256, out_features=128)
    self.value2 = torch.nn.Linear(in_features=128, out_features=1)

    self.tanh = torch.nn.Tanh()
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    batch_size = x.shape[0]

    x = x.view(batch_size, 1, 3, 3)

    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)

    x = x.view(batch_size, -1)

    p = self.policy1(x)
    p = self.relu(p)
    p = self.policy2(p)

    v = self.value1(x)
    v = self.relu(v)
    v = self.value2(v)
    v = self.tanh(v)

    return p, v

    return x

# NeuralNode closure for building input
def build_tensor(node):
  state = node.environment.get_canonical_state()
  return torch.Tensor(state).unsqueeze(0)