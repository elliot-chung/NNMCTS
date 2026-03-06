import torch

# Input Shape: [batch_size, 2, 81]
# 2 Channels: First channel show piece placements (Xs and Os), Second shows valid moves locations
class UTTTNet(torch.nn.Module):
  def __init__(self):
    super(UTTTNet, self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, stride=3, padding=0)
    self.bn1 = torch.nn.BatchNorm2d(128)
    self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
    self.bn2 = torch.nn.BatchNorm2d(256)
    self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
    self.bn3 = torch.nn.BatchNorm2d(512)

    self.policy1 = torch.nn.Linear(in_features=512, out_features=256)
    self.policy2 = torch.nn.Linear(in_features=256, out_features=128)
    self.policy3 = torch.nn.Linear(in_features=128, out_features=81)

    self.value1 = torch.nn.Linear(in_features=512, out_features=256)
    self.value2 = torch.nn.Linear(in_features=256, out_features=128)
    self.value3 = torch.nn.Linear(in_features=128, out_features=1)

    self.tanh = torch.nn.Tanh()
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    batch_size = x.shape[0]
    leading_dim = x.shape[:-1]

    # mask = x[:, 1, :]

    x = x.view(*leading_dim, 3, 3, 3, 3)
    x = x.permute(*range(len(leading_dim)), -4, -2, -3, -1)
    x = x.contiguous()
    x = x.view(*leading_dim, 9, 9)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = x.view(batch_size, -1)

    p = self.policy1(x)
    p = self.relu(p)
    p = self.policy2(p)
    p = self.relu(p)
    p = self.policy3(p)

    v = self.value1(x)
    v = self.relu(v)
    v = self.value2(v)
    v = self.relu(v)
    v = self.value3(v)
    v = self.tanh(v)

    return p, v

    return x

# NeuralNode closure for building input
def build_tensor(node):
  state, mask = node.environment.get_canonical_state()
  state = torch.Tensor(state)
  mask = torch.Tensor(mask)
  return torch.stack((state, mask), dim=0).unsqueeze(0)