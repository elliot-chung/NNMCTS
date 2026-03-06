import torch
from torch.utils.data import Dataset

class TTTRecordDataset(Dataset):
  def __init__(self, records):
    # Each elem is state, mask, policy, reward
    self.data = [] # State and policy
    self.rewards = [] # Reward

    self.augmented_data = None

    for record in records:
      for state, policy in zip(record["player_one"]["states"], record["player_one"]["policies"]):
        self.data.append((state, policy))
        reward = record["winner"]
        self.rewards.append(reward)
      for state, policy in zip(record["player_two"]["states"], record["player_two"]["policies"]):
        self.data.append((state, policy))
        reward = -record["winner"]
        self.rewards.append(reward)

    self.data = torch.Tensor(self.data)
    self.rewards = torch.Tensor(self.rewards)

  def augment_data(self):
    # [dataset_size, 2, 9]
    def to_2d(x):
      dataset_size = x.shape[0]
      # Reshape flat tensor to 2D tensor
      x = x.view(dataset_size, 2, 3, 3)
      return x, dataset_size

    def to_flat(x, dataset_size):
      # Reshape back to flat
      x = x.contiguous()
      x = x.view(dataset_size, 2, 9)
      return x

    self.augmented_data = []
    curr, dataset_size = to_2d(self.data)
    for _ in range(4):
      self.augmented_data.append(curr)
      curr = torch.rot90(curr, 1, [2, 3])
      flip = torch.flip(curr, [3])
      self.augmented_data.append(flip)

    self.augmented_data = [ to_flat(x, dataset_size) for x in self.augmented_data ]
    self.augmented_data = torch.cat(self.augmented_data, dim=0)

  def __len__(self):
    return len(self.rewards) if self.augmented_data is None else len(self.augmented_data)

  # State, Policy, Reward
  def __getitem__(self, idx):
    if self.augmented_data is not None:
      base_idx = idx % len(self.rewards)
      return self.augmented_data[idx][0], self.augmented_data[idx][1], self.rewards[base_idx]
    return self.data[idx][0], self.data[idx][1], self.rewards[idx]