import torch
from torch.utils.data import Dataset

class UTTTRecordDataset(Dataset):
  def __init__(self, records):
    # Each elem is state, mask, policy, reward
    self.data = [] # State, mask, and policy
    self.rewards = [] # Reward

    self.augmented_data = None

    for record in records:
      for state, policy in zip(record["player_one"]["states"], record["player_one"]["policies"]):
        self.data.append((state[0], state[1], policy))
        self.rewards.append(record['winner'])
      for state, policy in zip(record["player_two"]["states"], record["player_two"]["policies"]):
        self.data.append((state[0], state[1], policy))
        self.rewards.append(-record["winner"])

    self.data = torch.Tensor(self.data)
    self.rewards = torch.Tensor(self.rewards)

  def augment_data(self):
    # [dataset_size, 3, 81]
    def to_2d(x):
      dataset_size = x.shape[0]
      # Reshape flat tensor to 2D tensor
      x = x.view(dataset_size, 3, 3, 3, 3, 3)
      x = x.permute(0, 1, 2, 4, 3, 5)
      x = x.contiguous()
      x = x.view(dataset_size, 3, 9, 9)
      return x, dataset_size

    def to_flat(x, dataset_size):
      # Reshape back to flat
      x = x.contiguous()
      x = x.view(dataset_size, 3, 3, 3, 3, 3)
      x = x.permute(0, 1, 2, 4, 3, 5)
      x = x.contiguous()
      x = x.view(dataset_size, 3, 81)

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
      return self.augmented_data[idx][0], self.augmented_data[idx][1], self.augmented_data[idx][2], self.rewards[base_idx]
    return self.data[idx][0], self.data[idx][1], self.data[idx][2], self.rewards[idx]