import torch
from torch.utils.data import TensorDataset
import collections

def deduplicate_and_average_dataset(original_dataset):
    """
    Deduplicates a dataset based on the 'State' tensor and averages the
    'Policy' and 'Reward' tensors for duplicate states.

    Args:
        original_dataset (torch.utils.data.Dataset): A dataset where each
            element is a tuple of three tensors (State, Policy, Reward).

    Returns:
        torch.utils.data.TensorDataset: A new dataset with unique states
            and averaged corresponding tensors.
    """
    # This dictionary will store the aggregated data.
    # Key: A hashable representation of the State tensor (its byte string).
    # Value: A dictionary holding the original state tensor, the running sum
    #        of policy and reward tensors, and a count of occurrences.
    data_aggregator = collections.OrderedDict()

    print("Processing and aggregating duplicate states...")
    # Use tqdm for a progress bar, which is helpful for large datasets.
    total_found = 0
    for state, policy, reward in original_dataset:
        # Tensors are not hashable, so we can't use them as dict keys directly.
        # We convert the tensor to its raw byte representation for a hashable key.
        state_key = state.numpy().tobytes()

        if state_key not in data_aggregator:
            # If we see this state for the first time, initialize its entry.
            # We store the original tensor to reconstruct the dataset later.
            # We use .clone() to ensure we're not modifying tensors in place.
            data_aggregator[state_key] = {
                "state": state,
                "policy_sum": policy.clone(),
                "reward_sum": reward.clone(),
                "count": 1,
            }
        else:
            # If we've seen this state before, update the sums and count.
            data_aggregator[state_key]["policy_sum"] += policy
            data_aggregator[state_key]["reward_sum"] += reward
            data_aggregator[state_key]["count"] += 1
            total_found += 1

    print(f"Total duplicate states found: {total_found}/{len(original_dataset)}")

    print("Aggregation complete. Creating new dataset...")

    # Prepare lists to hold the final, processed tensors
    final_states = []
    final_policies = []
    final_rewards = []

    # Iterate through the aggregated data to calculate the averages
    for entry in data_aggregator.values():
        count = entry["count"]
        avg_policy = entry["policy_sum"] / count
        avg_reward = entry["reward_sum"] / count

        final_states.append(entry["state"])
        final_policies.append(avg_policy)
        final_rewards.append(avg_reward)

    # Stack the lists of tensors into single tensors for the TensorDataset
    if not final_states:
        # Handle the edge case of an empty input dataset
        return TensorDataset(
            torch.empty(0), torch.empty(0), torch.empty(0)
        )

    final_states_tensor = torch.stack(final_states)
    final_policies_tensor = torch.stack(final_policies)
    final_rewards_tensor = torch.stack(final_rewards)

    # Create the new, deduplicated dataset
    new_dataset = TensorDataset(
        final_states_tensor, final_policies_tensor, final_rewards_tensor
    )

    return new_dataset