import torch
import heapq

def get_pair_counts(ids: torch.Tensor) -> dict[tuple[int, int], int]: 
    pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
    pairs_tuple = [tuple(pair.tolist()) for pair in pairs]  
    pair_counts = {}
    for pair in pairs_tuple:
        if pair in pair_counts:
            pair_counts[pair] += 1
        else:
            pair_counts[pair] = 1
    return pair_counts

def merge_ids(ids: torch.Tensor, pair: tuple[int, int], idx: int) -> torch.Tensor:
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i].item(), ids[i + 1].item()) == pair:
            new_ids.append(idx)
            i += 2 
        else:
            new_ids.append(ids[i].item())
            i += 1
    return torch.tensor(new_ids, dtype=ids.dtype, device=ids.device)


def generate_merges(ids: torch.Tensor, num_merges: int) -> dict[tuple[int, int], int]:
    merges = {}
    i = 256
    count = 0
    pair_count = get_pair_counts(ids)
    max_heap = [(-count, pair) for pair, count in pair_count.items()]
    heapq.heapify(max_heap) 
    while count < num_merges and max_heap:
        _, merge_pair = heapq.heappop(max_heap)
        merge_pair = tuple(merge_pair)
        ids = merge_ids(ids, merge_pair, i)
        merges[merge_pair] = i
        i += 1
        count += 1
        pair_count = get_pair_counts(ids)
        max_heap = [(-count, pair) for pair, count in pair_count.items()]
        heapq.heapify(max_heap)
    return merges