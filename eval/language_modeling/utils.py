from collections import defaultdict
from dataclasses import dataclass
import torch
from typing import Iterator
from torch.utils.data import Sampler

@dataclass
class ContextDataCollator:
    def __call__(self, batch):
        new_batch = defaultdict(list)

        for item in batch:
            new_batch["input_ids"].append(torch.tensor(item["input_ids"], dtype=torch.long))
            labels = torch.tensor(item["labels"], dtype=torch.long)
            new_batch["labels"].append(labels)
            if "encoder_input_ids" in item:
                new_batch["encoder_input_ids"].append(torch.tensor(item["encoder_input_ids"], dtype=torch.long))

            if "encoder_attention_mask" in item:
                new_batch["encoder_attention_mask"].append(torch.tensor(item["encoder_attention_mask"], dtype=torch.long))

            if "distill_prob" in item:
                new_batch["distill_prob"].append(torch.tensor(item["distill_prob"], dtype=torch.float32))
                new_batch["distill_index"].append(torch.tensor(item["distill_index"], dtype=torch.long))

        for key in new_batch:
            new_batch[key] = torch.stack(new_batch[key])
            if key == "encoder_input_ids" and len(new_batch[key].shape) == 4:
                # each item maybe have two encoder input, and we want to merge them in the second dimension
                # shape is (bsz, 2, num_context, context_size)
                new_batch[key] = new_batch[key].view(new_batch[key].size(0), -1, new_batch[key].size(-1))

        return dict(new_batch)

class BatchSequentialSampler(Sampler):
    def __init__(self, data, batch_size, num_process=None) -> None:
        self.total_len = len(data)
        # self.data = data[: self.total_len - self.total_len % batch_size]
        self.batch_size = batch_size
        self.num_process = num_process

    def __iter__(self) -> Iterator[int]:
        if self.num_process is None:
            per_batch_nums = self.total_len // self.batch_size
            for i in range(per_batch_nums):
                for j in range(self.batch_size):
                    yield i + j * per_batch_nums
        else:
            per_batch_nums = self.total_len // self.batch_size // self.num_process
            for i in range(per_batch_nums):
                for j in range(self.batch_size):
                    for k in range(self.num_process):
                        yield i + j * per_batch_nums + k * per_batch_nums * self.batch_size