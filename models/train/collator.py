from typing import no_type_check

import torch


class Collator:
    def __init__(self, pad_token_id: int, max_seq_len: int, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

    @no_type_check
    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for feature in features:
            input_ids = feature["input_ids"]
            padded_input_ids = torch.full((self.max_seq_len,), self.pad_token_id, dtype=torch.long)
            seq_len = input_ids.size(0)
            padded_input_ids[:seq_len] = input_ids[:seq_len]
            input_ids_list.append(padded_input_ids)

            attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            attention_mask[:seq_len] = 1
            attention_mask_list.append(attention_mask)

            labels = feature["labels"]
            padded_labels = torch.full((self.max_seq_len,), self.ignore_index, dtype=torch.long)
            padded_labels[:seq_len] = labels[:seq_len]
            labels_list.append(padded_labels)

        return {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
        }
