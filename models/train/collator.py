from typing import no_type_check

import torch


class PretrainingCollator:
    def __init__(self, pad_token_id: int, max_seq_len: int):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    @no_type_check
    def __call__(self, features: list[dict[str, torch.Tensor | int]]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for feature in features:
            input_ids = feature["input_ids"]
            valid_length = int(feature["valid_length"])

            padded_input_ids = torch.full(
                (self.max_seq_len,),
                self.pad_token_id,
                dtype=torch.long,
            )
            padded_input_ids[:valid_length] = input_ids[:valid_length]

            attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
            attention_mask[:valid_length] = 1

            input_ids_list.append(padded_input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(padded_input_ids.clone())

        return {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "attention_mask": torch.stack(attention_mask_list, dim=0),
            "labels": torch.stack(labels_list, dim=0),
        }
