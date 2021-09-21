import torch
from pandas import DataFrame

from moderator_intervention_full.args import DataTrainingArguments


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_classification_dataset(df: DataFrame, tokenizer, data_args: DataTrainingArguments, label: str):
    df['COMMENT PARENTS CONTENT'] = df['COMMENT']
    if data_args.use_post_id:
        df['COMMENT PARENTS CONTENT'] = df['POST ID'].fillna('').astype(str) + \
                                        tokenizer.sep_token + tokenizer.sep_token + \
                                        df['COMMENT PARENTS CONTENT']
    for i in range(1, data_args.comment_parents_num + 1):
        df['COMMENT PARENTS CONTENT'] = df['COMMENT PARENTS CONTENT'].astype(str) + \
                                        tokenizer.sep_token + tokenizer.sep_token + \
                                        df[f'COMMENT PARENT {i} CONTENT']
    encodings = tokenizer(df['COMMENT PARENTS CONTENT'].tolist(),
                          truncation=True, padding=True,
                          max_length=data_args.max_seq_length)
    return ClassificationDataset(encodings, df[label].astype(int).tolist())
