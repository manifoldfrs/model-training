import datetime
import itertools
import json
import os
import time

import nlpaug.augmenter.word as naw
import numpy as np
import pytz
import torch
import torch.distributed as dist
from sklearn.model_selection import KFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    DataCollatorForTokenClassification,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)


# Function to get the current time in PST
def get_pst_time():
    tz = pytz.timezone("America/Los_Angeles")  # PST timezone
    return datetime.datetime.now(tz).strftime("%Y%m%d_%H%M%S")


# Initialize the accelerator
# accelerator = Accelerator()

# Define labels
label_list = [
    "O",  # Outside of any entity
    "B-STREETADDRESS",  # Beginning of a street address
    "I-STREETADDRESS",  # Inside of a street adddress
    "B-CITY",  # Beginning of a street address
    "I-CITY",  # Inside of a street adddress
    "B-STATE",  # Beginning of a street address
    "I-STATE",  # Inside of a street adddress
    "B-ZIPCODE",  # Beginning of a street address
    "I-ZIPCODE",  # Inside of a street adddress
    "B-COUNTRY",  # Beginning of a street address
    "I-COUNTRY",  # Inside of a street adddress
    "B-SSN",  # Beginning of a Social Security Number
    "I-SSN",  # Inside of a Social Security Number
    "B-DOB",  # Beginning of a Date of Birth
    "I-DOB",  # Inside of a Date of Birth
    "B-FIRSTNAME",  # Beginning of a person's name
    "I-FIRSTNAME",  # Inside of a person's name
    "B-MIDDLENAME",  # Beginning of a person's name
    "I-MIDDLENAME",  # Inside of a person's name
    "B-LASTNAME",  # Beginning of a person's name
    "I-LASTNAME",  # Inside of a person's name
    "B-BANKACCOUNTNUMBER",  # Beginning of a Bank Account Number
    "I-BANKACCOUNTNUMBER",  # Inside of a Bank Account Number
    "B-PHONENUMBER",  # Beginning of a Phone Number
    "I-PHONENUMBER",  # Inside of a Phone Number
    "B-CREDITCARDNUMBER",  # Beginning of a Credit Card Number
    "I-CREDITCARDNUMBER",  # Inside of a Credit Card Number
    "B-WISCONSINTAXID",  # Beginning of a Wisconsin Tax ID
    "I-WISCONSINTAXID",  # Inside of a Wisconsin Tax ID
]
label_map = {label: i for i, label in enumerate(label_list)}
# Entity name mapping to label list
entity_label_mapping = {
    "Street Address": "STREETADDRESS",
    "City": "CITY",
    "State": "STATE",
    "Zip Code": "ZIPCODE",
    "Country": "COUNTRY",
    "First Name": "FIRSTNAME",
    "Middle Name": "MIDDLENAME",
    "Last Name": "LASTNAME",
    "Bank Account Number": "BANKACCOUNTNUMBER",
    "Credit Card Number": "CREDITCARDNUMBER",
    "Date of Birth": "DOB",
    "Phone Number": "PHONENUMBER",
    "SSN": "SSN",
    "Wisconsin Tax ID": "WISCONSINTAXID",
}


class NERDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.max_len = max(len(ids) for ids in encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Pad input_ids, attention_mask, and labels
        padding_length = self.max_len - len(item["input_ids"])
        item["input_ids"] = torch.cat(
            [item["input_ids"], torch.zeros(padding_length, dtype=torch.long)]
        )
        item["attention_mask"] = torch.cat(
            [item["attention_mask"], torch.zeros(padding_length, dtype=torch.long)]
        )
        item["labels"] = torch.cat(
            [item["labels"], torch.full((padding_length,), -100, dtype=torch.long)]
        )
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# Function for data augmentation
def augment_data(data, augmenter, num_augments=1):
    print("Augmenting data...")
    augmented_data = []
    for item in data:
        augmented_texts = [augmenter.augment(item["text"]) for _ in range(num_augments)]
        for aug_text in augmented_texts:
            new_item = item.copy()
            new_item["text"] = aug_text[0] if aug_text else ""
            augmented_data.append(new_item)
    return augmented_data + data


# Function to compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Calculate accuracy and F2 score
    accuracy = accuracy_score(true_labels, true_predictions)
    f2 = (
        5
        * precision_score(true_labels, true_predictions)
        * recall_score(true_labels, true_predictions)
    ) / (
        4 * precision_score(true_labels, true_predictions)
        + recall_score(true_labels, true_predictions)
    )

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "f2": f2,
        "accuracy": accuracy,
        "report": classification_report(true_labels, true_predictions),
    }


# Load and preprocess data
def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def tokenize_and_align_labels(text, entities, tokenizer):
    tokenized_input = tokenizer(text, truncation=True, is_split_into_words=False)
    labels = ["O"] * len(tokenized_input["input_ids"])

    for entity in entities:
        entity_type = entity_label_mapping[entity["entity"]]
        entity_start = text.find(entity["value"])
        entity_end = entity_start + len(entity["value"])

        if entity_start != -1:
            char_to_token = tokenized_input.char_to_token(0, entity_start)
            if char_to_token is not None:
                labels[char_to_token] = f"B-{entity_type}"
                for char in range(entity_start + 1, entity_end):
                    token_index = tokenized_input.char_to_token(0, char)
                    if token_index is not None:
                        labels[token_index] = f"I-{entity_type}"

    label_ids = [label_map[label] for label in labels]

    return tokenized_input, label_ids


def preprocess_data(data, tokenizer):
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in data:
        text, entities = item["text"], item["entities"]
        tokenized_input, label_ids = tokenize_and_align_labels(
            text, entities, tokenizer
        )

        tokenized_inputs["input_ids"].append(tokenized_input["input_ids"])
        tokenized_inputs["attention_mask"].append(tokenized_input["attention_mask"])
        tokenized_inputs["labels"].append(label_ids)

    return tokenized_inputs


# Start time
start_time = time.time()


# Initialize the distributed environment
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# Training function
def train(rank, world_size, data, label_list):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    kf = KFold(n_splits=3)
    fold_results = []

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"Training on fold {fold+1} of {kf.n_splits}...")
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        train_data = augment_data(train_data, naw.SynonymAug(aug_src="wordnet"))
        train_data = preprocess_data(train_data, tokenizer)
        test_data = preprocess_data(test_data, tokenizer)

        train_dataset = NERDataset(train_data)
        test_dataset = NERDataset(test_data)

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank
        )

        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=32
        )
        eval_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

        model = DistilBertForTokenClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(label_list)
        ).to(device)

        model = DDP(model, device_ids=[rank])

        training_args = TrainingArguments(
            output_dir=f"./checkpoint_results_fold{fold}",
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"./logs_fold{fold}",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Eval results: {eval_results}")
        fold_results.append(eval_results)

        if rank == 0:
            current_datetime = get_pst_time()
            model_save_path = f"./model_fold{fold}_lr{training_args.learning_rate}_bs{training_args.per_device_train_batch_size}_{current_datetime}"
            model.module.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)

    if rank == 0:
        # Save the evaluation results to a JSON file
        eval_results_filename = "evaluation_results.json"
        formatted_results = {}
        for fold_index, results in enumerate(fold_results):
            formatted_results[f"Fold_{fold_index + 1}"] = {
                "Accuracy": results["eval_accuracy"],
                "Precision": results["eval_precision"],
                "Recall": results["eval_recall"],
                "F1 Score": results["eval_f1"],
                "F2 Score": results["eval_f2"],
                "Classification Report": results["eval_report"],
            }

        with open(eval_results_filename, "w") as f:
            json.dump(formatted_results, f, indent=4)

        print(f"Saved evaluation results to {eval_results_filename}")

    cleanup()


# Main function
def main():
    world_size = torch.cuda.device_count()
    data = load_data("ner_data.json")

    torch.multiprocessing.spawn(
        train, args=(world_size, data, label_list), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    main()
