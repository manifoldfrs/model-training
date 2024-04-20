import json

import numpy as np
import torch
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

# Define labels
label_list = [
    "O",  # Outside of any entity
    "B-LOCATION",  # Beginning of a location
    "I-LOCATION",  # Inside of a location
    "B-SSN",  # Beginning of a Social Security Number
    "I-SSN",  # Inside of a Social Security Number
    "B-DOB",  # Beginning of a Date of Birth
    "I-DOB",  # Inside of a Date of Birth
    "B-PERSON",  # Beginning of a person's name
    "I-PERSON",  # Inside of a person's name
    "B-BANK",  # Beginning of a Bank Account Number
    "I-BANK",  # Inside of a Bank Account Number
    "B-PHONE",  # Beginning of a Phone Number
    "I-PHONE",  # Inside of a Phone Number
    "B-CREDIT",  # Beginning of a Credit Card Number
    "I-CREDIT",  # Inside of a Credit Card Number
    "B-WISCONSIN",  # Beginning of a Wisconsin Tax ID
    "I-WISCONSIN",  # Inside of a Wisconsin Tax ID
]
label_map = {label: i for i, label in enumerate(label_list)}

# Entity name mapping to label list
entity_label_mapping = {
    "Location": "LOCATION",
    "Bank Account Number": "BANK",
    "Credit Card Number": "CREDIT",
    "Date of Birth": "DOB",
    "Person": "PERSON",
    "Phone Number": "PHONE",
    "SSN": "SSN",
    "Wisconsin Tax ID": "WISCONSIN",
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


def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def tokenize_and_align_labels(text, entities):
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
        tokenized_input, label_ids = tokenize_and_align_labels(text, entities)

        tokenized_inputs["input_ids"].append(tokenized_input["input_ids"])
        tokenized_inputs["attention_mask"].append(tokenized_input["attention_mask"])
        tokenized_inputs["labels"].append(label_ids)

    return tokenized_inputs


# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load the trained model
model_path = "./distilbert_training_20240104/appguard_distilbert_20240105/"
model = DistilBertForTokenClassification.from_pretrained(model_path)

# Print out number of params for size check
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {num_params} trainable parameters.")

# Load and preprocess the evaluation data
eval_data_filename = "ner_eval_data.json"
eval_data_raw = load_data(eval_data_filename)
eval_data = preprocess_data(eval_data_raw, tokenizer)
eval_dataset = NERDataset(eval_data)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./eval_results",  # Output directory
    do_predict=True,  # Whether to run predictions on the test set
)

# Initialize the Trainer
trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset)

# Display the results
print("Evaluation Results:")
print(json.dumps(eval_results))
