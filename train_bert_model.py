import json
import time
from datetime import datetime

import nlpaug.augmenter.word as naw
import numpy as np
import torch
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Define labels
label_list = [
    "O",
    "B-STREETADDRESS",
    "I-STREETADDRESS",
    "B-CITY",
    "I-CITY",
    "B-STATE",
    "I-STATE",
    "B-ZIPCODE",
    "I-ZIPCODE",
    "B-COUNTRY",
    "I-COUNTRY",
    "B-SSN",
    "I-SSN",
    "B-DOB",
    "I-DOB",
    "B-FIRSTNAME",
    "I-FIRSTNAME",
    "B-MIDDLENAME",
    "I-MIDDLENAME",
    "B-LASTNAME",
    "I-LASTNAME",
    "B-BANKACCOUNTNUMBER",
    "I-BANKACCOUNTNUMBER",
    "B-PHONENUMBER",
    "I-PHONENUMBER",
    "B-CREDITCARDNUMBER",
    "I-CREDITCARDNUMBER",
    "B-WISCONSINTAXID",
    "I-WISCONSINTAXID",
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


def tokenize_and_align_labels(
    text, entities, tokenizer, entity_label_mapping, label_map
):
    tokenized_input = tokenizer(text, truncation=True, is_split_into_words=False)
    labels = ["O"] * len(tokenized_input["input_ids"])

    # Sort entities by their start index in ascending order
    entities = sorted(entities, key=lambda e: text.find(e["value"]))

    for entity in entities:
        entity_type = entity_label_mapping[entity["entity"]]
        entity_value = entity["value"]
        entity_start = text.find(entity_value)
        entity_end = entity_start + len(entity_value)

        if entity_start != -1:
            prev_token_index = None
            for char_index in range(entity_start, entity_end):
                token_index = tokenized_input.char_to_token(0, char_index)
                if token_index is not None:
                    if prev_token_index is None or token_index != prev_token_index:
                        label_prefix = "B-" if prev_token_index is None else "I-"
                        labels[token_index] = f"{label_prefix}{entity_type}"
                    prev_token_index = token_index

    breakpoint()
    label_ids = [label_map[label] for label in labels]
    return tokenized_input, label_ids


def preprocess_data(data):
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in data:
        text, entities = item["text"], item["entities"]
        tokenized_input, label_ids = tokenize_and_align_labels(
            text, entities, tokenizer, entity_label_mapping, label_map
        )

        tokenized_inputs["input_ids"].append(tokenized_input["input_ids"])
        tokenized_inputs["attention_mask"].append(tokenized_input["attention_mask"])
        tokenized_inputs["labels"].append(label_ids)

    return tokenized_inputs


# Start time
start_time = time.time()

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Load Data
print("Loading data...")
data = load_data("ner_data.json")
augmenter = naw.SynonymAug(aug_src="wordnet")

# K-Fold Cross-Validation Setup
kf = KFold(n_splits=3)
fold_results = []
batch_size = 32
lr = 3e-5

for fold, (train_index, test_index) in enumerate(kf.split(data)):
    print(f"Training on fold {fold+1} of {kf.n_splits}...")
    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]

    # Augment only training data
    train_data = augment_data(train_data, augmenter)

    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # Create PyTorch datasets
    train_dataset = NERDataset(train_data)
    test_dataset = NERDataset(test_data)

    # Model initialization with dropout (for regularization)
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_list),
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./checkpoint_results_fold{fold}",
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs_fold{fold}",
        save_strategy="epoch",
        save_total_limit=2,
        evaluation_strategy="epoch",
    )

    # Trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluation
    eval_results = trainer.evaluate()
    fold_results.append(eval_results)
    print(f"Results for fold {fold+1}, LR: {lr}, BS: {batch_size}: {eval_results}")

    current_utc_datetime = datetime.utcnow()
    formatted_utc_datetime = current_utc_datetime.strftime("%Y%m%dT%H%M%SZ")

    # Save the model with datetime in path
    model_save_path = f"./model_fold{fold}_{formatted_utc_datetime}"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)  # Also save the tokenizer

    print(f"Model saved to {model_save_path}")

eval_results_filename = (
    "evaluation_results.json"  # Set a filename for the evaluation results
)
with open(eval_results_filename, "w") as f:
    # Prepare a dictionary to store the formatted results
    formatted_results = {}
    for fold_index, eval_results in enumerate(fold_results):
        formatted_results[f"Fold_{fold_index}"] = {
            "Accuracy": eval_results["eval_accuracy"],
            "Precision": eval_results["eval_precision"],
            "Recall": eval_results["eval_recall"],
            "F1 Score": eval_results["eval_f1"],
            "F2 Score": eval_results["eval_f2"],
            "Classification Report": eval_results["eval_report"],
        }
    json.dump(formatted_results, f, indent=4)

print(f"Saved evaluation results to {eval_results_filename}")

# End time and training duration
end_time = time.time()
training_duration_seconds = end_time - start_time
training_duration_hours = training_duration_seconds / 3600  # Convert to hours

print(f"Training took: {training_duration_seconds:.2f} seconds")
print(f"Which is approximately: {training_duration_hours:.2f} hours")

# Note: Implement error analysis and consider ensemble methods after training.
