import json

import numpy as np
import onnxruntime as ort
import torch
from transformers import DistilBertTokenizerFast

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


def load_evaluation_data(filename):
    with open(filename, "r") as file:
        return json.load(file)


def preprocess_data(data, tokenizer, max_length=512):
    print("Preprocessing eval data...")
    input_ids = []
    attention_masks = []

    for item in data:
        encoded = tokenizer.encode_plus(
            item["text"],
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    return (
        torch.cat(input_ids, dim=0).numpy(),
        torch.cat(attention_masks, dim=0).numpy(),
    )


# Load the tokenizer used during training
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load evaluation data
eval_data_filename = "ner_eval_data.json"  # Update this path
eval_data = load_evaluation_data(eval_data_filename)

# Preprocess the evaluation data
input_ids, attention_masks = preprocess_data(eval_data, tokenizer)

# Load the ONNX model
session = ort.InferenceSession("app_guard_distilbert.onnx")  # Update this path

predicted_labels = []
for i in range(len(input_ids)):
    print(f"Running inference on input id {i}...")
    # Run inference
    outputs = session.run(
        None,
        {
            "input_ids": input_ids[i : i + 1],
            "attention_mask": attention_masks[i : i + 1],
        },
    )

    # Apply softmax to convert scores to probabilities
    probabilities = np.exp(outputs[0]) / np.sum(
        np.exp(outputs[0]), axis=-1, keepdims=True
    )

    # Get the label with the highest probability
    predicted_label_ids = np.argmax(probabilities, axis=-1)

    # Convert label IDs to label names
    id_to_label = {id: label for label, id in label_map.items()}
    predicted_labels.append([id_to_label[id] for id in predicted_label_ids[0]])

# Display the predicted labels
print("Printing labels...")
for labels in predicted_labels:
    print(labels)
