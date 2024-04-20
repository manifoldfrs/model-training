import torch
from transformers import BertForTokenClassification

# Load the trained model
model = BertForTokenClassification.from_pretrained(
    "./training_bert_100k_20240110/training_bert_20240110/model_fold2_20240111T024739Z/"
)

# Specify input dimensions
input_ids = torch.zeros(1, 512, dtype=torch.long)
attention_mask = torch.zeros(1, 512, dtype=torch.long)

# Export the model
torch.onnx.export(
    model,
    (input_ids, attention_mask),
    "app_guard_distilbert.onnx",
    input_names=["input_ids", "attention_mask"],
)
