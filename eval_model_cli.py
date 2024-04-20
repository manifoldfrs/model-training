import json
import time

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TokenClassificationPipeline,
)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


# Model and Tokenizer names
# model_name = "./training_20240108/model_fold2_20240109T105425Z/"
# tokenizer_name = "./training_20240108/model_fold2_20240109T105425Z/"

model_name = "training_bert_cased_100k_20240111/model_fold2_20240112T064526Z"
tokenizer_name = "training_bert_cased_100k_20240111/model_fold2_20240112T064526Z"

start_time = time.time()
print("Total Time:")

# Tokenizer Instantiation
tokenizer_instantiation_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=False)
print("Tokenizer Instantiation:", time.time() - tokenizer_instantiation_start)

# Model Instantiation
model_instantiation_start = time.time()
model = AutoModelForTokenClassification.from_pretrained(
    model_name, local_files_only=False
)
print("Model Instantiation:", time.time() - model_instantiation_start)

# Pipeline Instantiation
pipeline_instantiation_start = time.time()
classifier = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
print("Pipeline Instantiation:", time.time() - pipeline_instantiation_start)

# Assuming you have a way to input text in your Python environment, e.g., input() function
text = input("Please enter your text: ")
print("Input Text: ", text)
# text = "My name is Mark Schmidt and I'm emailing regarding my property at 1242 18th Street, Madison, WI 53719 and my SSN is 425 11 2657. I received a notice on August 15th saying I owed $14,552 in state taxes. I made a partial payment on July 14th of $7,624 (confirmation number: !gZTTB5873). This does not seem to be reflected in my balance. I would like to confirm that the payment has been received and that the overdue fees have been removed."
# text = "hello my name is james tse. i am from new york city."
print(f"Input Text: {text}")


# Pipeline Execution
pipeline_execution_start = time.time()
results = classifier(text)
print("Pipeline Execution:", time.time() - pipeline_execution_start)

print(results)
with open("results.json", "w") as f:
    json.dump(results, f, cls=CustomEncoder)

print("Total Time:", time.time() - start_time)
