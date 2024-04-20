# tokes-model-faris

```bash

# Install deps
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# I suggest making a new directory because the model will save at multiple checkpoints

# Create fake data and labels
python3 create_training_data.py

# Train model and evaluate
python3 train_model.py

# Create eval data
python3 create_eval_data.py

# Run inference
python3 eval_model.py

# Export to ONNX
python3 export_to_onnx.py

# Verify ONNX Model
python3 verify_onnx_model.py
```
