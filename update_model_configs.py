import json

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

# Load the config.json file
config_file = "training_bert_400k_20240110/model_fold2_20240111T101204Z/config.json"  # Update this to the correct path
with open(config_file, "r") as file:
    config = json.load(file)

# Update the id2label and label2id in the config
config["id2label"] = {str(i): label for label, i in label_map.items()}
config["label2id"] = {label: str(i) for label, i in label_map.items()}

# Save the updated config back to the file
with open(config_file, "w") as file:
    json.dump(config, file, indent=4)

print("Updated config.json successfully.")
