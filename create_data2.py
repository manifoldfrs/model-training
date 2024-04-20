import dataclasses
import json
import random
import re
from pathlib import Path
from typing import List, Optional, Union, Generator

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider
from pandas import DataFrame
from tqdm import tqdm

from presidio_evaluator.data_generator.faker_extensions import (
    FakerSpansResult,
    NationalityProvider,
    OrganizationProvider,
    UsDriverLicenseProvider,
    IpAddressProvider,
    AddressProviderNew,
    SpanGenerator,
    RecordsFaker,
    PhoneNumberProviderNew,
    AgeProvider,
)

class PresidioDataGenerator:
    # ... (copy the entire PresidioDataGenerator class here) ...

# Create a RecordsFaker (Faker object which prefers samples multiple objects from one record)
faker = RecordsFaker(records=fake_data_df, local="en_US")
faker.add_provider(IpAddressProvider)
faker.add_provider(NationalityProvider)
faker.add_provider(OrganizationProvider)
faker.add_provider(UsDriverLicenseProvider)
faker.add_provider(AgeProvider)
faker.add_provider(AddressProviderNew)  # More address formats than Faker
faker.add_provider(PhoneNumberProviderNew)  # More phone number formats than Faker

# Create Presidio Data Generator
data_generator = PresidioDataGenerator(custom_faker=faker, lower_case_ratio=0.05)
data_generator.add_provider_alias(provider_name="name", new_name="person")
data_generator.add_provider_alias(
    provider_name="credit_card_number", new_name="credit_card"
)
data_generator.add_provider_alias(
    provider_name="date_of_birth", new_name="birthday"
)

def generate_data(num_samples):
    templates = [
        "My name is {{name}}.",
        "I live at {{address}}.",
        # Add more templates as needed...
    ]
    return data_generator.generate_fake_data(templates, num_samples)

num_samples = 100000  # Adjust the number of samples as needed
data = generate_data(num_samples)

with open("ner_data.json", "w") as f:
    json.dump(data, f, indent=4)