import json
import random

import faker

fake = faker.Faker()

us_fake = faker.Faker("en_US")


def maybe_uppercase(text):
    return text.upper() if random.random() < 0.5 else text


def generate_phone_number():
    formats = [
        "###-###-####",
        "##########",  # With and without dash, with area code
        "###-####",
        "#######",  # With and without dash, without area code
    ]
    return fake.numerify(random.choice(formats))


def maybe_add_noise(text, noise_probability=0.1):
    if random.random() < noise_probability:
        index_to_change = random.randrange(len(text))
        return (
            text[:index_to_change] + fake.random_letter() + text[index_to_change + 1 :]
        )
    return text


def generate_wisconsin_tax_id():
    tax_id = f"{random.randint(1, 999):03}-{random.randint(1, 999999999):09}-"
    check_digits = (
        sum(
            int(digit) * (3 if i % 2 == 0 else 1)
            for i, digit in enumerate(tax_id.replace("-", "")[:13])
        )
        % 10
    )
    tax_id += f"{check_digits:02}"
    return tax_id


def format_ssn_with_dashes():
    ssn = fake.ssn()
    return (
        ssn
        if random.choice([True, False])
        else ssn[:3] + "-" + ssn[3:5] + "-" + ssn[5:]
    )


def generate_data(num_samples):
    data = []
    for _ in range(num_samples):
        # name = fake.name().split()
        # first_name, last_name = maybe_uppercase(name[0]), maybe_uppercase(name[-1])
        # middle_name = maybe_uppercase(name[1]) if len(name) > 2 else ""
        first_name = maybe_uppercase(fake.first_name())
        middle_name = maybe_uppercase(fake.first_name())
        last_name = maybe_uppercase(fake.last_name())
        street_address = us_fake.street_address()
        # address_line2 = us_fake.secondary_address()
        address_city = us_fake.city()
        address_state = random.choice([us_fake.state_abbr(), us_fake.state()])
        address_zip = us_fake.zipcode()
        city = maybe_uppercase(fake.city())
        country = maybe_uppercase(fake.country())
        ssn = format_ssn_with_dashes()
        dob_formats = ["%B %d, %Y", "%m/%d/%Y", "%B %d"]
        dob = fake.date_of_birth().strftime(random.choice(dob_formats))
        bank_account_length = random.randint(8, 12)
        bank_account = fake.numerify("##########"[:bank_account_length])
        credit_card_number = fake.credit_card_number()
        phone_number = generate_phone_number()
        wisconsin_tax_id = generate_wisconsin_tax_id()
        random_money_amount = f"${round(random.uniform(100, 5000), 2)}"
        random_date_formats = ["%B %d, %Y", "%m/%d/%Y", "%B %d"]
        random_date = fake.date_between(start_date="-30y", end_date="today").strftime(
            random.choice(random_date_formats)
        )

        # Adjust logic here to control which entities are included in the data
        include_name = random.choice([True, False])
        include_first_name_only = random.choice([True, False])
        include_middle_name = random.choice([True, False])
        include_us_address = random.choice([True, False])
        include_country = False if include_us_address else True
        include_ssn = random.choice([True, False])
        include_dob = random.choice([True, False])
        include_bank = random.choice([True, False])
        include_credit_card = random.choice([True, False])
        include_phone_number = random.choice([True, False])
        include_wisconsin_tax_id = random.choice([True, False])

        text = ""
        entities = []
        scenario = [
            f"I recently received a notice saying I owe {random_money_amount} in taxes. Could you help me understand why this is?",
            "I'm writing to inquire about the status of my tax return.",
            "I need to update my address and phone number in your records.",
            "I'm having trouble accessing my online account; can you assist me?",
            "I made a payment recently, but it doesn't seem to be reflected in my balance.",
            f"I'd like to confirm if you've received my payment of {random_money_amount} which I sent on {random_date}.",
            "Can you provide me with details about the tax deductions applicable for homeowners?",
            f"I'm planning to move to {fake.country()} next month and need to know the tax implications.",
            "I have not received my tax refund for this year. Could you provide an update?",
            "I need assistance in setting up a payment plan for my outstanding tax balance.",
            "Could you explain how the recent tax changes will affect my small business?",
            "I am writing to dispute a penalty charge added to my tax account.",
            "I need guidance on how to declare my income from freelance work.",
            "I'm reaching out to verify the authenticity of a tax notice I received via email.",
            "I would like to know the deadline for filing tax returns this year.",
            "I am interested in learning more about tax credits for education expenses.",
            "Could you help me understand the tax exemptions available for military personnel?",
            "I need to update my filing status following my recent marriage.",
            "Can you provide information on how to claim tax relief for medical expenses?",
            f"I recently received a notice saying I owe {random_money_amount} in taxes. Could you help me understand why this is?",
            "I need to update my address and phone number in your records.",
            "I'm having trouble accessing my online account; can you assist me?",
            "I made a payment recently, but it doesn't seem to be reflected in my balance.",
            "I'd like to update my personal details following my recent move. Can you guide me through the process?",
            "I'm calling to report a lost credit card and request a replacement.",
            "I need to set up a new bank account and would like some information about your services.",
            "I'm filling out the registration form and want to confirm the details you need.",
            "I'd like to verify my identity for the account setup; what information do you require?",
            "I'm planning a birthday party for next month and need to book a venue. Could you suggest some places in the area?",
            "I need to change the phone number associated with my account. What is the procedure?",
            "I'm enrolling in your loyalty program and have a few questions about the registration.",
            "I'm updating my emergency contact details and need to provide my spouse's phone number.",
            "I would like to apply for a loan and need guidance on the documents required.",
            "I'm scheduling a medical appointment and they've asked for my health insurance details.",
            "I'm registering for an online course and they require some personal information for their records.",
            "I lost my wallet and need to report my cards as missing. Can you assist me with that?",
            "I’m creating my profile on your website and need to fill in some personal details for verification.",
            "I'm opening an investment account and need to provide some financial information.",
        ]

        if include_name:
            if include_first_name_only:
                name_text = first_name
                entities.append({"entity": "First Name", "value": first_name})
            elif include_middle_name:
                name_text = f"{first_name} {middle_name} {last_name}"
                entities.append({"entity": "First Name", "value": first_name})
                entities.append({"entity": "Middle Name", "value": middle_name})
                entities.append({"entity": "Last Name", "value": last_name})
            else:
                name_text = f"{first_name} {last_name}"
                entities.append({"entity": "First Name", "value": first_name})
                entities.append({"entity": "Last Name", "value": last_name})

            intro_phrases = [
                "Hello, my name is",
                "Hi there! I'm",
                "Hey, I'm",
                "Greetings, I'm",
                "Nice to meet you, I go by",
                "Call me",
                "Pleased to meet you, I'm",
                "Good day, my name's",
                "Hello there, you can call me",
                "Hi, I go by the name of",
                "My name is",
                "Hi! My name is",
                "Hello! My name is",
                "Greetings! My name is",
                "Hey! My name is",
                "Hi there! My name is",
                "Hey there! My name is",
            ]

            text += maybe_add_noise(f"{random.choice(intro_phrases)} {name_text}. ")

        if include_us_address:
            entities.append({"entity": "Street Address", "value": street_address})
            entities.append({"entity": "City", "value": address_city})
            entities.append({"entity": "State", "value": address_state})
            entities.append({"entity": "Zip Code", "value": address_zip})

            address_text = (
                f"{street_address}, {address_city}, {address_state}, {address_zip}"
            )

            location_phrases = [
                "I am from",
                "I hail from",
                "I live in",
                "My home city is",
                "I reside in",
                "I'm originally from",
                "I spent my childhood in",
                "Originally from",
                "I grew up in",
                "My hometown is",
                "I'm emailing regading my property at",
                "I'm contacting you about my property at",
                "I'm writing to you about my property at",
                "I'm emailing in regards to my property at",
                "I'm writing about my property at",
            ]

            text += f"{random.choice(location_phrases)} {address_text}. "

        if include_country:
            entities.append({"entity": "City", "value": city})
            entities.append({"entity": "Country", "value": country})

            location_phrases = [
                "I am from",
                "I hail from",
                "I live in",
                "My home city is",
                "I reside in",
                "I'm originally from",
                "I spent my childhood in",
                "Originally from",
                "I grew up in",
                "My hometown is",
            ]

            location_text = f"{city}, {country}"
            text += f"{random.choice(location_phrases)} {location_text}. "

        if include_ssn:
            ssn_phrases = [
                "My SSN is",
                "SSN:",
                "You'll find my SSN is",
                "Social Security Number:",
                "My Social Security Number is",
                "my SSN is",
                "my ssn is",
            ]
            text += f"{random.choice(ssn_phrases)} {ssn}. "
            entities.append({"entity": "SSN", "value": ssn})

        if include_dob:
            dob_phrases = [
                "I was born on",
                "My birthday falls on",
                "DOB:",
                "Date of Birth:",
                "I celebrate my birthday on",
                "I was brought into this world on",
            ]
            text += f"{random.choice(dob_phrases)} {dob}. "
            entities.append({"entity": "Date of Birth", "value": dob})

        if include_bank:
            bank_phrases = [
                "My bank account number is",
                "Account Number:",
                "Banking details:",
                "I handle my finances through account number",
                "My savings are kept at",
                "My bank number is",
            ]
            text += f"{random.choice(bank_phrases)} {bank_account}. "
            entities.append({"entity": "Bank Account Number", "value": bank_account})

        if include_credit_card:
            cc_phrases = [
                "My credit card number is",
                "Card Number:",
                "Credit Card:",
                "My credit card is",
                "The number for my credit card is",
            ]
            text += f"{random.choice(cc_phrases)} {credit_card_number}. "
            entities.append(
                {"entity": "Credit Card Number", "value": credit_card_number}
            )

        if include_phone_number:
            phone_phrases = [
                "My phone number is",
                "Call me at",
                "Contact Number:",
                "Reach me on",
                "You can reach me at",
                "My contact is",
                "Feel free to call me at",
                "Here's my number",
                "My number is",
                "My phone number is",
                "My cell number is",
                "My mobile number is",
                "My number is",
                "Call me maybe?",
            ]
            text += f"{random.choice(phone_phrases)} {phone_number}. "
            entities.append({"entity": "Phone Number", "value": phone_number})

        if include_wisconsin_tax_id:
            tax_id_phrases = [
                "My Wisconsin Tax ID is",
                "Tax ID:",
                "WI Tax Number:",
                "Please find my Wisconsin Tax ID as:",
            ]
            text += f"{random.choice(tax_id_phrases)} {wisconsin_tax_id}. "
            entities.append({"entity": "Wisconsin Tax ID", "value": wisconsin_tax_id})

        if not any(
            [
                include_name,
                include_us_address,
                include_country,
                include_ssn,
                include_dob,
                include_bank,
                include_credit_card,
                include_phone_number,
                include_wisconsin_tax_id,
            ]
        ):
            text += fake.paragraph(nb_sentences=5)

        text += random.choice(scenario)
        text += " " + fake.paragraph(nb_sentences=3)
        data.append({"text": text.strip(), "entities": entities})

    return data


num_samples = 10  # Adjust the number of samples as needed
data = generate_data(num_samples)

with open("ner_data.json", "w") as f:
    json.dump(data, f, indent=4)
