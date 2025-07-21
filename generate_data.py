import pandas as pd
import random
import string
import re
from faker import Faker

# Initialize Faker and seed for reproducibility
fake = Faker()
random.seed(42)
Faker.seed(42)

def random_phone():
    """Generate a uniform 10-digit phone number formatted XXX-XXX-XXXX."""
    digits = ''.join(random.choices(string.digits, k=10))
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"

def generate_variations(original_value, field_type):
    """Generate realistic variations for name, email, phone, and address."""
    if field_type == 'name':
        parts = original_value.split()
        first, last = parts[0], parts[-1]
        return [
            original_value,
            original_value.upper(),
            original_value.lower(),
            f"{first} {last}",
            f"{first[0]}. {last}",
            original_value + " Jr.",
            "Dr. " + original_value,
        ]
    elif field_type == 'email':
        local, domain = original_value.split('@')
        return [
            original_value,
            original_value.upper(),
            f"{local}+work@{domain}",
            f"{local.replace('.', '')}@{domain}",
            f"{local}123@{domain}",
        ]
    elif field_type == 'phone':
        digits = re.sub(r'\D', '', original_value)
        return [
            original_value,
            f"({digits[:3]}) {digits[3:6]}-{digits[6:]}",
            f"{digits[:3]}.{digits[3:6]}.{digits[6:]}",
            f"{digits[:3]} {digits[3:6]} {digits[6:]}",
            digits,
        ]
    elif field_type == 'address':
        return [
            original_value,
            original_value.replace('Street', 'St'),
            original_value.replace('Avenue', 'Ave'),
            original_value.replace('Road', 'Rd'),
            original_value.replace('Drive', 'Dr'),
            original_value.upper(),
            original_value + ', Apt 2A',
        ]
    else:
        return [original_value]

def create_source_data(n_records=150):
    """Create source dataset with consistent email and phone formats."""
    data = []
    for i in range(n_records):
        full_name = fake.name()
        first, last = full_name.split()[0], full_name.split()[-1]
        domain = fake.free_email_domain()
        email_address = f"{first.lower()}.{last.lower()}@{domain}"
        phone = random_phone()
        data.append({
            'source_id':        f"SRC_{i+1:04d}",
            'full_name':        full_name,
            'email_address':    email_address,
            'phone':            phone,
            'street_address':   fake.address().replace('\n', ', '),
            'company':          fake.company(),
            'job_title':        fake.job(),
            'birth_date':       fake.date_of_birth(minimum_age=18, maximum_age=80),
            'registration_date':fake.date_between(start_date='-2y', end_date='today'),
            'status':           random.choice(['Active','Inactive','Pending','Suspended']),
            'score':            round(random.uniform(0,100),2),
            'category':         random.choice(['Premium','Standard','Basic','VIP']),
        })
    return pd.DataFrame(data)

def create_target_data(source_df, harmonization_rate=0.8):
    """Create target dataset mixing harmonizable and new records."""
    n = len(source_df)
    n_harmonizable = int(n * harmonization_rate)
    harmonizable_indices = random.sample(range(n), n_harmonizable)
    data = []
    for i in range(n):
        tgt_id = f"TGT_{i+1:04d}"
        if i in harmonizable_indices:
            src = source_df.iloc[i]
            data.append({
                'target_id':       tgt_id,
                'customer_name':   random.choice(generate_variations(src['full_name'], 'name')),
                'contact_email':   random.choice(generate_variations(src['email_address'], 'email')),
                'telephone':       random.choice(generate_variations(src['phone'],      'phone')),
                'mailing_address': random.choice(generate_variations(src['street_address'], 'address')),
                'organization':    src['company'] if random.random()>0.3 else fake.company(),
                'position':        src['job_title'] if random.random()>0.4 else fake.job(),
                'date_of_birth':   src['birth_date'],
                'signup_date':     src['registration_date'],
                'account_status':  src['status'] if random.random()>0.2 else random.choice(['Active','Inactive','Pending','Suspended']),
                'rating':          src['score'] + random.uniform(-10,10),
                'tier':            src['category'] if random.random()>0.3 else random.choice(['Gold','Silver','Bronze','Platinum']),
                'region':          fake.state(),
            })
        else:
            data.append({
                'target_id':       tgt_id,
                'customer_name':   fake.name(),
                'contact_email':   fake.email(),
                'telephone':       random_phone(),
                'mailing_address': fake.address().replace('\n', ', '),
                'organization':    fake.company(),
                'position':        fake.job(),
                'date_of_birth':   fake.date_of_birth(minimum_age=18, maximum_age=80),
                'signup_date':     fake.date_between(start_date='-2y', end_date='today'),
                'account_status':  random.choice(['Active','Inactive','Pending','Suspended']),
                'rating':          round(random.uniform(0,100),2),
                'tier':            random.choice(['Gold','Silver','Bronze','Platinum']),
                'region':          fake.state(),
            })
    return pd.DataFrame(data)

def main():
    print("Generating synthetic data for harmonization testing...")
    source_df = create_source_data(150)
    target_df = create_target_data(source_df, harmonization_rate=0.8)

    

    source_df.to_csv('source_data.csv', index=False)
    target_df.to_csv('target_data.csv', index=False)

    print(f"Saved {len(source_df)} source records to source_data.csv")
    print(f"Saved {len(target_df)} target records to target_data.csv")
    print("~ 80% harmonizable via realistic variations — now in a random order.")

if __name__ == "__main__":
    main()
