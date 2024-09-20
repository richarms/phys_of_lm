import random
from datetime import datetime, timedelta
import textwrap
import ollama

output_file = 'BioS.txt'
n_bios=1000

# Function to generate random dates between given years
def random_date(start_year=1800, end_year=2000):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 28)
    random_days = random.randint(0, (end_date - start_date).days)
    random_birthdate = start_date + timedelta(days=random_days)
    return random_birthdate.strftime("%B %d, %Y")

# Data lists
with open('first_names.txt', 'r') as file:
    first_names = [line.strip() for line in file]

with open('surnames.txt', 'r') as file:
    last_names = [line.strip() for line in file]

with open('universities.txt', 'r') as f:
    universities = [line.strip() for line in f]

with open('companies_cities.txt', 'r') as file:
    companies, cc = [i for i in zip(*(line.split(':') for line in file))]
    company_cities = {company.strip(): city.strip() for company, city in zip(companies, cc)}

# companies = ["Meta Platforms", "Google", "Microsoft", "Amazon", "Apple", "Tesla", "Netflix", "Oracle", "IBM", "Intel"]

# company_cities = {
#     "Meta Platforms": "Menlo Park, CA",
#     "Google": "Mountain View, CA",
#     "Microsoft": "Redmond, WA",
#     "Amazon": "Seattle, WA",
#     "Apple": "Cupertino, CA",
#     "Tesla": "Palo Alto, CA",
#     "Netflix": "Los Gatos, CA",
#     "Oracle": "Austin, TX",
#     "IBM": "Armonk, NY",
#     "Intel": "Santa Clara, CA"
# }

us_cities = ["Princeton, NJ", "Austin, TX", "Seattle, WA", "San Francisco, CA", "Chicago, IL", "Boston, MA", "Phoenix, AZ", "Portland, OR", "Denver, CO", "Miami, FL"]
#universities = ["Harvard University", "Stanford University", "MIT", "Yale University", "Columbia University", "UC Berkeley", "University of Chicago", "Princeton University", "Cornell University", "University of Pennsylvania"]
majors = ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Biomedical Engineering", "Chemical Engineering", "Aerospace Engineering", "Materials Science and Engineering", "Environmental Engineering", "Industrial Engineering"]
            

# with open 'us_cities.txt', 'r' as file:
#     us_cities = [line.strip() for line in file]
# with open 'universities.txt', 'r' as file:
#     universities = [line.strip() for line in file]


# Function to generate a unique full name
def generate_unique_name(first_names, last_names, used_names):
    while True:
        first = random.choice(first_names)
        last = random.choice(last_names)
        full_name = f"{first} {last}"
        if full_name not in used_names:
            used_names.add(full_name)
            return full_name

# Main function to generate biographies
def generate_biographies(n=n_bios, output_file="synthetic_biographies.txt"):
    used_names = set()
    with open(output_file, "w") as f:
        for _ in range(n):
            # Generate unique name
            full_name = generate_unique_name(first_names, last_names, used_names)
            
            # Generate birth date
            birth_date = random_date()
            
            # Select a random birthplace, university, company, and related city
            birthplace = random.choice(us_cities)
            university = random.choice(universities)
            company = random.choice(companies)
            company_city = company_cities[company]
            major = random.choice(majors)
            
            
            # Write the biography in the required format
            biography = (f"{full_name} was born on {birth_date}. "
                         f"They were born in {birthplace}. "
                         f"They attended {university}. "
                         f"They completed their education with a focus on {major}. "
                         f"They were employed at at {company} in {company_city}. ")
            
            # randomly rewrite every tenth bio with ollama
            if random.random() < 1.:#0.1:
                biography = ollama.chat(model='mistral', messages=[
                        {
                            'role': 'user',
                            'content': f'rewrite and reorder the following sentence in a narrative way, and try to infer most likely pronouns from their first name. Make sure to use their full name at some point: {biography}',
                        },
                    ])['message']['content'] + '\n'


            # Wrap the biography text to 60 characters
            # biography = textwrap.fill(biography, width=60)

            f.write(biography)
            #f.write(aug_bio)
# Generate synthetic biographies
generate_biographies()

