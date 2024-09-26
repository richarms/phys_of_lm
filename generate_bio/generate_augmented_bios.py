import random
from datetime import datetime, timedelta
import textwrap
import ollama

# output_file = 'BioS.txt'
n_bios=20

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

us_cities = ["Princeton, NJ", "Austin, TX", "Seattle, WA", "San Francisco, CA", "Chicago, IL", "Boston, MA", "Phoenix, AZ", "Portland, OR", "Denver, CO", "Miami, FL"]
majors = ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Civil Engineering", "Biomedical Engineering", "Chemical Engineering", "Aerospace Engineering", "Materials Science and Engineering", "Environmental Engineering", "Industrial Engineering"]
            

# with open 'us_cities.txt', 'r' as file:
#     us_cities = [line.strip() for line in file]
# with open 'universities.txt', 'r' as file:
#     universities = [line.strip() for line in file]

bio_templates = [
    "{name} was born on {birth_date} in {birth_city}. After graduating from {college} with a speciality in {major}, {pronoun} went on to work at {employer} in {employer_city}.",
    "{name}, born on {birth_date} in {birth_city}, pursued higher education at {college}. Now, {name} works for {employer} located in {employer_city}.",
    "Hailing from {birth_city}, {name} was born on {birth_date}. {name} attended {college} and currently works at {employer} in {employer_city}.",
    "Born in {birth_city} on {birth_date}, {name} studied at {college}. {pronoun} now works for {employer} in {employer_city}.",
    "On {birth_date}, {name} was born in {birth_city}. After completing a degree at {college}, {name} secured a position at {employer}, based in {employer_city}.",
    "{name} was born on {birth_date}. {pronoun} was born in {birth_city}. {pronoun} attended {college}. {pronoun} completed their education with a focus on {major}. {pronoun} was employed at at {employer} in {employer_city}.",
]

qa_templates = [
    "What is the birth date of {name}? Answer: {birth_date}. What is the birth city of {name}? Answer: {birth_city}. Which university did {name} study at? Answer: {college}. What major did {name} study? Answer: {major}. Which company does {name} work for? Answer: {employer}. Where does {name} work? Answer: {employer_city}.",
    "When was {name} born? Answer: {birth_date}. In which city was {name} born? Answer: {birth_city}. At which university did {name} study? Answer: {college}. What was {name}'s major? Answer: {major}. For which company does {name} work? Answer: {employer}. In which city does {name} work? Answer: {employer_city}.",
    "What is {name}'s birth date? Answer: {birth_date}. In what city was {name} born? Answer: {birth_city}. Which institution of higher learning did {name} attend? Answer: {college}. What was {name}'s area of study? Answer: {major}. For which organization does {name} work? Answer: {employer}. In what location does {name} work? Answer: {employer_city}.",
    "On what date was {name} born? Answer: {birth_date}. In what town was {name} born? Answer: {birth_city}. At which college or university did {name} study? Answer: {college}. In what field of study did {name} specialize? Answer: {major}. For what company does {name} currently work? Answer: {employer}. In what city is {name}'s place of employment located? Answer: {employer_city}.",
    "When did {name} come into the world? Answer: {birth_date}. In what place on earth did {name} make their entrance? Answer: {birth_city}. At which institution of learning did {name} pursue their academic pursuits? Answer: {college}. In what discipline did {name} focus their studies? Answer: {major}. For which corporation does {name} currently lend their talents? Answer: {employer}. In what urban center is the headquarters of {name}'s employer located? Answer: {employer_city}."
]

# Function to generate random dates between given years
def random_date(start_year=1890, end_year=2004):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 28)
    random_days = random.randint(0, (end_date - start_date).days)
    random_birthdate = start_date + timedelta(days=random_days)
    return random_birthdate.strftime("%B %d, %Y")

# Function to generate a unique full name
def generate_unique_name(first_names, last_names, used_names):
    while True:
        first_idx = random.randint(0, len(first_names) - 1)
        # first 200 names are male, rest are female
        pronoun = 'He' if first_idx < 200 else 'She'
        first = first_names[first_idx]
        last = random.choice(last_names)
        full_name = f"{first} {last}"
        if full_name not in used_names:
            used_names.add(full_name)
            return full_name, pronoun

# Main function to generate biographies
def generate_biographies(n=n_bios, bio_file="output/synthetic_biographies.txt", qa_file="output/question_answer_pairs.txt"):
    used_names = set()
    with open(bio_file, "w") as bio_f, open(qa_file, "w") as qa_f:
        for _ in range(n):
            # Generate unique name
            full_name, pronoun = generate_unique_name(first_names, last_names, used_names)
            
            # Generate birth date
            birth_date = random_date()
            
            # Select a random birthplace, university, company, and related city
            birthplace = random.choice(us_cities)
            university = random.choice(universities)
            company = random.choice(companies)
            company_city = company_cities[company]
            major = random.choice(majors)
            
            # Choose a random bio template
            template = random.choice(bio_templates)

            # Write the biography in the required format
            biography = template.format(
                name=full_name,
                birth_date=birth_date,
                birth_city=birthplace,
                college=university,
                employer=company,
                employer_city=company_city,
                major=major,
                pronoun=pronoun
            )

            # randomly rewrite every tenth bio with ollama
            if random.random() < 1.:#0.1:
                biography = ollama.chat(model='llama3.2', messages=[
                        {
                            'role': 'user',
                            'content': f'rewrite and reorder the following sentence in a narrative, concise manner. Make sure to use the subjects full name at some point: {biography}',
                        },
                    ])['message']['content'] + '\n'


            # Wrap the biography text to 60 characters
            # biography = textwrap.fill(biography, width=60)

            # Write the biographies to file
            bio_f.write(biography)

            # Generate a set of QAs 
            template = random.choice(qa_templates)
            qa_set = template.format(
                name=full_name,
                birth_date=birth_date,
                birth_city=birthplace,
                college=university,
                employer=company,
                employer_city=company_city,
                major=major
            )

            # Write the QA set to file
            qa_f.write(qa_set)

# Generate synthetic biographies
generate_biographies()

