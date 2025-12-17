import pandas as pd
import csv
import re
import json
import random
import ast



# /-------------------------------------------- make_dataset_coherent -------------------------------------------------/
# Function that takes the CSV dataset as input and checks whether there are any placeholders (among ours) in the text that 
# have not yet been replaced.
# If so, it replaces the placeholders with one of the possible values ​​we passed to it in "options" for all placeholders, 
# the replacements are done randomly.

def make_dataset_coherent(filename: str, outputfile: str):
    df = pd.read_csv(filename)
    options = {
        "<author_clinical_condition>": ['Hypotension', 'Ischemic heart disease', 'Myocardial infarction',
                                        'Angina pectoris', 'Cardiac arrhythmias',
                                        'Atrial fibrillation', 'Tachycardia', 'Bradycardia', 'Heart failure',
                                        'Cardiomyopathy', 'Myocarditis',
                                        'Pericarditis', 'Endocarditis', 'Mitral valve disease', 'Aortic stenosis',
                                        'Aortic regurgitation', 'Aortic aneurysm',
                                        'Deep vein thrombosis', 'Pulmonary embolism', 'Varicose veins',
                                        'Peripheral arterial disease', 'Intermittent claudication',
                                        'Raynaud\'s phenomenon', 'Phlebitis', 'Hypercholesterolemia', 'Dyslipidemia',
                                        'Atherosclerosis', 'Bronchial asthma',
                                        'Chronic obstructive pulmonary disease', 'Pulmonary emphysema',
                                        'Chronic bronchitis', 'Acute bronchitis', 'Pneumonia',
                                        'Pleurisy', 'Pneumothorax', 'Pleural effusion', 'Pulmonary fibrosis',
                                        'Sarcoidosis', 'Sleep apnea',
                                        'Obstructive sleep apnea syndrome', 'Allergic rhinitis', 'Sinusitis',
                                        'Pharyngitis', 'Laryngitis', 'Tonsillitis',
                                        'Bronchiectasis', 'Pulmonary tuberculosis', 'Silicosis', 'Asbestosis',
                                        'Gastritis', 'Gastric ulcer', 'Duodenal ulcer',
                                        'Gastroesophageal reflux disease', 'Esophagitis', 'Irritable bowel syndrome',
                                        'Crohn\'s disease',
                                        'Ulcerative colitis', 'Diverticulitis', 'Diverticulosis', 'Hemorrhoids',
                                        'Anal fissures', 'Chronic constipation',
                                        'Chronic diarrhea', 'Malabsorption', 'Celiac disease', 'Lactose intolerance',
                                        'Gastroenteritis', 'Pancreatitis',
                                        'Gallstones', 'Cholecystitis', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C',
                                        'Liver cirrhosis', 'Fatty liver disease',
                                        'Intestinal polyposis', 'Type 1 diabetes mellitus', 'Type 2 diabetes mellitus',
                                        'Gestational diabetes', 'Hypoglycemia',
                                        'Hyperglycemia', 'Hypothyroidism', 'Hyperthyroidism',
                                        'Hashimoto\'s thyroiditis', 'Goiter', 'Thyroid nodules',
                                        'Graves\' disease', 'Adrenal insufficiency', 'Cushing\'s syndrome',
                                        'Addison\'s disease', 'Hyperaldosteronism',
                                        'Pheochromocytoma', 'Hypoparathyroidism', 'Hyperparathyroidism', 'Osteoporosis',
                                        'Osteomalacia', 'Polycystic ovary syndrome',
                                        'Hypogonadism', 'Hypergonadism', 'Acromegaly', 'Pituitary dwarfism',
                                        'Diabetes insipidus', 'Migraine',
                                        'Tension headache', 'Cluster headache', 'Epilepsy', 'Parkinson\'s disease',
                                        'Essential tremor', 'Multiple sclerosis',
                                        'Amyotrophic lateral sclerosis', 'Alzheimer\'s disease', 'Vascular dementia',
                                        'Senile dementia', 'Ischemic stroke',
                                        'Hemorrhagic stroke', 'Transient ischemic attack', 'Diabetic neuropathy',
                                        'Peripheral neuropathy', 'Trigeminal neuralgia',
                                        'Facial nerve palsy', 'Carpal tunnel syndrome', 'Herniated disc',
                                        'Cervical spondylosis', 'Low back pain', 'Sciatica',
                                        'Meningitis', 'Encephalitis', 'Myasthenia gravis', 'Rheumatoid arthritis',
                                        'Osteoarthritis', 'Psoriatic arthritis',
                                        'Ankylosing spondylitis', 'Gout', 'Pseudogout', 'Fibromyalgia',
                                        'Polymyalgia rheumatica', 'Systemic lupus erythematosus',
                                        'Sjögren\'s syndrome', 'Scleroderma', 'Dermatomyositis', 'Polymyositis',
                                        'Bursitis', 'Tendinitis', 'Epicondylitis',
                                        'Rotator cuff syndrome', 'Sprain', 'Contusion', 'Fracture', 'Dislocation',
                                        'Muscle strain', 'Muscle cramps', 'Myositis',
                                        'Scoliosis', 'Kyphosis', 'Lordosis', 'Cystitis', 'Urethritis', 'Pyelonephritis',
                                        'Glomerulonephritis', 'Chronic kidney disease',
                                        'Acute kidney injury', 'Kidney stones', 'Benign prostatic hyperplasia',
                                        'Prostatitis', 'Prostate cancer', 'Erectile dysfunction',
                                        'Urinary incontinence', 'Urinary retention', 'Hematuria', 'Proteinuria',
                                        'Nephrotic syndrome', 'Nephritic syndrome',
                                        'Polycystic kidney disease', 'Endometriosis', 'Uterine fibroids',
                                        'Ovarian cysts', 'Vaginitis', 'Cervicitis', 'Dysmenorrhea',
                                        'Amenorrhea', 'Atopic dermatitis', 'Contact dermatitis',
                                        'Seborrheic dermatitis', 'Psoriasis', 'Eczema', 'Hives', 'Acne',
                                        'Rosacea', 'Melasma', 'Actinic keratosis', 'Herpes simplex', 'Herpes zoster',
                                        'Chickenpox', 'Molluscum contagiosum', 'Warts',
                                        'Cutaneous mycoses', 'Cutaneous candidiasis', 'Impetigo', 'Cellulitis',
                                        'Folliculitis', 'Alopecia', 'Hirsutism', 'Melanoma',
                                        'Basal cell carcinoma', 'Squamous cell carcinoma', 'Keloids',
                                        'Hypertrophic scars', 'Iron deficiency anemia',
                                        'Megaloblastic anemia', 'Hemolytic anemia', 'Aplastic anemia',
                                        'Sickle cell anemia', 'Thalassemia', 'Acute leukemia',
                                        'Chronic leukemia', 'Hodgkin\'s lymphoma', 'Non-Hodgkin\'s lymphoma',
                                        'Multiple myeloma', 'Thrombocytopenia',
                                        'Thrombocytosis', 'Immune thrombocytopenic purpura', 'Hemophilia',
                                        'von Willebrand disease',
                                        'Disseminated intravascular coagulation', 'Polycythemia vera', 'Myelofibrosis',
                                        'Myelodysplastic syndrome',
                                        'Food allergies', 'Respiratory allergies', 'Drug allergies', 'Anaphylaxis',
                                        'Primary immunodeficiency',
                                        'Secondary immunodeficiency', 'Autoimmune diseases', 'Vasculitis',
                                        'Antiphospholipid syndrome', 'Behçet\'s disease',
                                        'Acquired immunodeficiency syndrome', 'Major depressive disorder',
                                        'Bipolar disorder', 'Generalized anxiety disorder',
                                        'Panic disorder', 'Obsessive-compulsive disorder',
                                        'Post-traumatic stress disorder', 'Social anxiety disorder',
                                        'Agoraphobia', 'Specific phobias', 'Borderline personality disorder',
                                        'Narcissistic personality disorder',
                                        'Antisocial personality disorder', 'Schizophrenia', 'Schizoaffective disorder',
                                        'Delusional disorder',
                                        'Attention-deficit/hyperactivity disorder', 'Autism spectrum disorders',
                                        'Sleep disorders', 'Insomnia',
                                        'Eating disorders', 'Anorexia nervosa', 'Bulimia nervosa',
                                        'Binge-eating disorder', 'Bronchiolitis',
                                        'Acute laryngitis', 'Otitis media', 'Pharyngotonsillitis',
                                        'Acute gastroenteritis', 'Pediatric asthma',
                                        'Pediatric gastroesophageal reflux', 'Infantile colic',
                                        'Functional constipation', 'Nocturnal enuresis',
                                        'Recurrent fever', 'Febrile seizures', 'Infantile epilepsy',
                                        'Neurodevelopmental disorders', 'Psychomotor retardation',
                                        'Learning disorders', 'Behavioral disorders', 'Fragility fractures',
                                        'Sarcopenia', 'Cachexia', 'Malnutrition in the elderly',
                                        'Immobilization syndrome', 'Pressure ulcers', 'Polypharmacy', 'Delirium'],
        "<author_medical_report>": ['Blood test results', 'Urine analysis', 'X-ray report', 'MRI scan', 'CT scan',
                                    'Ultrasound report', 'ECG report',
                                    'Echocardiogram', 'Biopsy results', 'Pathology report', 'Lab culture results',
                                    'Genetic test results',
                                    'Allergy test panel', 'Hormone levels', 'Cholesterol panel', 'Liver function tests',
                                    'Kidney function tests',
                                    'Thyroid function tests', 'Cardiac stress test', 'Pulmonary function test',
                                    'Bone density scan',
                                    'Colonoscopy report', 'Endoscopy report', 'Mammography report', 'Pap smear results',
                                    'PSA test results'],
        "<author_genetic>": ['BRCA1 mutation', 'BRCA2 mutation', 'APOE e4 allele', 'Factor V Leiden',
                             'Prothrombin mutation', 'MTHFR mutation',
                             'HLA-B27 positive', 'Tay-Sachs carrier', 'Sickle cell trait', 'Hemochromatosis mutation',
                             'Cystic fibrosis carrier', 'Huntington\'s disease gene', 'Lynch syndrome',
                             'Familial hypercholesterolemia'],
        "<author_fertility>": ['Infertility', 'Polycystic ovary syndrome', 'Endometriosis', 'Male factor infertility',
                               'Ovulation disorders',
                               'Tubal factor infertility', 'Unexplained infertility', 'Recurrent pregnancy loss',
                               'Premature ovarian failure', 'Varicocele', 'Azoospermia'],
        "<author_disability>": ['Visual impairment', 'Hearing impairment', 'Mobility impairment',
                                'Cognitive impairment', 'Speech impairment',
                                'Learning disability', 'Autism spectrum disorder', 'Intellectual disability',
                                'Chronic fatigue syndrome', 'Fibromyalgia', 'Multiple sclerosis', 'Spinal cord injury',
                                'Traumatic brain injury'],
        "<author_addiction>": ['Alcohol dependency', 'Nicotine addiction', 'Opioid dependency', 'Cocaine addiction',
                               'Cannabis dependency',
                               'Benzodiazepine dependency', 'Gambling addiction', 'Internet addiction',
                               'Food addiction',
                               'Prescription drug abuse', 'Stimulant abuse']
    }

    processed = []

    for _, row in df.iterrows():
        try:
            text = str(row['final_text'])
            text = re.sub(r'(?s).*?Text:\s*', '', text)
            text = re.sub(r'(?s)\s*Note:.*', '', text)
            text = text.strip()

            try:
                subs_dict = ast.literal_eval(row['final_substitutions_dict'])
            except Exception:
                subs_dict = {}

            placeholders = set(re.findall(r"<[^>]+>", text))
            for ph in placeholders:
                if ph not in subs_dict or isinstance(subs_dict[ph], str) and subs_dict[ph].startswith('<'):
                    value = random.choice(options.get(ph, ["N/A"]))
                    subs_dict[ph] = value
                text = text.replace(ph, subs_dict[ph])

            processed.append({
                'final_text': text,
                'substitutions_dictionary': json.dumps(subs_dict, ensure_ascii=False)
            })

        except Exception as e:
            print(row)
            print("Errore: ", e)
            continue

    final_df = pd.DataFrame(processed)
    final_df.to_csv(outputfile, index=False)


# /---------------------------------------- delete_rows_starting_here_is ----------------------------------------------/
# Controlliamo se le righe contengono la frase "Here is" e "Task", in quel caso eliminiamo la riga in quanto contiene dei
# commenti che il modello ha inserito per errore.
def delete_rows_starting_here_is(input_path: str, output_path: str):

    with open(input_path, newline='', encoding='utf-8') as infile, \
            open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            text = row.get('final_text', '')
            if "Here is" not in text and "Task" not in text:
                writer.writerow(row)

    print(f"Dataset filtrato salvato in: {output_path}")


# /------------------------------------------- delete_rows_by_regex ---------------------------------------------------/
# We check if the lines contain placeholders between <> or [] in the TEXT, in that case we delete the line because the 
# template inserted those placeholders by mistake.
# We also do a line-by-line check to see if the text begins with "I cannot." If so, we delete the line because the model
# was unable to generate the text following the instructions we provided.

def delete_rows_by_regex(input_path: str, output_path: str):
    start_re = re.compile(r'^I cannot')
    pattern = re.compile(r'(<[^>]+>|\[[^\]]+\]|n/a)', re.IGNORECASE)

    with open(input_path, newline='', encoding='utf-8') as infile, \
            open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row_idx, row in enumerate(reader, start=1):
            text = row.get('final_text', '')
            match = start_re.match(text)
            if match:
                print(f"Riga {row_idx + 1}: Contiene \"I cannot\"")
                continue
            matches = pattern.findall(text)
            if matches:
                for placeholder in matches:
                    print(f"Riga {row_idx + 1}: trovato placeholder `{placeholder}`")
                continue

            writer.writerow(row)

    print(f"\nElaborazione completata. File senza placeholder salvato in: {output_path}")


# /--------------------------------------------- clean_dictionary -----------------------------------------------------/
# This function takes the csv file as input and checks the "substitutions_dictionary" column to see if there are any 
# placeholders that the template has indicated like this <something) replacing the ")" with ">".
# Additionally, it checks to see if the "substitutions_dictionary" column contains placeholders other than ours 
# (specified in the "ALLOWED" variable). If so, it deletes the row because the template inserted those placeholders 
# by mistake.
# It also checks to see if there are any "N/A" values ​​in the dictionary, in which case the model has made an error and 
# we delete the row.

def clean_dictionary(input_csv: str, output_csv: str):
    import csv
    import ast
    import sys

    ALLOWED = {
        "<author_personal_name>",
        "<author_personal_email>",
        "<author_personal_phone_number>",
        "<author_personal_url>",
        "<author_personal_address>",
        "<author_clinical_condition>",
        "<author_personal_identifier>",
        "<author_personal_username>",
        "<author_medical_report>",
        "<author_genetic>",
        "<author_fertility>",
        "<author_disability>",
        "<author_addiction>",
    }

    written = 0
    skipped = 0

    with open(input_csv, newline='', encoding='utf-8') as f_in, \
            open(output_csv, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in)
        if "substitutions_dictionary" not in reader.fieldnames:
            print("Errore: manca la colonna 'substitutions_dictionary' nel CSV", file=sys.stderr)
            sys.exit(1)

        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader, start=1):
            str_dict = row['substitutions_dictionary']
            if "\'None\'" in str_dict:
                skipped += 1
                continue

            subs_str = str_dict.strip()
            if not subs_str:
                writer.writerow(row)
                written += 1
                continue

            try:
                subs_dict = ast.literal_eval(subs_str)
                if not isinstance(subs_dict, dict):
                    raise ValueError
            except Exception:
                skipped += 1
                continue

            fixed_dict = {}
            for ph, val in subs_dict.items():
                if ph.startswith("<") and ph.endswith(")"):
                    ph = ph[:-1] + ">"
                fixed_dict[ph] = val

            unexpected = [ph for ph in fixed_dict if ph not in ALLOWED]
            if unexpected:
                skipped += 1
                continue


            row["substitutions_dictionary"] = repr(fixed_dict)
            writer.writerow(row)
            written += 1

    print(f"Righe scritte (valide):   {written}")
    print(f"Righe scartate (non valide): {skipped}")
    print(f"File risultante: {output_csv}")


#/----------------------------------------------- fix_maiuscole_func ------------------------------------------------------/
# This function takes a text string as input and corrects it as follows: if an uppercase letter is found, the function 
# checks whether it is at the beginning of the text, after a period (.), an exclamation point (!), or a question mark (?).
# In all other cases, uppercase is converted to lowercase.
def fix_maiuscole_func(text):
    if not text:
        return text

    result = []
    i = 0

    while i < len(text):
        char = text[i]

        if i == 0:
            result.append(char)
            i += 1
            continue

        if char.isupper() and char.isalpha():
            prev_non_space_idx = i - 1
            while prev_non_space_idx >= 0 and text[prev_non_space_idx].isspace():
                prev_non_space_idx -= 1

            if prev_non_space_idx >= 0:
                prev_char = text[prev_non_space_idx]

                if prev_char in '.!?':
                    result.append(char)
                else:
                    result.append(char.lower())
            else:
                result.append(char)
        else:
            result.append(char)

        i += 1

    return ''.join(result)


# /------------------------------------------- fix_maiuscole ---------------------------------------------------/
def fix_maiuscole(input_csv: str, output_csv: str):
    row_with_medical_pii = []
    num_row = 0

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            subs = row.get('substitutions_dictionary', '') or ''
            final_text = row.get('final_text', '')

            cleaned_text = fix_maiuscole_func(final_text)

            row_with_medical_pii.append({
                'final_text': cleaned_text,
                'substitutions_dictionary': subs
            })
            num_row += 1

    with open(output_csv, 'w', newline='', encoding='utf-8') as file_output:
        fieldnames = ['final_text', 'substitutions_dictionary']
        writer = csv.DictWriter(file_output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_with_medical_pii)

    print(f"Row new dataset = {num_row}")
    print(f"Dataset pulito salvato in: {output_csv}")


if __name__ == '__main__':
    make_dataset_coherent('dataset_shield_8000_filtered_only_valid.csv', 'complete_dataset_shield_coherent.csv')
    delete_rows_starting_here_is('complete_dataset_shield_coherent.csv', 'complete_dataset_shield_coherent_filtered_from_here_is.csv')
    delete_rows_by_regex('complete_dataset_shield_coherent_filtered_from_here_is.csv', 'complete_dataset_shield_coherent_filtered_by_regex.csv')
    clean_dictionary('complete_dataset_shield_coherent_filtered_by_regex.csv', 'complete_dataset_shield_final_cleaned_seed.csv')
    fix_maiuscole('complete_dataset_shield_final_cleaned_seed.csv','complete_dataset_shield_final_cleaned_seed_fix_maisc.csv')
