# **Code structure**:
Here is a brief explanation of the code structure in this repository, organized into the Dataset, LLM, and NER folders:

## ***Dataset***
This folder contains the code for generating the training dataset, cleaning it, and converting it from CSV to JSON format.

### No_PHI_Dataset_Generation.py
Script used to create a synthetic dataset of doctor-patient dialogues NOT containing PHI (Protected health information).

### PHI_Dataset_Generation.py
Script used to create a synthetic dataset of doctor-patient dialogues containing PHI.

### create_json.ipynb
This script was used to convert the previously created CSV file into a JSON file suitable for training the NER model. The resulting JSON contains tokenized text and a BIO label for each token, following the format required by the model.

### uniq_clean_dataset.py
Script used to mitigate errors produced by the LLAMA-8B model during dataset generation.

## ***LLM***

The LLM folder contains the Python files that allow training and testing of the LLM model, which detects the presence or absence of PHI in a text segment.

```bash
LLM/
├── custom_libs_clinical_t5_base/
│   ├── ClinicalT5_ModelTrainer.py
│   ├── data_handler.py
│   └── pii_data_loader.py
├── testing_clinicalT5-base.py
└── training_clinicalT5-base.py
```

### ClinicalT5_ModelTrainer.py
Contains the training, testing, and inference methods for the ClinicalT5 model. In particular, it also includes the method for plotting the ROC curve during testing.

### data_handler.py
Defines data handling during the training and testing phases, in particular dataset loading, cleaning, and splitting.

### pii_data_loader.py
Manages and tokenizes the datasets, from which it creates DataLoaders for training.

### training_clinicalT5-base.py
Trains the ClinicalT5 model on a dataset containing PHI and PII. For usage, consider the following options:

- Training strategy
   -b/--standard &emsp;train-validation-test split 
  
- -n/--num_splits &emsp;&nbsp;&nbsp;&nbsp; Number of  splits
- -e/--num_epochs &nbsp;&nbsp;&nbsp; Number of epoch
- --dataset &emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; Path of dataset.

Example

```bash
python3 training_clinicalT5-base.py -s -n 3 -e 5 --dataset datasets/training_dataset.csv --num_labels 2
```


### testing_clinicalT5-base.py
Performs testing of the model created during training. To use it:
```bash
python3 testing_clinicalT5-base.py
```


## ***NER***

### base_config.cfg
Configuration file that defines the hyperparameters, dataset paths, tokenization settings, and training options required to train the NER model.

### spacy-ner-spy.ipynb
Script used to train the NER model.

### test-ner.ipynb
Script used to test the previously trained NER model in order to evaluate its performance on a new test dataset.
