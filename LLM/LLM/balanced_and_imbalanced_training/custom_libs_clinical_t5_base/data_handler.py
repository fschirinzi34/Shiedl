from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd

class DataHandler:
	""" A class to handle data loading, cleaning, and splitting for machine learning tasks """
	
	def __init__(self, dataset_path, train_size=0.75, valid_size=0.0, test_size=0.25, seed=10, n_splits=5):
		""" Initialize the DataHandler with dataset path and split parameters """

		self.dataset_path = dataset_path #is a folder, not a file
		self.df_generated = None
		self.df_temist = None
		self.train_size = train_size
		self.valid_size = valid_size
		self.test_size = test_size
		self.n_splits = n_splits # Number of folds for cross-validation
		self.seed = seed
	
	def load_data(self):
		""" Load data from CSV file and perform basic preprocessing """

		# Load CSV with specific encoding and parsing options
		self.df_generated = pd.read_csv(self.dataset_path + "generated_dataset.csv", encoding='utf-8', quotechar='"', skipinitialspace=True)
		print(f"Columns number for generated dataset:\t{self.df_generated.columns}") # Display column names for verification

		# Ensure Sentence column is treated as string
		self.df_generated['Sentence'] = self.df_generated['Sentence'].astype(str)

		self.df_temist = pd.read_csv(self.dataset_path + "zenodo_dataset.csv", encoding='utf-8', quotechar='"', skipinitialspace=True)
		print(
			f"Columns number for TEMIST dataset:\t{self.df_temist.columns}")  # Display column names for verification

		# Ensure Sentence column is treated as string
		self.df_temist['Sentence'] = self.df_temist['Sentence'].astype(str)

	
	def clean_data(self):
		""" Clean the text data by removing unwanted characters """

		# Remove leading/trailing whitespace, periods, and quotes
		self.df_generated['Sentence'] = self.df_generated['Sentence'].str.strip().str.replace('.', '', regex=False).str.replace('"', '', regex=False)
		self.df_temist['Sentence'] = self.df_temist['Sentence'].str.strip().str.replace('.', '', regex=False).str.replace('"', '', regex=False)

	
	def split_data(self, n_rows=None, dataset_type=True):
		""" Split data into training, validation, and test sets using specified proportions """

		# if percentage is None and n_rows is None:
		# 	print("Error: you must specify percentage or n_rows!")
		# 	exit()
		# elif percentage is not None and n_rows is not None:
		# 	print("Error: you must specify percentage or n_rows, not both!")
		# 	exit()

		if dataset_type:
			df = self.df_generated
		else:
			df = self.df_temist

		# Shuffle the dataframe with fixed random state for reproducibility
		df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
		print(self.seed)
		
		# Calculate split indices
		total_size = len(df)
		if n_rows is not None:
			train_end = min(int(n_rows*self.train_size), int(self.train_size*total_size))
			test_end = train_end + min(int(n_rows*self.test_size), int(self.test_size*total_size))
		else:
			train_end = int(self.train_size * total_size)
			test_end = train_end + int(self.test_size * total_size)

		print(f"Number of train rows:\t{train_end}")
		print(f"Number of test rows:\t{test_end}")
		# Extract training data
		train_texts = df.iloc[:train_end]['Sentence'].tolist()
		train_labels = df.iloc[:train_end]['Label'].tolist()
		
		# Extract test data (remaining portion)
		test_texts = df.iloc[train_end:test_end]['Sentence'].tolist()
		test_labels = df.iloc[train_end:test_end]['Label'].tolist()

		print(f"Check Number of train rows:\t{len(train_texts)}")
		print(f"Check Number of test rows:\t{len(test_texts)}")

		print("\n\n")
		print(train_texts[:5])
		return train_texts, train_labels, None, None, test_texts, test_labels

	
	def kfold_split_data(self):
		""" Generate K-fold cross-validation splits using KFold """

		# Initialize KFold with shuffling and fixed random state
		kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=123)
		
		# Convert dataframe columns to lists for easier indexing
		sentences = self.df['Sentence'].tolist()
		labels = self.df['Label'].tolist()
		
		# Generate each fold
		for train_index, test_index in kf.split(sentences):
			# Extract training data for current fold
			train_texts = [sentences[i] for i in train_index]
			train_labels = [labels[i] for i in train_index]
			
			# Extract test data for current fold
			test_texts = [sentences[i] for i in test_index]
			test_labels = [labels[i] for i in test_index]
			
			yield train_texts, train_labels, test_texts, test_labels
	
	def stratified_kfold_split_data(self):
		""" Generate stratified K-fold cross-validation splits using StratifiedKFold """

		# Initialize StratifiedKFold with shuffling and fixed random state
		skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=123)
		
		# Convert dataframe columns to lists for easier indexing
		sentences = self.df['Sentence'].tolist()
		labels = self.df['Label'].tolist()
		
		# Generate each stratified fold
		for train_index, test_index in skf.split(sentences, labels):
			# Extract training data for current fold
			train_texts = [sentences[i] for i in train_index]
			train_labels = [labels[i] for i in train_index]
			
			# Extract test data for current fold
			test_texts = [sentences[i] for i in test_index]
			test_labels = [labels[i] for i in test_index]
			
			yield train_texts, train_labels, test_texts, test_labels
	
	def get_test_data(self):
		""" Extract only the test portion of the data for final model evaluation """

		# Shuffle the dataframe with fixed random state for reproducibility
		self.df = self.df.sample(frac=1, random_state=123).reset_index(drop=True)
		
		# Calculate split indices (same logic as split_data method)
		total_size = len(self.df)
		train_end = int(self.train_size * total_size)
		valid_end = train_end + int(self.valid_size * total_size)
		
		# Extract only test data (final portion after train and validation)
		test_texts = self.df.iloc[valid_end:]['Sentence'].tolist()
		test_labels = self.df.iloc[valid_end:]['Label'].tolist()
		
		return test_texts, test_labels