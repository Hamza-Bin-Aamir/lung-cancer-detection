# %% [markdown]
# # Project MGS (Medical Graphing Study)
# 
# This is the first version of project MGS.

# %%
# numerical python library needed for mathematical operations
import numpy as NumPy
print("numpy version: ", NumPy.__version__)
# this library is useful for processing images
import cv2 as OpenCV
print("OpenCV version: ", OpenCV.__version__)
# we will be storing our data in a dataframe
import pandas as Pandas
print("Pandas version: ", Pandas.__version__)
from matplotlib import pyplot as PyPlot

import os # useful for file handling operations
# pyTorch is a common library for creating neural networks and AI applications
#%pip install torch
# pyTorch dependency
#%pip install torchvision
# pyTorch dependency
#%pip install torchaudio

import torch
import torch.nn as nn
import torch.optim as optim


# This command installs the GPU version of PyTorch
# %pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#print("PyTorch version: ", PyTorch.__version__)
print("CUDA Available: ", torch.cuda.is_available())
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("Please check if this CUDA is available unless your device does not support CUDA")
print("Uncommect the command to install GPU if your device does support CUDA")

# The images we receive are in DICOM format, therefore, we use pyDICOM

import pydicom as DICOM
print("DICOM Version: ", DICOM.__version__)

import random

# %% [markdown]
# # Citations

# %% [markdown]
# ## DICOM to Numpy Array
# Original by Esa Anjum (https://stackoverflow.com/a/74886257), Modified for our use case

# %%
def dicom_to_numpy(ds, Show = False):
		DCM_Img = ds
		image_data = DCM_Img.pixel_array
		# Get slope and intercept values (assuming they exist in the DICOM data)
		slope = DCM_Img.RescaleSlope if hasattr(DCM_Img, 'RescaleSlope') else 1.0
		intercept = DCM_Img.RescaleIntercept if hasattr(DCM_Img, 'RescaleIntercept') else 0.0
		
		scaled_image = image_data * slope + intercept
		if Show:
			OpenCV.imshow(scaled_image)
			OpenCV.waitKey(0)

		return scaled_image

# %% [markdown]
# # Step One: Image Acquisition
# 
# We are primarily using the [Cancer Imaging Archive (LIDC-IDRI) Dataset](https://www.cancerimagingarchive.net/collection/lidc-idri/) for our project. This dataset provides the following information that we will be using for the project:
# 1. Images
# 2. Metadata
# 3. Nodule Counts
# 4. Patient Diagnoses
# 
# ## Data Acquisition and Usage
# 1. Load list of patients -> ```Patients: Dataframe```
# 2. Load patient diagnoses -> ```Patients: Dataframe```
# 3. Load images from top -> ```Patients: Dataframe``` (Location of Top Views in ```TopImages```) and ```TopImages: List of NumPy Arrays```
# 4. Load images from side -> ```Patients: Dataframe``` (Location of Side Views in ```SideImages```) and ```SideImages: List of NumPy Arrays```
# 5. Load images from front -> ```Patients: Dataframe``` (Location of Front Views in ```FrontImages```) and ```FrontImages: List of NumPy Arrays```
# 6. Metadata -> ```Patients: Dataframe```

# %% [markdown]
# ## Image Acquisition Functions

# %%
def LoadImage(Location:os.path, Show:bool = False) -> OpenCV.Mat:
	Image = DICOM.dcmread(Location)
	Image = dicom_to_numpy(Image)
	
	if Show:
		OpenCV.imshow("Images", Image)
		OpenCV.waitKey(0)

	return OpenCV.Mat(Image)

# %% [markdown]
# ## Implementation 

# %%
data = Pandas.DataFrame()
number_of_patients = 6  # Adjust this to the desired number of IDs
patient_list = ["LIDC-IDRI-" + f"{i:04}" for i in range(1, number_of_patients + 1)]
diagnoses = Pandas.read_excel('./LIDC-META/tcia-diagnosis-data-2012-04-20.xls')
complete_diagnoses = []
for patient in patient_list:
	try:
		complete_diagnoses.append(diagnoses['Diagnosis'][
			(diagnoses['Patient ID'])
			[
				diagnoses['Patient ID'] == patient
			].index[0]
			])
	except:
		complete_diagnoses.append(0)

complete_diagnoses = Pandas.Series(complete_diagnoses)

# add directories to all files in a 'files' list
location = './LIDC-IDRI'
patients = []
for item in sorted(os.listdir(location)):  # Sort patient folders alphabetically
	item_path = os.path.join(location, item)
	if os.path.isdir(item_path):
		patient_files = []
		for dirpath, _, filenames in os.walk(item_path):  # Efficiently traverse folders
			for filename in filenames:
				if not filename.endswith(".xml"):
					filepath = os.path.join(dirpath, filename)
					patient_files.append(filepath)
		patients.append(patient_files)	

# %% [markdown]
# # Image Processing
# ## Parameters
# Adjustable to suit your needs and for A/B testing

# %%
HistogramAlphaScaling = 1000
GaussianKernel = (3, 3)
GaussianSigma = 6

# %% [markdown]
# ## Functions
# 
# ### Image Normalisation
# 
# The contrast in the image is too high or too low in most cases, we would prefer to have a standard contrast throughout the dataset, this is why we are applying a histogram normalisation method to make the image data more readable for the model.

# %%
def HistNormalize(image:OpenCV.Mat) -> None:
	OpenCV.normalize(image, image, HistogramAlphaScaling)
	image = OpenCV.GaussianBlur(image, GaussianKernel, GaussianSigma)

# %% [markdown]
# # Training The Model

# %% [markdown]
# ## Splitting the Dataset
# 
# We split our dataset into a random train/test split 

# %%
def train_test_split(data, labels, test_size=0.2):
	"""
	Splits data and labels into training and testing sets manually.

	Args:
		data: A list or NumPy array containing the data points.
		labels: A list or NumPy array containing the corresponding labels.
		test_size: The proportion of data to be used for the testing set (default: 0.2).

	Returns:
		A tuple containing four elements:
			- X_train: The training data.
			- X_test: The testing data.
			- y_train: The training labels.
			- y_test: The testing labels.
	"""
	data_length = len(data)
	test_index = int(data_length * test_size)

	# Shuffle data and labels together for balanced split
	combined = list(zip(data, labels))
	random.shuffle(combined)
	data, labels = zip(*combined)

	X_train = data[:test_index]
	X_test = data[test_index:]
	y_train = labels[:test_index]
	y_test = labels[test_index:]

	return X_train, X_test, y_train, y_test

# %%
X = patients
y = diagnoses['Diagnosis']

test_size = 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)

# %% [markdown]
# ## Training the CNN

# %% [markdown]
# ## Making a Data Loader
# We use the image acquisition and processing algos we wrote before

# %%
class CustomDataLoader(torch.utils.data.Dataset):
	def __init__(self, pat, dia):
		self.patients = pat
		self.diagnoses = dia

	def __len__(self):
		return len(self.diagnoses)

	def __getitem__(self, idx):
		image = []
		diag = self.diagnoses[idx]

		count = 0
		print(idx)
		for loc in self.patients[idx]:
			count += 1
			if count > 100:
				break
			img = LoadImage(loc)
			HistNormalize(img)
			
			# Convert to float version of NumPy array
			imge = img.astype(NumPy.float32) / 255.0
			imge = torch.from_numpy(imge).unsqueeze(0)

			init_shape = imge[0].shape
			for im in imge:
				if im.shape == init_shape:
					image.append(im)
		
		image = torch.cat(image, dim=0)

		return image, diag

# %% [markdown]
# ## Assembling our CNN
# 
# We have chosen the VGG16 architecture after careful consideration

# %%
class VGG16(nn.Module):
	def __init__(self, num_classes=1000):  # Modify num_classes for your task
		super(VGG16, self).__init__()
		self.block_1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.block_2 = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.block_3 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)

		in_features = 256 * 7 * 7
		self.fc = nn.Linear(in_features=in_features, out_features=num_classes)
		

	def forward(self, x):
		# Block 1
		x = self.block_1(x)

		# Block 2
		x = self.block_2(x)

		# Block 3
		x = self.block_3(x)
		# ... (rest of the architecture, if any)

		# Final output layer (modify for your task)
		x = self.fc(x)  # Example fully-connected layer (replace with your output layer)
		return x

# %% [markdown]
# ## Implementation 

# %%
model = VGG16()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 10

for epoch in range(num_epochs):
	print("STARTED")
  	# Create a new dataloader for each epoch (optional for shuffling)
	custom_dataloader = CustomDataLoader(X_train, y_train)
	  
	for i, (images, labels) in enumerate(custom_dataloader):
		# Check if images is a list (multiple images per patient)
		if isinstance(images, list):
			combined = []
		
		# Combine images into a single tensor (e.g., using torch.cat)
		
	images = images.unsqueeze(0)  # Add a batch dimension

	# Forward pass
	print(f"Forward Pass: {epoch}")
	outputs = model(images)
	print(f"Calculating Loss: {epoch}")
	loss = criterion(outputs, y_train)

	# Backward pass
	print(f"Backward Pass: {epoch}")
	optimizer.zero_grad()
	print(f"Backward Loss: {epoch}")
	loss.backward()
	print(f"Optimising: {epoch}")
	optimizer.step()

	# Print information (optional)
	if i % 100 == 0:
	  print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(custom_dataloader)}], Loss: {loss.item():.4f}')