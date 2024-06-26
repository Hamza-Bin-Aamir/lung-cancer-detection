__version__ = "2.0.0-alpha"
ImageSetSize = 100
HistogramAlphaScaling = 1000
GaussianKernel = (3, 3)
GaussianSigma = 6
ImageResolution = (256, 256)
ImageSetSize = 100
NumPatients = 200
DatasetLoc = "./LIDC-IDRI"
Epochs = 20
Train_Ratio = 0.8
Save_Model = True

try:
	import numpy as NumPy
	import cv2 as OpenCV
	import pandas as Pandas
	import torch as PyTorch
	import torch.nn as NeuralNet
	import torch.optim as Optimiser
	from torch.utils.data import Dataset
	import pydicom as PyDICOM
	from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
	import os
	import glob
	import xlrd

	print("******************** All core libraries imported successfully ********************")
	print(f"NumPy Version: {NumPy.__version__}")
	print(f"OpenCV Version: {OpenCV.__version__}")
	print(f"Pandas Version: {Pandas.__version__}")
	print(f"PyTorch Version: {PyTorch.__version__}")
	print(f"PyDICOM Version: {PyDICOM.__version__}")
	print(f"XLRD version: {xlrd.__version__}")
	print("----------------------------------------")
	print(f"CUDA Available: {PyTorch.cuda.is_available()}")
	print("Please verify that CUDA is available, otherwise the training will be done on CPU (extremely slow)")
	print("If CUDA Available is False, it is most likely an issue with your installation of PyTorch")
	print("----------------------------------------")


except ImportError as E:
	print(f"Failed to import library: {E.name}, Since it is a core library, the program is exiting")
	exit(-1)
except:
	print("General error while importing core libraries, the program is exiting")
	exit(-1)

# MARK: User-Defined Functions


def dicom_to_numpy(ds: PyDICOM.FileDataset, Show:bool = False) -> OpenCV.Mat:
	"""
	AUTHOR: 	Azeem Liaqat
	CITATION:	Esa Anjum (Stack Overflow)
	DATE: 		5 May 2024
	CATEGORY:	Image Acquisition

	DICOM files contain several tags as well as image data, in this function,
	we extract the pixels and convert it to a numpy array (OpenCV Matrix).

	However, the DICOM format stores the pixel intensity data as well as "rescaling data" that 
	helps us achieve the correct data representation.

	Esa Anjum's original formula did not consider these slope intercepts, so we decided to add them
	to our codebase to achieve better image quality. A before and after is attached in the wiki.  
	"""
	# TODO: Add before after to wiki
	try:
		DCM_Img = ds
		image_data = DCM_Img.pixel_array
		# Get slope and intercept values (assuming they exist in the DICOM data)
		slope = DCM_Img.RescaleSlope if hasattr(DCM_Img, 'RescaleSlope') else 1.0
		intercept = DCM_Img.RescaleIntercept if hasattr(DCM_Img, 'RescaleIntercept') else 0.0
		
		scaled_image = image_data * slope + intercept
		if Show:
			OpenCV.imshow("Image", scaled_image)
			OpenCV.waitKey(0)

		return OpenCV.Mat(scaled_image)
	
	except Exception as E:
		print(f"An error occurred while converting DICOM to Numpy {E.__cause__}")
		raise ValueError("An error occurred while converting DICOM to Numpy")
	
def LoadImage(Location: str, Show:bool = False) -> OpenCV.Mat:
	"""
	AUTHOR: 	Ahmed Abdullah
	DATE: 		5 May 2024
	CATEGORY:	Image Acquisition

	Loads a DICOM format image given a location
	"""
	try:
		Image = PyDICOM.dcmread(Location)
		Image = dicom_to_numpy(Image)
	except ValueError as E:
		print(f"An error occurred while loading the image: {E.__cause__}")
		raise ValueError("An error occurred while loading the image")
	except Exception as E:
		print(f"An unknown error occured while loading the image: {E.__cause__}")
		raise E

	if Show:
		OpenCV.imshow("Images", Image)
		OpenCV.waitKey(0)

	return Image

def ImageProcessingStack(image: OpenCV.Mat) -> None:
	"""
	AUTHOR: 	Muhammad Aoun Abdullah
	DATE: 		5 May 2024
	CATEGORY:	Image Processing

	Increases contrast and reduces "streaking" in the image
	This function works "in-place" as in, it modifies the image directly
	"""
	global HistogramAlphaScaling, GaussianKernel, GaussianSigma

	OpenCV.normalize(image, image, HistogramAlphaScaling)
	image = OpenCV.GaussianBlur(image, GaussianKernel, GaussianSigma)

def PlotConfusion(real, expected) -> None:
	cm = confusion_matrix(real, expected)
	ConfusionMatrixDisplay(cm).plot()


# MARK: User-Defined Classes
class CTScanDataset(Dataset):
	"""
	AUTHOR: 	Hamza Bin Aamir
	DATE: 		6 May 2024
	CATEGORY:	Temporal Image Classification
	"""
	def __init__(self, diagnoses:list[str], images:list[list[str]]):
		"""
		Takes the patient names and image locations in the constructor
		"""
		global ImageSetSize

		# First, we need to normalize the time dimension of images (keep same number of images)
		# The decision-making for this particular method is elaborated on in the readme
		FinalImages 	= 	[]
		FinalDiagnoses 	= 	[]
		for i in range(len(images)):
			if len(images[i]) == ImageSetSize:
				FinalImages.append(images[i])
				FinalDiagnoses.append(diagnoses[i])
			elif len(images[i]) < ImageSetSize:
				# We will duplicate the last image if image set is too small
				images[i].extend(	[images[i][-1]] 		* (ImageSetSize - len(images[i])))
				print("WARNING: Images for a patient have been *duplicated* to normalize the dataset")
				FinalImages.append(images[i])
				FinalDiagnoses.append(diagnoses[i])
			elif len(images[i]) > ImageSetSize:
				# We will split the lists into multiple patients to normalize long image sets
				curr_idx = 0
				curr_img = []
				for j in range(len(images[i])):
					curr_idx += 1
					curr_img.append(images[i][j])
					if curr_idx == ImageSetSize:
						FinalImages.append(curr_img)
						FinalDiagnoses.append(diagnoses[i])
						curr_img = []
						curr_idx = 0
				print("WARNING: Images for a patient have been *split* to normalize the dataset") 
				if curr_idx != 0:
					# If we have some images left, we will duplicate the last image
					curr_img.extend([curr_img[-1]] * (ImageSetSize - len(curr_img)))
					FinalImages.append(curr_img)
					FinalDiagnoses.append(diagnoses[i])
					print("WARNING: Images for a patient have been *duplicated* (after a split) to normalize the dataset") 
		
		self.diagnoses 	= 	FinalDiagnoses
		self.images 	= 	FinalImages

	def __len__(self) -> int:
		"""
		Nice and simple, just return the length of diagnoses
		"""
		return len(self.diagnoses)

	def __getitem__(self, idx: int) -> dict[str, list[str]|list[list[PyTorch.Tensor]]]:
		"""
		Return a dictionary containing the diagnosis and images for a particular patient number
		"""
		global ImageResolution
		RetData = {}
		ImageSet = []

		for img in self.images[idx]:
			Raw = LoadImage(img)

			# Process the image
			# Why not preprocess and resize all the images? Refer to the performance section of the wiki!
   			# TODO: Add Wiki Performance Section
			ImageProcessingStack(Raw)
			Fixed = OpenCV.resize(Raw, ImageResolution)

			# Tensorize the image
			Float = NumPy.array(Fixed).astype("float32")
			Tensor = PyTorch.from_numpy(Float)

			ImageSet.append(Tensor)
		
		ImageSet = PyTorch.cat(ImageSet, 0).reshape(ImageSetSize, ImageResolution[0], ImageResolution[1])

		RetData["diagnosis"] = self.diagnoses[idx]
		RetData["images"] = ImageSet

		return RetData
	
class TemporalVGG16(NeuralNet.Module):
	"""
		AUTHOR: 	Hamza Bin Aamir, Ahmed Abdullah, Muhammad Aoun Abdullah, Azeem Liaqat
		DATE: 		6 May 2024
		CATEGORY:	Temporal Image Classification

		This is our implementation of the VGG 16 Architecture, with channels modified to contain temporal information
		rather than color information.

		This will be a slow model, yes. But we do not require realtime analysis so we can afford the computational expense
		considering that accuracy is a higher priority.
	"""
	def __init__(self, num_classes=1000):
		super(TemporalVGG16, self).__init__()
		# Define the convolutional blocks
		self.block1 = NeuralNet.Sequential(
			NeuralNet.Conv2d(in_channels=ImageSetSize, 
					out_channels=64, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(64),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(64),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.MaxPool2d(kernel_size=2, stride=2)
		)
		self.block2 = NeuralNet.Sequential(
			NeuralNet.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(128),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(128),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.MaxPool2d(kernel_size=2, stride=2)
		)
		self.block3 = NeuralNet.Sequential(
			NeuralNet.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(256),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(256),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(256),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.MaxPool2d(kernel_size=2, stride=2)
		)
		self.block4 = NeuralNet.Sequential(
			NeuralNet.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(512),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(512),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(512),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.MaxPool2d(kernel_size=2, stride=2)
		)
		self.block5 = NeuralNet.Sequential(
			NeuralNet.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(512),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(512),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
			NeuralNet.BatchNorm2d(512),
			NeuralNet.ReLU(inplace=True),
			NeuralNet.MaxPool2d(kernel_size=2, stride=2)
		)

		# Define the fully connected layers
		self.avgpool = NeuralNet.AdaptiveAvgPool2d((7, 7))
		self.fc = NeuralNet.Linear(in_features=512 * 7 * 7, out_features=num_classes)

	def forward(self, x):
		# Pass the input through the convolutional blocks
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		x = self.block5(x)

		# Flatten the features before feeding to fully connected layers
		x = self.avgpool(x)
		x = PyTorch.flatten(x, 1)

		# You can add additional fully connected layers here depending on your task
		# This example assumes the final layer has num_classes output neurons
		x = self.fc(x)
		return x

# MARK: UD Funcs Cont.
def LoadModel(Location:str):
	# Get the script directory (where your script is located)
	script_dir = os.path.dirname(__file__)

	# Join the script directory with the relative location provided
	model_path = os.path.join(script_dir, Location)
	try:
		model = TemporalVGG16()
		state_dict = PyTorch.load(model_path)
		model.load_state_dict(state_dict)
		return model
	except FileNotFoundError:
		print(f"Error: Model file not found at {Location}")
	except RuntimeError as e:
		print(f"Error loading model: {e}")
	raise FileNotFoundError()

def LoadDiagnoses(Location:str, Queries:list[str]) -> Pandas.Series:
	# Get the script directory (where your script is located)
	script_dir 		= os.path.dirname(__file__)

	# Join the script directory with the relative location provided
	diagnoses_path 	= os.path.join(script_dir, Location)
	diagnoses_file	= Pandas.read_excel(diagnoses_path)
	PateintsCol		= "ID"
	DiagnosesCol	= "Diagnosis"	
	Responses 		= []

	for query in Queries:
		MatchedDataframe = diagnoses_file.loc[diagnoses_file[PateintsCol] == query]
		matched_values = MatchedDataframe[DiagnosesCol] 
		try:
			Responses.append(matched_values.iloc[0])
		except:
			Responses.append(0)


	return Responses

def getSubDirs(Location:str):
	return [d for d in os.listdir(os.path.join(os.path.dirname(__file__), Location)) if os.path.isdir(os.path.join(os.path.join(os.path.dirname(__file__), Location), d))]

def getSubFiles(Location: str, ext: str):
	Subdirs = getSubDirs(Location)
	Subsubdirs = []
	final_dirs = []
	image_locs = []
	for Subsubdir in Subdirs:
		Subsubdirs.append(getSubDirs(Location + "\\" + Subsubdir))
	for Subsubsubdir in os.listdir(os.path.join(os.path.dirname(__file__), os.path.join(Location, Subsubdir))):
		final_dirs.append(getSubDirs(Location + "\\" + Subsubdir + "\\" + Subsubsubdir))
	for image_location in os.listdir(os.path.join(os.path.dirname(__file__), os.path.join(Location, os.path.join(Subsubdir, Subsubsubdir)))):
		if(image_location.endswith(".dcm")):
			image_locs.append(os.path.join(os.path.dirname(__file__), os.path.join(Location, os.path.join(Subsubdir, os.path.join(Subsubsubdir, image_location)))))

	return image_locs


def convert_outputs_to_labels(outputs, threshold:float=1):
	"""
	Converts model outputs to predicted labels for binary classification with a threshold.

	Args:
		outputs: A tensor of model outputs with shape (batch_size,).
		threshold: The threshold value for classifying outputs as 0 or 1 (default: 0.5).

	Returns:
		A list of predicted labels (integers) for each sample in the batch.
	"""

	# Apply threshold to get binary labels (0 or 1)
	predicted_labels = (outputs > threshold).float().tolist()  # Convert to float for list conversion
	return predicted_labels