# Project MGS demonstration using pretrained model
print("------------------------ WELCOME TO THE MGS DEMO ------------------------ \n")

#----------------------------------------------------------------------------------------
# MARK: Import the essential "mgs" module
#----------------------------------------------------------------------------------------
print("*** Importing libraries ***")
try:
	import torch as PyTorch
	import torch.nn as NeuralNet
	import matplotlib.pyplot as plt
	from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
	import mgs # this is where our user-defined funcs and classes live
	print(f"SUCCESS: Imported MGS version: {mgs.__version__}")
	
except ImportError as E:
	print(f"Failed to import library: {E.name}, Since it is a core library, the program is exiting")
	exit(-1)
except:
	print("General error while importing core libraries, the program is exiting")
	exit(-1) 

#----------------------------------------------------------------------------------------
# MARK: Load config 
#----------------------------------------------------------------------------------------
print("*** Loading Configuration Variables ***")

config = {}
files = {}
settings = {}

# fill in file values
files["DatasetLoc"]					= "./Dataset"
files["ModelLocation"]				= "./Models/best.pt"

# fill in setting values
settings["HistogramAlphaScaling"] 	= 1000
settings["GaussianKernel"]			= 3
settings["GaussianSigma"]			= 6
settings["ImageResolution"]			= 1024
settings["ImageSetSize"]			= mgs.ImageSetSize

config["files"] 	= files
config["settings"] 	= settings

#----------------------------------------------------------------------------------------
# MARK: Load model
#----------------------------------------------------------------------------------------
print("*** Loading model ***")
model			= mgs.LoadModel(config["files"]["ModelLocation"])

print("SUCCESS: Model Loaded")

#----------------------------------------------------------------------------------------
# MARK: Load dataset
#----------------------------------------------------------------------------------------
print("*** Loading dataset ***")
PatientNames 	= mgs.getSubDirs("./LIDC-IDRI")
Diagnoses 		= mgs.LoadDiagnoses("./LIDC-META/diagnoses.xls", PatientNames)
Images			= []
for Patient in PatientNames:
	Images.append(mgs.getSubFiles(".\\LIDC-IDRI\\" + Patient, ".dcm"))

ValidationDataset = mgs.CTScanDataset(Diagnoses, Images)
ValidationLoader  = PyTorch.utils.data.DataLoader(ValidationDataset, batch_size=1, shuffle=True)

criterion = NeuralNet.L1Loss()
cumulativeLoss = 0.0 

for batch in ValidationLoader:
	inputs, labels = batch['images'], batch['diagnosis']
	
	outputs = model(inputs)

	cumulativeLoss += criterion(outputs, labels)
	cumulativeLoss = outputs.shape[0] * cumulativeLoss.item()

print("Cumulative Loss: " + str(cumulativeLoss))