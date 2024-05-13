# Project MGS demonstration using pretrained model
print("------------------------ WELCOME TO THE MGS DEMO ------------------------ \n")

#----------------------------------------------------------------------------------------
# MARK: Import the essential "mgs" module
#----------------------------------------------------------------------------------------
print("*** Importing libraries ***")
try:
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
config = None

# try to load from file, otherwise use default values
try:
	import configparser
	config = configparser.ConfigParser()
	config.read("config.ini")
	assert config['files'] is not None

except ImportError as E:
	print("WARNING: UNABLE TO FIND \"configparser\" MODULE, USING DEFAULT VALUES")
	mgs.loadDefault(config)
except Exception as E:
	print("WARNING: UNABLE TO READ CONFIG FILE, USING DEFAULT VALUES")
	mgs.loadDefault(config)
finally:
	print("SUCCESS: Configuration file (\"config.ini\") or default values loaded")

