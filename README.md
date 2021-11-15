# Water Clarity Detection
This project aims to automatically provide a numerical value to gauge water quality using only an image and a ruler. The processing will be done client side through a mobile app, but a web implementation is also planned where an image will be allowed to be uploaded.  
## Instructions for use

Two virtual environments were used.

requirements.txt was used for image segmentation and processing to determine the blur values. Steps 1, 2, & 3

requirementsTraining.txt was used for model training and predictions. Steps 4 & 5

Order To Run:
1. rename_folder.py    # Formats image names
2. prepare_data.py     # Allows segmentation of data for training and testing
3. format_data.py      # Takes segmented image data and outputs to csv
4. model_training.py   # Takes csv and trains model on section of data
5. main.py             # The main file to run, takes model folder, input data, returns output accuracy and prediction


main.py will take the processed collected csv data and run it through the model to produce a "beautified" output

Usage:
main.py datafile.csv locationToModelFile

format_data.py will take an output name and the directory of images to parse where it will extract the info to a csv

Usage:
format_data.py outputname.csv imageFolderPath

model_training.py will train the model on a given csv

Usage:
model_training.py inputfile.csv outputFolderPath

prepare_data.py will take a folder of images and allow the user to prepare training segmented data

Usage:
prepare_data.py "foldersToParse" "foldersToParse" etc...

rename_folder.py will rename the contents of a folder to work with the rest of the python files.

Usage:
rename_folder.py "pathToFolder" "actualMeasuredDepth" "dateTimeMeasured"

test_code.py contains the old methods of object segmentation including Hough Line Transforms and YoloV4 recognition

Non-functional code, but a basis for future expansion (automatic ruler segmentation
### Currently Working On
- 
### Working
- Ruler spacing detection
- Ruler edge detection
- Image thresholding (preprocessing)
- Ruler segmentation along areas of interest
- Blur scoring detection
- Model Training Algorithm
- Processed images
### Not Working
- Automatically determining areas of interest
- Clean edge detection
- Mobile/Web implementation

Updated November 15, 2021
