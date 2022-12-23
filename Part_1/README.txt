Part 1 of the project - Processing H&E images and predicting cancer with Efficientnet

H&E images were downloaded from the GDC Data Portal (https://portal.gdc.cancer.gov/) and annotations for each slide was downloaded from the Dryad database of Data from: High-throughput adaptive sampling for whole-slide histopathology image analysis (HASHI) via convolutional neural networks: application to invasive breast cancer detection by Cruz-Roa, A et. al. (https://datadryad.org/stash/dataset/doi:10.5061/dryad.1g2nt41)

These are all of the downloaded samples:

TCGA-BH-A0B6-01Z-00-DX1
TCGA-A1-A0SE-01Z-00-DX1
TCGA-AR-A0TX-01Z-00-DX1
TCGA-A2-A0CV-01Z-00-DX1
TCGA-B6-A0RP-01Z-00-DX1
TCGA-E2-A153-01Z-00-DX1
TCGA-B6-A0X5-01Z-00-DX1
TCGA-A2-A0T7-01Z-00-DX1
TCGA-A8-A07E-01Z-00-DX1
TCGA-BH-A0BO-01Z-00-DX1
TCGA-BH-A0DE-01Z-00-DX1
TCGA-C8-A26V-01Z-00-DX1

each experiment I ran has a test file associated with it that describes which slides were used and if they were used for training, testing, or validation/prediction.

save_svs_to_tiles.py:
  Adapted from the Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer method for tile formation
  The file takes an svs file and outputs tiles in png format at the equivalent of 40x magnification

To run save_svs_to_tiles.py:

python {PATH_TO_CODE_FILE}/save_svs_to_tiles.py {PATH_TO_SVS_FOLDER} {PATH_TO_OUTPUT_FOLDER} {SVS_FILE_NAME}

align.py:
  Designed to align each tile with its corresponding area on its annotated mask. The output is a labels file with a binary value assigned to each tile. A 1 is a positive cancer tile and a 0 is a negative cancer tile. The slide height and width can be retrieved from the output text of save_svs_to_tiles.py.
  
python {PATH_TO_CODE_FILE}/align.py  {PATH_TO_TILES} {SLIDE_NAME} {SLIDE_HEIGHT} {SLIDE_WIDTH}

crop_image.py:
  Designed to perform preprocessing on each tile to make the tile fit the parameters of EfficientnetB0. The output is either the cancer-positive cropped images or cancer-negative cropped images depending on the int that type is set to. The path must be set to where the cropped images will go before running this function. The input text file is a text file of the aligned tile folders to be cropped together. The training and test files can be combined at this step for easier processing.
  
{PATH_FOR_OUTPUT_FOLDER} python {PATH_TO_CODE_FILE}/crop_image.py {PATH_TO_INPUT_TXT_FILE} {PATH_TO_SVS_FOLDER}   

