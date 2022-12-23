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

save_svs_to_tiles.py is adapted from the Utilizing Automated Breast Cancer Detection to Identify Spatial Distributions of Tumor Infiltrating Lymphocytes in Invasive Breast Cancer method for tile formation

python c:/Users/emm75/Documents/BMI_771/BMI771_Project/cancer_training/patch_extraction_cancer_40X/save_svs_to_tiles.py C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_svs C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles TCGA-D8-A1Y0-01Z-00-DX1.10F40197-4174-43CC-AAD3-8CB85154FB2D.svs
