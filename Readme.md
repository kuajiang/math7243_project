# Readme

## There are three steps in processing the images:
1. Data cleaning: Determine whether an image file has sufficient quality for subsequent processing. A classification model is developed to evaluate the image quality. The result is a number between 0 and 4, with 2 representing the best quality, 1 and 3 considered usable, and 0 and 4 deemed unsuitable.
2. Hemorrhage type classification: A multi-label classification model is developed to identify the presence of a hemorrhage in the brain. The five types of hemorrhage are: epidural, intraparenchymal, intraventricular, subarachnoid, and subdural. Each label is assigned a value of 0 or 1 after prediction. If all five labels have a value of 0, there is no hemorrhage, and segmentation is unnecessary.
3. Segmentation: A predicted mask image is generated, indicating the area of hemorrhage.

## Testing
#### We have randomly selected 40 images for testing, ensuring they were not used during training, and placed them in the test_images folder. To run the test, ensure the following libraries are installed:
* matplotlib
* numpy
* pandas
* Pillow
* scikit-learn
* tensorflow
#### To execute the three steps mentioned earlier on a randomly selected image from the test_images folder, run the following command:
```
python test.py
```
Alternatively, you can run the test.ipynb file in Jupyter Notebook.

## Weight files after training
- `models/`
  - `cleaning-1681785936-weight.h5`
  - `classify_weights_epoch_12.h5`
  - `seg_unet_1550_59_weight.h5`




## Training
#### To reproduce the training results, first uncomment the code in the 'main' section at the end of the following files:
  - `step1_cleaning.py`
  - `step2_classify.py`
  - `step3_segmentation.py`



#### Then, ensure that the image files are organized according to the required directory structure:
- `renders/`
  - `epidural/`
    - `brain_window/`
  - `intraparenchymal/`
    - `brain_window/`
  - `intraventricular/`
    - `brain_window/`
  - `multi/`
    - `brain_window/`
  - `normal/`
    - `brain_window/`
  - `subarachnoid/`
    - `brain_window/`
  - `subdural/`
    - `brain_window/`
#### Additionally, you should have the labeled segmentation data in the specified directory:
- `labels/`
  - `Results_Epidural_Hemorrhage_Detection_2020-11-16_21.31.26.148.csv`
  - `Results_Intraparenchymal_Hemorrhage_Detection_2020-11-16_21.39.31.268.csv`
  - `Results_Intraventricular_Hemorrhage_Tracing_2020-09-28_15.21.52.597.csv`
  - `Results_Multiple_Hemorrhage_Detection_2020-11-16_21.36.24.018.csv`
  - `Results_Subarachnoid_Hemorrhage_Detection_2020-11-16_21.36.18.668.csv`
  - `Results_Subdural_Hemorrhage_Detection_2020-11-16_21.37.19.745.csv`
