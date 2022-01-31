# MI-prediction-from-patient-features
Here are the trained models with codes for the paper "Prediction of Myocardial Infarction from Patient Features with Machine Learning".

The repository intends to predict the percentage of infarct or PMO (Persistent Myocardial Obstruction) in the left ventricle with only 12 pieces of clinical, paraclinical and physiological information.
The predictions can be realised with Random Forest Regressor to quantify the PIM (Percentage of Infarcted Myocardium), or with Random Forest Classifier to classify the presence of infarct or PMO.
The work can be found here "paper under review, will be avaible soon" and the employed data are illsutrated here: [Emidec: A Database Usable for the Automatic Evaluation of Myocardial Infarction from Delayed-Enhancement Cardiac MRI](https://www.mdpi.com/2306-5729/5/4/89) 

Four pre-trained models are provided and trained with 150 cases of patient information, while only the detailed information of the 100 cases is publicaly availble (the training set of EMIDEC dataset).

## How to use:

### Training:
```
# 1. Prepare your training data. The easiest way is following the sample template_test_sample.csv
# 2. Train your model by choosing the model. Modify the nessesary paths (inputs and outputs) in train.py. 
#    Be care of the target tissue (infarct or PMO) and the object (quantification or classification) and run:
python train.py
```
### Prediction:
```
# 1. Prepare your trained model and patient features to be used for your predicton. The easiest way is following the sample template_train_sample.csv
# 2. Modify the model and the .csv paths in prediction.py and run:
python prediction.py
# or pass the paths as arguments:
python prediction.py [path_model] [path_csv]
```
