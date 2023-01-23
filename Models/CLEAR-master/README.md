## 2 July 2021 Update

CLEAR has been updated to now include penalized logistic regression, and to also improve its handling of dummy variables. Nevertheless the recommended default for CLEAR remains to use (non-penalized) logistic regression with AIC.  The parameters specifying numeric/categorical data are now contained in CLEAR_settings.py . A second update will follow shortly in which CLEAR will switch from Tensorflow to PyTorch. Please let me know via email if there you have any requests for improved functionality. A new CLEAR paper has been published which applies CLEAR to images https://arxiv.org/abs/2106.14556 ; the code for this will be uploaded to GitHub once the paper is published.

# Counterfactual Local Explanations via Regression (CLEAR)

CLEAR explains single predictions of machine learning classifiers. It is based on the view that a satisfactory explanation of a single prediction needs to both
explain the value of that prediction and answer ’what-if-things-had-been-different’ questions. In doing this it needs to state the relative importance of the input features and show how they interact. A
satisfactory explanation must also be measurable and state how well it can explain a model. It *must know when it does not know*

Please note CLEAR is designed to explain an AI system's classification probabilities NOT classification classes

### Prerequisites

CLEAR is written in Python 3.7. It runs on Windows 10.The clear.yml file specifies CLEAR's dependencies .

### Installation

Download a copy of the CLEAR repository into a new directory on your PC. The file CLEAR_settings.py contains the parameter variables for CLEAR. Open CLEAR_settings and change the value of parameter *CLEAR_path* to the name of the directory you have created for CLEAR e.g. CLEAR_path='D:/CLEAR/'

### Running CLEAR

CLEAR's parameters for the experiment should first be set. These are all in CLEAR_settings.py. The admissible values for each parameter are shown in the comment to the right of the parameter eg for *case_study* the admissible values are 'Census','PIMA Indians Diabetes','Credit Card','BreastC'. The pdf file 'Input parameters for CLEAR' documents the input parameters.

CLEAR is then run by running CLEAR.py. The user has two options:
(a) run one of the sample models/datasets provided in CLEAR_sample_models_datasets.py .To do this CLEAR.py should include the command Run_CLEAR_with_sample_model()
(b) run CLEAR with a user created model and dataset. To do this CLEAR.py needs to have details of the user model and also include a command to run Run_CLEAR(). For example:
```python
if __name__ == "__main__":
    X_train = pd.read_pickle('D:/CLEAR/X_train_Adult')
    X_test_sample = pd.read_pickle('D:/CLEAR/X_test_sample_Adult')
    model = tf.keras.models.load_model('D:/CLEAR/CLEAR_Adult.h5')

```

CLEAR will generate a report explaining a single prediction if the parameters (in CLEAR.py) 'first_obs' and 'last_obs' are set to the same value e.g first_obs=7, last_obs=7 will generate a report explaining observation 7 in the test dataset. The report is entitled 'CLEAR_prediction_report.html'

There are two detailed csv files created for each run. The first file's name consist of the characters 'CLRreg_' and the date/time it was created eg 'CLRreg_20190522-1618.csv' This contains details of the regression for each observation e.g. adjusted R-squared score, coefficient weights and so forth. The second file's name consists of the characters 'wPerturb_' and the date time. This contains details of each b-perturbation for each observation. A error histogram is also created for each run, the name consisting of characters 'Hist_' and the date/time.
