# TrackPopularityPredictor

Project for the ***Statistical Methods for Machine Learning*** course (2023 edition) held by Nicolò Cesa-Bianchi at *University of Milan*

### (Kernel) Ridge Regression

Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and perform ridge regression to predict the tracks’ popularity.<br>
Note that this dataset contains both numerical and categorical features.<br> 
The student is thus required to follow these guidelines:
- first, train the model using only the numerical features,
- second, appropriately handle the categorical features (for example, with one-hot encoding or other techniques) and use them together with the numerical ones to train the model,
- in both cases, experiment with different training parameters,
- use 5-fold cross validation to compute your risk estimates,
- thoroughly discuss and compare the performance of the model

The student is required to implement from scratch (without using libraries, such as Scikit-learn) the code for the ridge regression, while it is not mandatory to do so for the implementation of the 5-fold cross-validation.

**Optional**: Instead of regular ridge regression, implement kernel ridge regression using a Gaussian kernel.
