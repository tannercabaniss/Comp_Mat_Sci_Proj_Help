# Data
The data required for these models is designed to be in plain text formatting. This includes .txt file such as the
[data.txt](data.txt) file in this repository or any other plain text file such as the [statistics.dat](statistics.dat) file provided in this repository.
While the two forementioned data files represent a near minimum number of samples required to make accurate estimations
with this model, they were helpful in tuning the models to their current state.

I would suggest running a multitude (n > 10) of simulations spanning an appropriate range of potentials for each element 
(e.g., -2 -> +2) to get a baseline reading for where to center your search. At this point you can use the result to
make the range more consise. The most important part of drawing the initial data set is discovering combinations of
potentials which result in higher and lower concentrations than the desired target for each element (e.g., if the alloy
contains 3 elements the target would 0.333 for each element's resulting concentration so find potentials which result
in concentrations greater and less than 0.333 for each respective element in the alloy).

Once the initial simulations have been run and the narrowed range for potentials has been determined, I would suggest,
depending on the narrowness of the potentials' search range, taking between 30 and 60 more data points to generate a model
which adequately generalizes the relationships between the chemical potentials and the resulting concentrations. Again,
see the data file 'data.txt' that I have provided for example of the minimum which may be required for accurate results.
The more centered your data points are around the target, the better the model will learn, and the more accurate the
predictions will be.

# Models
After rounds of refining and tuning, I determined the two best approaches were random forest and a variation of gradient boosting.
While these approaches contain models for classification as well as regression, we will only be using the latter in this case.
Please see the following links for more information about those approaches if you wish.
* Random Forest: https://builtin.com/data-science/random-forest-python
* Xtreme Gradient Boosting: https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/

## Random Forest Regressor (RFR)
I have included two different RFR models in the relevant source code file [RFR](HfNbTi_Potentials_RandomForest.py).
The first is a base model which has default hyperparameters and does not utilize cross validation during training.
The second is a model which uses halving random search with cross validation to test many hyperparameter combinations
and determine the optimal combination.

While I have left some comments in the source code, I will move into a more elaborate explanation here for using and
altering the model as needed. Much of this will be similar conceptually for both models, but I will make sure to
reiterate everything so you don't have to repeatedly switch sections with in this document.

### Importing Libraries (Lines 1-9)
If you opt to use the anaconda distribution package, many of these libraries are already included. But if you run into
'module not found' errors, you will need to install the relevant libraries using pip install. If you are using a 
plain distribution, you will need to install all of the libraries via pip. Please see the following
link for more information about using pip install.
* https://packaging.python.org/en/latest/tutorials/installing-packages/

### Loading Data (Lines 12-23)
The first step in the models is to load in the data from the data file into a data frame. I would suggest placing your
data file as well as the two example data files in the same directory as the source code as is done in this repo to
make the file path short and easily changable. On line 12 you can change the file name from 'data.txt' to the one which
contains your relevant data.

As long as your data is in the space delimited format shown in the example data files,
line 13 can be left alone. Lines 15 and 16 serve to ensure that the data within the data frame are float values just
in case the import function converted the plain text to string objects.

Lines 19 and 20 are very important for making sure the data is properly seperated (i.e., chemical potentials are input 
and concentrations are output). For the example case provided, there were 2 chemical potentials used as inputs in the
simulations and 3 concentrations received as outputs. The X variable in this case are the chemical potentials. So,
if you have a situation with say 3 chemical potentials, you would need to change the index range from 
[:, :2] to [:, :3]. The y variable is the target or output concentrations from the data set. To adjust this in the case
of more potentials or concentrations you take the X indice arrangement and flip the index number and the colon (i.e., 
X = [:, :3] -> y = [:, 3:]).

Line 23 breaks apart the input and output values into training and testing sets. This allows the model to be evaluated
on data it has not learned from which is helpful in seeing how well the model has generalized to the underlying
relationships of the data. The test_size parameter can be increased to put a greater proportion of the data in the
test set or decreased to put a lesser proportion of the data in the test set. 0.2 is a widely used proportion and has
shown good results thus far so I would leave changing this as a last resort. A random state is also set in this
function which will make sure the data is split in the same way each time so the model is more easily tuned.

### Hyperparameter Grid (Lines 26-32)
The hyperparameter grid contains all the hyperparameter which can be evaluated during the random search tuning process.
Please see the following link for more information about each hyperparameters function for the RFR model:
https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

Their main functionalities are outlined below:
* n_estimators: controls the number of trees in the forest. while generally more is better, you will lose model sensitivity if the value is too high as well as greatly increasing the computational cost of training the model.
* max_depth: determines the number of branches which can form on each tree. Too high of a number here will increase computational cost and can overfit the model. The 'none' option allows the trees to branch until the minimum samples per leaf is reached.
* min_samples_split: determines the minimum number of samples on a leaf for a split (branch) to occur. This works hand in hand with max_depth in regulating the complexity and compuational cost of the model. This number must be greater than 2.
* min_samples_leaf: determines the minimum number of samples which can left on leaf following a split. This number must be greater than 1.
* bootstrap: if true, the training data will be randomly sampled seperately for each tree. If false, the entire training dataset will be used for each tree. Setting false, can be helpful if underfitting is occuring, while true, can be helpful for preventing overfitting.

### Model Setup and Fitment (Lines 35-45)
The base model (line 35) should not be altered as a baseline is essential for determining the effects of hyperparameter
tuning. The tuned model (line 38) utilizes halving random search with cross validation as previously described. The
'estimator' and 'param_distributions' should not be altered. The scoring used here is the mean squared error. This
is the default scoring method and should be adequate for this application. The verbose parameter determines how much
print output there is as the model is running. You can change it to 0 if you don't want any printout, but I would 
suggest against altering it to 2 as that results in lots of excess print output that is not necessary. The important
varialbes in this function are factor and cv. Factor adjusts the split reduction of the random search mechanism. A 
larger number will result in less total fits which can help reduce computational cost. However, I have found that the
minimum value, 2, generates the best results for the small datasets with which we are working. CV determines the number of cross validation folds taken from the training dataset. While the typical value is 5, we simply don't have
enough data to adequately train the model when only looking at a fifth of the data (~10 data points). As is such, the minimum value, 2, has worked best thus far, but experimentation can be done with larger values if necessary.

The models are fitted to the training data in lines 42 and 45 for the base and random search models respectively. No
changes will be needed here.

### Predictions and Evaluation (Lines 48-104)
This section of code uses the models trained in the previous section to predict the output values given input values
on the testing portion of the dataset. Then evalution metrics are drawn from the models performance on both the
training and testing datasets with a print output of these metrics in table form. No modificiations should be needed
in this section.

### Inverse Predictions for Determining Chemical Potentials (Lines 106-136)
This section is used to inversely predict inputs (chemical potentials) from a predescribed output (equal 
concentration). Two different modifications which may be needed in this section are the 'method' for the minimize
function and the target value. The minimize function determines the input values which result in the minimization
of the objective function (lines 106-107). The minimize function itself uses a specific method for determining
which values to try next on its path towards minimization. In the current code, this method is 'Nelder-Mead', but 
many other options are avaiable and may be better in minimizing the objective function. Please see the link
for more information: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

The target value is the equal concentration of each element in the alloy. This is determined as part of the objective
function in line 107 and in the desired_output variable in line 110. If you are working with a 3 element alloy, as the
one in the example data, these values will all be 0.333. If you are using a 4 element alloy with 4 resulting
concentrations, this value would need to be changed to 0.25 everywhere there is as 0.333 in lines 107 and 110. If a
5 element alloy is being evaluated the value would be changed to 0.2 and so on.

### Tuning the Model
Although I touched on this subject recently, I will discuss the process of tuning the model in more detail here.
After running the model for the first time, you will get a table of the metrics from the base and random search models
for their respective performance on the training and testing datasets. There are two trends which you are likely to
see: overfitting and underfitting. Overfitting is when the model fits too aggressively to the training dataset and
does not generalize to the testing dataset. This is seen as the evalution metrics for the training data being much
better than those of the testing data. Typically, this can be avoided by altering the parameter ranges
for the hyperparameters. It can take many iterations to get this right but use small changes and keep track of the
previous state in case the modifications made do not have the desired effect.

Underfitting is shown as the both the training data and testing data having poor evalution metrics such as high
mean squared error values or low R-squred values. Underfitting can be solved by increasing max depth or reducing min
samples split or min samples leaf. However, if these modifications are already maxed out (i.e., max_depth = None, 
min_sample_split = 2, and min_samples_leaf = 1) then often times more datapoints are needed or the dataset needs
refinement (e.g., remove values that do not provide adequate information such as concentrations of 0).

## Xtreme Gradient Boosting (XGB)
I have included two different XGB models in the relevant source code file [XGB](HfNbTi_Potentials_XGBoost.py).
The first is a base model which has default hyperparameters and does not utilize cross validation during training.
The second is a model which uses halving random search with cross validation to test many hyperparameter combinations
and determine the optimal combination.

While I have left some comments in the source code, I will move into a more elaborate explanation here for using and
altering the model as needed. Much of this will be similar conceptually for both models, but I will make sure to
reiterate everything so you don't have to repeatedly switch sections with in this document.

### Importing Libraries (Lines 1-9)
If you opt to use the anaconda distribution package, many of these libraries are already included. But if you run into
'module not found' errors, you will need to install the relevant libraries using pip install. If you are using a 
plain distribution, you will need to install all of the libraries via pip. Please see the following
link for more information about using pip install.
* https://packaging.python.org/en/latest/tutorials/installing-packages/

### Loading Data (Lines 12-23)
The first step in the models is to load in the data from the data file into a data frame. I would suggest placing your
data file as well as the two example data files in the same directory as the source code as is done in this repo to
make the file path short and easily changable. On line 12 you can change the file name from 'data.txt' to the one which
contains your relevant data.

As long as your data is in the space delimited format shown in the example data files,
line 13 can be left alone. Lines 15 and 16 serve to ensure that the data within the data frame are float values just
in case the import function converted the plain text to string objects.

Lines 19 and 20 are very important for making sure the data is properly seperated (i.e., chemical potentials are input 
and concentrations are output). For the example case provided, there were 2 chemical potentials used as inputs in the
simulations and 3 concentrations received as outputs. The X variable in this case are the chemical potentials. So,
if you have a situation with say 3 chemical potentials, you would need to change the index range from 
[:, :2] to [:, :3]. The y variable is the target or output concentrations from the data set. To adjust this in the case
of more potentials or concentrations you take the X indice arrangement and flip the index number and the colon (i.e., 
X = [:, :3] -> y = [:, 3:]).

Line 23 breaks apart the input and output values into training and testing sets. This allows the model to be evaluated
on data it has not learned from which is helpful in seeing how well the model has generalized to the underlying
relationships of the data. The test_size parameter can be increased to put a greater proportion of the data in the
test set or decreased to put a lesser proportion of the data in the test set. 0.2 is a widely used proportion and has
shown good results thus far so I would leave changing this as a last resort. A random state is also set in this
function which will make sure the data is split in the same way each time so the model is more easily tuned.

### Hyperparameter Grid (Lines 26-33)
The hyperparameter grid contains all the hyperparameter which can be evaluated during the random search tuning process.
Please see the following link for more information about each hyperparameters function for the XGB model:
https://xgboost.readthedocs.io/en/stable/parameter.html

Their main functionalities are outlined below:
* n_estimators: controls the number of trees in the forest. while generally more is better, you will lose model sensitivity if the value is too high as well as greatly increasing the computational cost of training the model.
* learning_rate: determines the rate at which the model is adjusted when encountering different trends. Too high of a rate can overfit the model to specific parts of the dataset and not as well to others. A very low rate will not allow the model to be trained effectively, especially on a small dataset.
* max_depth: determines the number of branches which can form on each tree. Too high of a number here will increase computational cost and can overfit the model. The 'none' option allows the trees to branch until the minimum samples per leaf is reached.
* min_child_weight: functions similarly to min_samples_split as described in the RFR section. This works hand in hand with max_depth in regulating the complexity and compuational cost of the model. This number must be greater than 1.
* colsample_bytree: determines the proportion of features (input potentials) to be analyzed by a given tree. In this case, 1 would be looking at all of the input potentials and 0.5 would be looking at half of them for a given tree. While there is a range listed here, for the limited features of our data, 1 is probably the best option.
* gamma: controls the loss redcution of the branches. This means that if there is inaccuracy greater than the gamma value a node can be split (branched), and if not, no split will occur. A lower value will aid in fitting the model better, but too low of a model can cause overfitting. Too high of a value is likely to cause underfitting as the refinement of the model is greatly limited.

### Model Setup and Fitment (Lines 36-46)
The base model (line 36) should not be altered as a baseline is essential for determining the effects of 
hyperparameter tuning. The tuned model (line 39) utilizes halving random search with cross validation as previously 
described. The 'estimator' and 'param_distributions' should not be altered. The scoring used here is the negative 
mean squared error. This scoring method can be altered if the model is not performing well. Please see the previous 
documentation in the hyperparameter section for more information. The verbose parameter 
determines how much print output there is as the model is running. You can change it to 0 if you don't want any 
printout, but I would suggest against altering it to 2 as that results in lots of excess print output that is not 
necessary. The important varialbes in this function are factor and cv. Factor adjusts the split reduction of the 
random search mechanism. A larger number will result in less total fits which can help reduce computational cost. 
However, I have found that the minimum value, 2, generates the best results for the small datasets with which we are 
working. CV determines the number of cross validation folds taken from the training dataset. While the typical value 
is 5, we simply don't have enough data to adequately train the model when only looking at a fifth of the data (~10 
data points). As is such, the minimum value, 2, has worked best thus far, but experimentation can be done with larger 
values if necessary.

The models are fitted to the training data in lines 43 and 46 for the base and random search models respectively. No
changes will be needed here.

### Predictions and Evaluation (Lines 49-105)
This section of code uses the models trained in the previous section to predict the output values given input values
on the testing portion of the dataset. Then evalution metrics are drawn from the models performance on both the
training and testing datasets with a print output of these metrics in table form. No modificiations should be needed
in this section.

### Inverse Predictions for Determining Chemical Potentials (Lines 107-136)
This section is used to inversely predict inputs (chemical potentials) from a predescribed output (equal 
concentration). Two different modifications which may be needed in this section are the 'method' for the minimize
function and the target value. The minimize function determines the input values which result in the minimization
of the objective function (lines 107-108). The minimize function itself uses a specific method for determining
which values to try next on its path towards minimization. In the current code, this method is 'Nelder-Mead', but 
many other options are avaiable and may be better in minimizing the objective function. Please see the link
for more information: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

The target value is the equal concentration of each element in the alloy. This is determined as part of the objective
function in line 107 and in the desired_output variable in line 110. If you are working with a 3 element alloy, as the
one in the example data, these values will all be 0.333. If you are using a 4 element alloy with 4 resulting
concentrations, this value would need to be changed to 0.25 everywhere there is as 0.333 in lines 108 and 111. If a
5 element alloy is being evaluated the value would be changed to 0.2 and so on.

### Tuning the Model
Although I touched on this subject recently, I will discuss the process of tuning the model in more detail here.
After running the model for the first time, you will get a table of the metrics from the base and random search models
for their respective performance on the training and testing datasets. There are two trends which you are likely to
see: overfitting and underfitting. Overfitting is when the model fits too aggressively to the training dataset and
does not generalize to the testing dataset. This is seen as the evalution metrics for the training data being much
better than those of the testing data. Typically, this can be avoided by altering the parameter ranges
for the hyperparameters. It can take many iterations to get this right but use small changes and keep track of the
previous state in case the modifications made do not have the desired effect.

Underfitting is shown as the both the training data and testing data having poor evalution metrics such as high
mean squared error values or low R-squred values. Underfitting can be solved by increasing max depth or reducing min
samples split or min samples leaf. However, if these modifications are already maxed out (i.e., max_depth = None, 
colsample_bytree = 1, min_child_weight = 1, and gamma = 0.0001) then often times more datapoints are needed or the 
dataset needs refinement (e.g., remove values that do not provide adequate information such as concentrations of 0).
