# Data
The data required for these models is designed to be in plain text formatting. This includes .txt file such as the
data.txt file in this repository or any other plain text file such as the statistics.dat file provided in this repository.
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
I have included two different RFR models in the relavant source code file [](HfNbTi_Potentials RandomForest.py)
