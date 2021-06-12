# Car-Quality-Prediction-using-Machine-Learning
Machine Learning model using K-nearest neighbors to find the Quality of a car

In this project, a Machine Learning model is created using K-nearest neighbors algorithm from scikit-learn to predict the quality of cars.

The features used in this project are as follows:

buying       v-high, high, med, low (buying price of the car where low being the lowest price and v-high being the highest)
maint        v-high, high, med, low (maintenance price of the car where low being the lowest price and v-high being the highest)
doors        2, 3, 4, 5-more (Number of doors present in the car)
persons      2, 4, more (Seating Capacity of the car. More being seating capacity greater than or equal to 5)
lug_boot     small, med, big (Space for keeping luggage. Small being low and big being very high)
safety       low, med, high (Safety rating of the car with low being very poor and high being good)

Since our model can only be trained on datasets having features with numerical values, non-numerical values are first converted to numerical values and modified. 
Our target feature or label is the Quality of the car since it is what we want to find. Features are all the attributes mentioned above. 
Our model is then trained using K-nearest neighbors algorithm using training dataset. It is then tested for accuracy using the test dataset

Since we want to predict the Quality of an unknown dataset, input is then taken from the user for all the features from buying price to safety of the car in numbers. The predicted value of the quality of the car is then outputed as 'Very Poor','Average','Good','Excellent' where very poor is the least rating and Excellent being the best.

