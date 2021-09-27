# DentalData

#### About
This repository contains data cleaning, feature engeineering, and a basic model for predicting
missing payment values. 

#### Process
First, I cleaned the data removing some data with null or empty values. Second, I engineered a few new features including elapsed days, payment frequency, and average payment for user. I plotted many of these features to understand the overall and patient-level trends. 

Figure 1: All patient payments graphed over time.
![all_data](https://user-images.githubusercontent.com/62564888/134968711-0b0d27b3-7e7c-4d04-8de4-bfd0bf4966a6.png)

I found predicting missing cells in the data an interesting regression problem. Using the features I had engineered, I created a simple feed forward neural network that originally accepted seven features, including time data, to make predicitions. However, looking at the data, I determined that time data did not seem to have as great of an effect on patient payment amounts as the user's payment data. I reduced the input features to just the user's average payment amount and number of payments and got better predictions.

Figure 2: Predictions of payments from the neural network with all other patient payment
![all_pres](https://user-images.githubusercontent.com/62564888/134969383-ea10485a-8855-4e40-9e20-a208e8284a61.png)

Figure 3: A payment prediction for patient 6
![prediction](https://user-images.githubusercontent.com/62564888/134969485-ab9137a1-1cc9-4b7b-a76a-e8b89ece7404.png)
