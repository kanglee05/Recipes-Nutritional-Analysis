# Recipes and Ratings Nutritional Statistical Analysis

by: Kang Lee, Jacob Kavanal

# Introduction

We chose to perform our analysis on 'Recipes and Ratings' a dataset containing 83782 recipes as well as thousands of ratings on these recipes. The 'recipes' dataframe contains a wide variety of data on recipes like 'nutrition', 'ingredients', 'steps', 'time submitted', 'tags', and 'avg rating.' The question we aimed to answer was: "Can we predict calories only by training a regression model on the other nutritional values?" This question interested us as people who care about the nutritional value of the food we make and eat. We were also interested in seeing how accurate the calorie values of these recipes are across the dataset in relation to other nutritional values. To perform analysis on the nutritional values seperately, we split the nutrition column to 6 columns. These columns are 'total fat (PDV)', 'saturated fat (PDV)', 'sugar (PDV)', 'sodium (PDV)', 'protein (PDV)', 'carbohydrates (PDV).'

# Data Cleaning and Exploratory Data Analysis

TODO

# Assessment of Missingness

After performing some analysis on the 'Avg Rating' column of our dataframe while assessing missingness, we found that the average time taken for a recipe with a missing 'avg rating' was 228 minutes. Alternatively, the average time for a recipe for which 'avg rating' was NOT missing was 111 minutes. Seeing this difference, we performed a permutation test to assess whether 'avg rating' was really dependent on this column. 

We found, with a p-value of .04, that missingness of 'avg rating' was dependent on 'minutes' with an average observed difference of 117. With a threshold of .05, we can reject the null hypothesis that missingness is not dependent on minutes. This makes the 'avg_rating' column MAR. 

With a p-value of .1, we also found that missingness is likely not dependent on protein value. With a threshold of .05 again, we fail to reject the null hypothesis that missingess of 'avg rating' is not dependent on protein value. 

Below is a graph displaying the proportion of missing values with respect to minutes. We can see that missingess increases by a factor of 2 as recipes get longer.

<iframe
  src="plots/missingessbyminutes.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

# Hypothesis Testing

Hypotheses:
- Null hypothesis: Protein values in recipes submitted during cold months are the same as recipes submitted in warm months.
- Alternative hypothesis: Protein values in recipes submitted during cold months are different than recipes submitted during warm months.
- Test statistic: Absolute difference between protein value averages of warm months and cold months.
- P-value Threshold: 0.05

We performed a permutation test to find the differences in the distributions of protein values in cold and warm months. Warm months were classified as March-September and cold months were all others. We found an observed difference of .3 grams and a p-value of .11. With our threshold of .05, we fail to reject the null hypothesis. We chose this pair of hypotheses to test because of personal interest in this question as well as its applicability in helping us greater understand the ditributions of nutritional values in this dataset. 

Below is a boxplot displaying the distribution of protein in warm vs. cold months. It highlights the lack of difference between the two seasons. 
<iframe
  src="plots/proteinbyseason.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

# Framing a Prediction Problem

After cleaning/exploring the dataset and analyzing missingness, distributions, and more, we decided to aim to predict calorie value from other nutritional values. At the time of prediction, we will have all the data necessary to predict this value. This is a multiple regression problem whose response variable is 'calorie'. We plan to use Mean Squared Error and R<sup>2</sup> for the following reasons:

1. MSE directly measures prediction accuracy by penalizing larger errors more heavily, which is important when predicting nutritional values where significant deviations could be problematic.
2.  R² indicates how well the other nutritional values explain the variance in calories, providing insight into the relationship's strength.
3. These metrics are more interpretable than alternatives like MAE or RMSE for nutrition prediction, as they relate directly to explained variance and error magnitude in calorie predictions.
