# Recipes and Ratings Nutritional Statistical Analysis

Recipes and Ratings Nutritional Statistical Analysis is a comprehensive data science project conducted at UCSD. The project consists of data wrangling, various analysis including EDA and hypothesis testing, creation of baseline and final models, and concluding with fairness analysis. The primary focus of the project is to investigate the nutritional values in recipes, seeing if they differ by factors such as season and preparation time, as well as to predict the calories count for recipes using the nutritional values.

By: Kang Lee, Jacob Kavanal

## Introduction

### General Introduction

Every recipe comes with various information such as its ingredients, steps, preparation time, and nutrition. Nutritional information for recipes comes in the form of calories, total fat, sugar, sodium, protein, saturated fat, and carbohydrates. This information is often a good indicator of how healthy a recipe is. Recipes with a good amount protein, calories, and carbohydrates are generally healthy while ones with a lot of sodium, sugar, and saturated fat may cause weight gain or high blood pressure.

Our project consists of two datasets: Recipes and Ratings from food.com. It was originally scraped and used by the authors of a recommender systems paper. Our datasets are only a subset of the raw data used in the original report since the original data is quite large. The Recipes dataset has over 80,000 entries while the Ratings dataset has over 700,000 entries.

We left merged the two datasets and filled all ratings of `0` with `np.nan`. Ratings of 0 are likely missing or placeholder data, probably for recipes with no ratings. We replaced the 0s with `np.nan` to avoid skewing the average rating for recipes and for more accurate analysis. Then, we found the average rating per recipe, as a Series. Finally, we left merged the resulting Series back into the Recipes dataset.

The primary question we aimed to answer was: **Can we predict the calories of recipes by training a regression model on the other nutritional values?** This question interested us since we care about the nutritional value of the food we cook and consume. We were also interested in seeing how accurate the calorie values of these recipes are across the dataset in relation to other nutritional values. We also want to use data analysis techniques to see how nutritional values differ by yearly seasons and cooking time.

### Introduction of Columns

The dataset introduces several arrays of columns featuring recipe information from thousands of recipes posted since 2008. There are 83,782 rows in the dataset, and here is a description of the key columns:

- `name`: This column represents the name of the recipe.

- `id`: This column represents the unique identifier for each recipe. It allows us to distinguish between different recipes in the dataset.

- `minutes`: The 'minutes' column represents the number of minutes it takes to prepare that recipe.

- `contributor_id`: This column represents the unique identifier for the person that posted this recipe. People may post multiple recipes.

- `submitted`: This column represents the date the recipe was posted. It is in the format: year-month-day.

- `tags`: The 'tags' column represents a list of tags that makes the recipes easily filterable for people looking for a specific type of recipe such as 'american' or 'chinese'.

- `nutrition`: The 'nutrition' column represents a list of nutritional information in the form: [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value”.

- `n_steps`: This column represents the number of steps the recipe requires.

- `steps`: The 'steps' column represents a list the steps needed to prepare to recipe.

- `description`: The 'description' column represents a description or comment of the recipe uploaded by the contributor.

- `ingredients`: The 'ingredients' column represents a list of ingredients needed to prepare the recipe.

- `n_ingredients`: This column represents the number of ingredidents in the recipe.

- `avg_rating`: The 'avg_rating' column represents average rating of the recipe as a float between 0 and 5.


## Data Cleaning and Exploratory Data Analysis

 To perform analysis on the nutritional values seperately, we split the nutrition column to 6 columns. These columns are `total fat (PDV)`, `saturated fat (PDV)`, `sugar (PDV)`, `sodium (PDV)`, `protein (PDV)`, `carbohydrates (PDV)`. PDV means Percent of Daily Value which is the recommended daily intake per person.

## Assessment of Missingness

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

## Hypothesis Testing

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

## Framing a Prediction Problem

After cleaning/exploring the dataset and analyzing missingness, distributions, and more, we decided to aim to predict calorie value from other nutritional values. At the time of prediction, we will have all the data necessary to predict this value. This is a multiple regression problem whose response variable is 'calorie'. We plan to use Mean Squared Error and R<sup>2</sup> for the following reasons:

1. MSE directly measures prediction accuracy by penalizing larger errors more heavily, which is important when predicting nutritional values where significant deviations could be problematic.
2.  R² indicates how well the other nutritional values explain the variance in calories, providing insight into the relationship's strength.
3. These metrics are more interpretable than alternatives like MAE or RMSE for nutrition prediction, as they relate directly to explained variance and error magnitude in calorie predictions.
