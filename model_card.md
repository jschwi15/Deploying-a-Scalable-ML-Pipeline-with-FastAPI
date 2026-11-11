# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was developed by Justin Schwinn in November 2026 as part of the Udacity / WGU "Deploying a Scalable ML Pipeline with FastAPI" project. It is a Random Forest classifier built with scikit-learn 1.5.1 on Python 3.9. The model uses 100 trees with a maximum depth of 20, a minimum of 2 samples per leaf, and a random state of 42 for reproducibility. Categorical features are one-hot encoded and the label is binarized before training.

An earlier version of the model used unlimited tree depth. Capping the depth and leaf size reduced the saved model from 70MB to 15MB and improved the F1 score from 0.6830 to 0.6926, which suggests the uncapped trees were overfitting the training data.

## Intended Use

The model predicts whether a person's annual income is above or below $50K based on demographic and employment information from census data. It is intended for educational purposes, specifically to demonstrate building, testing, and deploying an ML pipeline with CI/CD and a REST API. It is not intended for real-world decisions about individuals, such as hiring, lending, housing, or eligibility for services.

## Training Data

The model was trained on the UCI Adult Census Income dataset (census.csv), which was extracted from the 1994 U.S. Census database. The full dataset has 32,561 rows and 15 columns: 14 features and the salary label. The features include 6 numeric columns (age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical columns (workclass, education, marital-status, occupation, relationship, race, sex, native-country).

The data was split 80/20 into training and test sets using a stratified split on salary with a random state of 42. This gave 26,048 training rows. The classes are imbalanced, with about 76% of people earning <=50K and 24% earning >50K.

The dataset has no true null values, but missing values are recorded as "?" in three columns: workclass (1,836 rows), occupation (1,843 rows), and native-country (583 rows). These were left in place and treated as their own category by the encoder.

## Evaluation Data

The model was evaluated on the 20% held-out test set, which contains 6,513 rows. Because the split was stratified, the test set keeps the same 76/24 class balance as the full dataset. The test set was processed using the encoder and label binarizer fit on the training data only, so no information from the test set leaked into training.

## Metrics

Because the classes are imbalanced, accuracy would be misleading, since a model that always predicts <=50K would already be about 76% accurate. The model is evaluated on precision, recall, and F1 score instead, with >50K as the positive class.

Overall performance on the test set:

- Precision: 0.7878
- Recall: 0.6180
- F1: 0.6926

In plain terms, when the model predicts someone earns more than $50K, it is right about 79% of the time, but it only identifies about 62% of the people who actually earn more than $50K.

Performance was also measured on slices of the data for every value of every categorical feature, and the full results are saved in slice_output.txt. A few notable results:

- sex: Female (2,158 rows) had an F1 of 0.6888 and recall of 0.5918, while sex: Male (4,355 rows) had an F1 of 0.6933 and recall of 0.6228.
- race: White (5,533 rows) had an F1 of 0.6960. race: Black (662 rows) had an F1 of 0.6538 with recall of 0.5484. race: Amer-Indian-Eskimo (73 rows) had an F1 of 0.4615 with recall of only 0.3333.
- education: 10th (204 rows) had a precision of 1.0 but recall of only 0.2, meaning the model rarely predicts >50K for people without a high school diploma.
- workclass: ? (363 rows), the missing value group, had an F1 of 0.5000, noticeably worse than most other groups.

## Ethical Considerations

The dataset includes sensitive attributes such as race, sex, marital status, and native country, and the model uses them as input features. This means the model can learn and repeat historical patterns of inequality from 1994 rather than anything about a person's actual earning potential.

The slice results show uneven performance across groups. Recall is lower for women than men, lower for Black individuals than white individuals, and much lower for the Amer-Indian-Eskimo group. In practice, this means the model is more likely to miss high earners in these groups. Any real-world use of a model like this could reinforce existing disparities, which is why it should not be used to make decisions about people.

## Caveats and Recommendations

The data is from 1994 and the $50K threshold is not adjusted for inflation, so the model does not reflect current incomes or the current workforce. Several slices are too small to trust. For example, workclass: Never-worked, workclass: Without-pay, and race: Other show perfect scores, but they are based on as few as 1 to 45 people.

Recommendations for improving the model:

- Try hyperparameter tuning or cross-validation instead of a single train/test split.
- Handle the "?" missing values more deliberately, such as imputing them or flagging them as a separate feature.
- Address class imbalance with class weights or resampling to improve recall.
- Test whether removing sensitive features like race and sex changes performance or reduces the gaps between groups.
- Retrain on more recent census data before drawing any conclusions about today's population.
