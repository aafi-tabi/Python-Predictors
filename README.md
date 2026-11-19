# Python Predictors

Small Python projects that use scikit-learn to predict or recommend things from simple data, such as study habits, skin type, or code statistics.

> **Note:** These projects use small demo datasets that I created, so the scores are for learning and practice, not real-world predictions.

## How This Was Built

I wrote the code for these projects myself, including the pipelines, models and evaluation. AI was used only for small help: expanding the demo datasets, adding the accuracy printouts, removing unused imports, and writing this README.

## Projects

| Project | What it does | Model | Result |
|---|---|---|---|
| [Student Success Predictor](student-success-predictor) | Predicts whether a student passes from study hours, sleep hours and attendance. Fills in missing values and reports accuracy, a confusion matrix and an ROC curve | Logistic Regression | 87.5% accuracy, ROC AUC 0.67 |
| [Bug Fix Success Predictor](bug-fix-success-predictor) | Predicts build success from lines of code and bugs fixed | Random Forest | 75% accuracy |
| [Skincare Routine Selector](skincare-routine-selector) | Suggests a skincare routine from skin type and age | Decision Tree | 100% accuracy on a 5-row test set |
| [Dress Style Recommender](dress-style-recommender) | Suggests a style from color and fabric scores, and compares 1, 2 and 3 neighbors | K-Nearest Neighbors | 100%, 67% and 67% accuracy |

## What I Practiced

- Building scikit-learn pipelines with scaling and one-hot encoding
- Handling missing values with an imputer
- Splitting data into training and test sets
- Checking results with accuracy, a confusion matrix and an ROC curve
- Comparing different settings of the same model

## Tech Stack

Python 3, scikit-learn, pandas, NumPy, Matplotlib

## Getting Started

```
pip install -r requirements.txt
cd student-success-predictor
python main.py
```

Each project folder has its own main.py, and you run it the same way.

## Limitations

The datasets are small, with 10 to 40 rows, so each test set has only a few rows. A high score here does not mean the model would work on real data. A next step is to repeat these projects with larger real datasets.

## License

MIT License
