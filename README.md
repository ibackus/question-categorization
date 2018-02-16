# question-categorization
Description

#### Dependencies
* python3
* numpy/scipy
* pandas
* scikit-learn
* gensim
* xgboost

## The problem
[Dataset](http://cogcomp.org/Data/QA/QC/)

## My models
I tried several different feature extractors and classifiers.  Here I present results using a bag of words feature extractor and two classifiers: extra trees and extreme gradient boosted trees (XGBoost).  The Extra-Trees classifiers were fast to train nd therefore useful for prototyping and for selecting good hyperparameters for the bag of words model.  The XGBoost classifier was then later trained to develop my final fiducial model.

### Other models
I also toyed around with using Word2Vec (`word2vec.py`) as a feature extractor, training it on just the corpus of words present in the questions.
Sentences were mapped to a vector space by averaging the word vectors in the sentence and treating words not present in the corpus as zero vectors.
However, this faired very poorly (about 0.68 accuracy for the coarse categories) and was therefore abandoned.
Using vectors trained on a more complete corpus may have faired better.

### Scripts - training the models
The models can all be trained by running the python 3 `train_all_models.py` script.
Note that this is *slow* and is not necessary, since the trained models are saved to disk in the `results` folder.
This slowness comes from a scan over hyperparameter space.
These models could be further scaled up with more data without much need to perform this scan again.

This script just runs several training scripts: `coarse_training.py`, `fine_training.py`, and `xgb_training.py`.
The first two train Extra Trees models and the third trains an XGBoost model.
The extra trees models train rapidly and were therefore used first for prototyping and exploring the data and models.

The `coarse_training.py` script runs over several grid searches of hyperparameter space for the bag of words feature selector and the extra trees classifier, which are used to inform and narrow down future grid searches.
Sets of hyperparameters are evaluated using a 5-fold cross validation score, using the matthews correlation coefficient to score them.

The `fine_training.py` script does a similar grid search to determine good hyperparameters for an extra trees model trained on the fine categories.

Finally, the `xgb_training.py` script uses the results of the previous training steps to inform the hyper parameter space search to find good hyper parameters for the final XGBoost classifier.
Since the hyper-parameter space for XGBoost is larger and since training the models takes longer, a random search over hyperparameters was used in place of an exhaustive grid search.

## Analyzing the trained models
The trained models are stored in `results/`.
These can be analyzed by running the python 3 `analyze.py` script which will report some scores and statistics on how the models perform.

## Results
For a full results summary, run the analysis script or see the output from `analyze.py` in `results/analysis.out`.
In brief:

| Model       | Category | Accuracy | Baseline Acc | MCC   |
|-------------|----------|----------|--------------|-------|
| Extra trees | coarse   | 0.846    | 0.230        | 0.809 |
|             | fine     | 0.776    | 0.176        | 0.751 |
| XGBoost     | coarse   | 0.87     | 0.230        | 0.837 |
|             | fine     | 0.79     | 0.176        | 0.766 |

## Contents
### Files
* `analyze.py` - Script to analyze trained models
* `coarse_training.py` - Script to train an Extra Trees model on the coarse categories
* `fine_training.py` - Script to train an Extra Trees model on the fine categories
* `train_all_models.py` - Wrapper script to run all training scripts
* `utils.py` - Contains utility function definitions for this project
* `word2vec.py` - A test script to try word2vec feature selection.  This was not very successful
* `xgb_training.py` - Script to train an XGBoost model
### Folders
* `data` - Contains the test/training datasets
  * `train_5500.label` - Training dataset
  * `TREC_10.label` - Test dataset
* `results` - Contains results of analysis and the trained models
* `scratch` - A folder with a few pieces of scratch-work that didn't go too far
