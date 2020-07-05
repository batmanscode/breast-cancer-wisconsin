import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('cleaned_data.csv')
features = tpot_data.drop('diagnosis', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['diagnosis'], random_state=42, test_size=0.2)

# Average CV score on the training set was: 0.9670329670329672
exported_pipeline = make_pipeline(
    StandardScaler(),
    RBFSampler(gamma=0.8500000000000001),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.55, min_samples_leaf=10, min_samples_split=15, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
