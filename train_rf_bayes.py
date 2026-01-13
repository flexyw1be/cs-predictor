import pandas as pd
import joblib
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config import BAYES_RF_SPACE

df_train = pd.read_csv('data/processed/train.csv')
feature_cols = ['map', 'rank_diff', 'abs_rank_diff', 'picked_by_is_A', 'is_decider', 'map_winrate_A', 'map_winrate_B', 'recent_form_A', 'recent_form_B', 'elo_diff', 'map_elo_diff']
X, y = df_train[feature_cols], df_train['winner_is_A']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), [c for c in feature_cols if c != 'map']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['map'])
])

X_proc = preprocessor.fit_transform(X)
opt = BayesSearchCV(RandomForestClassifier(random_state=85), BAYES_RF_SPACE, n_iter=25, cv=GroupKFold(n_splits=5), n_jobs=-1)
opt.fit(X_proc, y, groups=df_train['match_id'])

joblib.dump(opt.best_estimator_, 'models/rf_bayes.pkl')
print(f"RF Bayes Accuracy: {opt.best_score_:.4f}")