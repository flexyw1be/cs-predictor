from skopt.space import Real, Categorical, Integer
from sklearn_genetic.space import Categorical as G_Categorical, Integer as G_Integer


BAYES_RF_SPACE = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(5, 30),
    'min_samples_leaf': Integer(1, 5)
}

BAYES_CB_SPACE = {
    'depth': Integer(4, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'iterations': Integer(100, 1000)
}


GENETIC_RF_SPACE = {
    'n_estimators': G_Integer(100, 500),
    'max_depth': G_Integer(5, 30),
    'min_samples_leaf': G_Integer(1, 5)
}