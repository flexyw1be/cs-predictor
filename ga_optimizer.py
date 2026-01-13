import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous, Categorical
from config import GA_PARAMS


def run_genetic_optimization(X_train, y_train):
    """
    Запускает генетический поиск гиперпараметров.
    """
    print("Starting Genetic Algorithm optimization...")

    # Базовая модель
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Пространство поиска (Гены)
    # GA будет "мутировать" эти значения, чтобы найти лучшие
    param_grid = {
        'n_estimators': Integer(50, 500),  # Кол-во деревьев
        'max_depth': Integer(3, 30),  # Глубина дерева
        'min_samples_split': Integer(2, 20),  # Минимум примеров для разделения
        'min_samples_leaf': Integer(1, 10),  # Минимум примеров в листе
        'criterion': Categorical(['gini', 'entropy']),
        'max_features': Categorical(['sqrt', 'log2'])
    }

    # TimeSeriesSplit важен! Мы не можем валидироваться случайно,
    # нужно учиться на прошлом и проверять на будущем внутри выборки.
    cv_split = TimeSeriesSplit(n_splits=3)

    # Настройка эволюции
    evolved_estimator = GASearchCV(
        estimator=clf,
        cv=cv_split,
        scoring='accuracy',  # Или 'roc_auc'
        param_grid=param_grid,
        population_size=GA_PARAMS['population_size'],
        generations=GA_PARAMS['generations'],
        crossover_probability=GA_PARAMS['crossover_probability'],
        mutation_probability=GA_PARAMS['mutation_probability'],
        verbose=GA_PARAMS['verbose'],
        keep_top_k=2,
        n_jobs=GA_PARAMS['n_jobs']
    )

    # Запуск эволюции
    evolved_estimator.fit(X_train, y_train)

    print("\nBest Parameters found by GA:")
    print(evolved_estimator.best_params_)

    return evolved_estimator.best_estimator_