from skopt.space import Real, Categorical, Integer
from genetic_algorithm import IntegerParam, RealParam, CategoricalParam


# ============================================================
# BAYES OPTIMIZATION SPACES (для skopt)
# ============================================================

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


# ============================================================
# GENETIC ALGORITHM SPACES (самописный GA)
# ============================================================

# Расширенное пространство для Random Forest (оптимизировано для скорости)
GENETIC_RF_SPACE = {
    'n_estimators': IntegerParam(100, 400),
    'max_depth': IntegerParam(5, 25),
    'min_samples_leaf': IntegerParam(1, 15),
    'min_samples_split': IntegerParam(2, 15),
    'max_features': CategoricalParam(['sqrt', 'log2']),
}

# Расширенное пространство для CatBoost
GENETIC_CB_SPACE = {
    'depth': IntegerParam(4, 12),
    'learning_rate': RealParam(0.01, 0.3, log_scale=True),
    'iterations': IntegerParam(200, 1500),
    'l2_leaf_reg': RealParam(1.0, 10.0),
    'bagging_temperature': RealParam(0.0, 1.0),
    'random_strength': RealParam(0.0, 2.0),
}

# Настройки генетического алгоритма
# Для маленьких датасетов (<5K) используем быстрые настройки
GA_SETTINGS_FAST = {
    'population_size': 12,
    'generations': 20,
    'mutation_rate': 0.25,
    'mutation_strength': 0.35,
    'crossover_rate': 0.8,
    'elite_size': 2,
    'tournament_size': 3,
    'early_stopping': 8,
}

# Для больших датасетов (>10K) используем полные настройки
GA_SETTINGS_FULL = {
    'population_size': 20,
    'generations': 35,
    'mutation_rate': 0.25,
    'mutation_strength': 0.3,
    'crossover_rate': 0.8,
    'elite_size': 3,
    'tournament_size': 4,
    'early_stopping': 10,
}

# Автоматический выбор настроек
def get_ga_settings(n_samples: int) -> dict:
    """Возвращает настройки GA в зависимости от размера датасета."""
    if n_samples < 5000:
        return GA_SETTINGS_FAST
    else:
        return GA_SETTINGS_FULL

# Для обратной совместимости
GA_SETTINGS = GA_SETTINGS_FAST


# ============================================================
# FEATURE COLUMNS (единый источник истины)
# ============================================================

# Базовые признаки (присутствуют в обоих датасетах)
BASE_FEATURES = [
    'rank_diff',
    'abs_rank_diff',
]

# Контекст карты (только в основном датасете)
MAP_CONTEXT_FEATURES = [
    'map',  # категориальный
    'picked_by_is_A',
    'is_decider',
]

# Исходные статистики (основной датасет)
ORIGINAL_STATS_FEATURES = [
    'map_winrate_A', 'map_winrate_B',
    'recent_form_A', 'recent_form_B',
]

# Статистика игроков (Kaggle датасет)
PLAYER_STATS_FEATURES = [
    'team_A_avg_rating', 'team_A_avg_kd', 'team_A_avg_adr', 'team_A_avg_kast',
    'team_B_avg_rating', 'team_B_avg_kd', 'team_B_avg_adr', 'team_B_avg_kast',
]

# Продвинутые фичи (вычисляются в features.py)
ADVANCED_FEATURES = [
    # Elo
    'elo_diff', 'map_elo_diff',
    # H2H
    'h2h_rate', 'h2h_games',
    # Momentum & Streak
    'momentum_diff', 'streak_A', 'streak_B',
    # Время с последнего матча
    'days_since_last_A', 'days_since_last_B',
    # Скользящие статистики
    'overall_winrate_A', 'overall_winrate_B', 'winrate_diff',
    # Сила соперников
    'opponent_strength_A', 'opponent_strength_B',
    # Опыт на карте
    'map_games_A', 'map_games_B', 'map_experience_diff',
]

# Производные фичи (вычисляются при тренировке)
DERIVED_FEATURES = [
    'rating_diff', 'kd_diff', 'adr_diff', 'kast_diff',
    'streak_diff', 'rest_diff',
]


def get_feature_cols(dataset_type='main', include_derived=False):
    """
    Возвращает список признаков для указанного типа датасета.
    
    Параметры:
    ----------
    dataset_type : str
        'main' - основной датасет с пик-банами
        'kaggle' - расширенный Kaggle датасет
    include_derived : bool
        Включать ли производные признаки
    """
    if dataset_type == 'main':
        features = (
            MAP_CONTEXT_FEATURES + 
            BASE_FEATURES + 
            ORIGINAL_STATS_FEATURES + 
            ADVANCED_FEATURES
        )
    elif dataset_type == 'kaggle':
        features = (
            BASE_FEATURES + 
            PLAYER_STATS_FEATURES + 
            ADVANCED_FEATURES
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    if include_derived:
        features = features + DERIVED_FEATURES
    
    return features