DATA_PATH = 'ml_dataset.csv'

DROP_COLS = [
    'match_id', 'date', 'team_A', 'team_B', 'map'
]

TARGET_COL = 'winner_is_A'


GA_PARAMS = {
    'population_size': 20,
    'generations': 10,
    'crossover_probability': 0.8,
    'mutation_probability': 0.1,
    'n_jobs': -1,
    'verbose': True
}