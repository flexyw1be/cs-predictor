"""
Обработка данных из Kaggle датасета для тренировки модели.
Включает вычисление Elo рейтингов, H2H статистики и других фичей.
"""
import pandas as pd
import numpy as np
import os
from features import compute_advanced_features


def process_kaggle_data(input_path='data/processed_kaggle/kaggle_matches.csv', 
                        output_dir='data/processed_kaggle'):
    """Обрабатывает Kaggle данные и создает train/val/test сплиты."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Загрузка данных из {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Загружено {len(df)} записей")
    
    # Убедимся, что даты отсортированы
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Период данных: {df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Вычисляем продвинутые фичи (Elo, H2H, momentum и т.д.)
    print("\nВычисление продвинутых признаков (Elo, H2H, momentum)...")
    df = compute_advanced_features(df)
    
    # Создаем временные сплиты (важно для предотвращения data leakage!)
    # Последние 10% - test, предпоследние 10% - val, остальное - train
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    df['split_row'] = 'train'
    df.loc[train_end:val_end-1, 'split_row'] = 'val'
    df.loc[val_end:, 'split_row'] = 'test'
    
    # Сохраняем полный датасет
    full_path = os.path.join(output_dir, 'full_ml_dataset.csv')
    df.to_csv(full_path, index=False)
    print(f"\nСохранен полный датасет: {full_path}")
    
    # Разделяем на Train, Val, Test
    train_df = df[df['split_row'] == 'train']
    val_df = df[df['split_row'] == 'val']
    test_df = df[df['split_row'] == 'test']
    
    # Сохраняем сплиты
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"\nРазмеры датасетов:")
    print(f"  Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")  
    print(f"  Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    # Статистика по фичам
    print("\n" + "="*60)
    print("Статистика ключевых признаков:")
    print("="*60)
    
    feature_cols = [
        'rank_diff', 'elo_diff', 'map_elo_diff', 'h2h_rate',
        'momentum_diff', 'winrate_diff', 'team_A_avg_rating', 'team_B_avg_rating'
    ]
    
    available_cols = [c for c in feature_cols if c in df.columns]
    if available_cols:
        print(df[available_cols].describe().round(3).to_string())
    
    # Проверка качества данных
    print("\n" + "="*60)
    print("Проверка качества данных:")
    print("="*60)
    
    # Процент с рейтингом
    has_rating_A = (df['team_A_avg_rating'] > 0).sum() / len(df) * 100
    has_rating_B = (df['team_B_avg_rating'] > 0).sum() / len(df) * 100
    print(f"Записей со статистикой team_A: {has_rating_A:.1f}%")
    print(f"Записей со статистикой team_B: {has_rating_B:.1f}%")
    
    # Распределение target
    print(f"\nРаспределение winner_is_A:")
    print(f"  Team A wins: {(df['winner_is_A'] == 1).sum()} ({(df['winner_is_A'] == 1).mean()*100:.1f}%)")
    print(f"  Team B wins: {(df['winner_is_A'] == 0).sum()} ({(df['winner_is_A'] == 0).mean()*100:.1f}%)")
    
    print(f"\nДанные готовы! Запустите python train_kaggle.py для тренировки")
    
    return df


if __name__ == "__main__":
    process_kaggle_data()
