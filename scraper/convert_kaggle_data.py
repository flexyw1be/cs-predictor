"""
Конвертация данных Kaggle в формат для нашей модели.

Поддерживаемые датасеты:
- mateusdmachado/csgo-professional-matches
- любой CSV с колонками: date, team1, team2, map, score, event
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime


def find_and_load_kaggle_data(kaggle_dir='data/kaggle'):
    """Найти и загрузить CSV файлы из Kaggle датасета"""
    
    csv_files = glob.glob(os.path.join(kaggle_dir, '**/*.csv'), recursive=True)
    
    if not csv_files:
        print(f"Не найдены CSV файлы в {kaggle_dir}")
        print("Скачайте датасет: bash scraper/download_kaggle.sh")
        return None
    
    print(f"Найдено {len(csv_files)} CSV файлов:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Пробуем загрузить каждый файл
    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"\n{csv_file}:")
            print(f"  Записей: {len(df)}")
            print(f"  Колонки: {list(df.columns)[:10]}...")
            dataframes.append((csv_file, df))
        except Exception as e:
            print(f"  Ошибка: {e}")
    
    return dataframes


def convert_mateusdmachado_dataset(df):
    """
    Конвертировать датасет mateusdmachado/csgo-professional-matches
    """
    print("\nКонвертация датасета Mateus D Machado...")
    
    # Проверяем наличие колонок
    print(f"Доступные колонки: {df.columns.tolist()}")
    
    # Стандартизация
    df_clean = pd.DataFrame()
    
    # Дата
    if 'date' in df.columns:
        df_clean['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'Date' in df.columns:
        df_clean['date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        df_clean['date'] = pd.Timestamp.now()
    
    # Команды
    df_clean['team_A'] = df.get('team_1', df.get('team1', df.get('Team1', '')))
    df_clean['team_B'] = df.get('team_2', df.get('team2', df.get('Team2', '')))
    
    # Карта - проверяем разные названия
    if '_map' in df.columns:
        df_clean['map'] = df['_map']
    elif 'map' in df.columns:
        df_clean['map'] = df['map']
    elif 'Map' in df.columns:
        df_clean['map'] = df['Map']
    elif 'map_name' in df.columns:
        df_clean['map'] = df['map_name']
    else:
        df_clean['map'] = 'Unknown'
    
    # Результат
    if 'result_1' in df.columns and 'result_2' in df.columns:
        df_clean['score_A'] = pd.to_numeric(df['result_1'], errors='coerce').fillna(0).astype(int)
        df_clean['score_B'] = pd.to_numeric(df['result_2'], errors='coerce').fillna(0).astype(int)
    else:
        df_clean['score_A'] = 0
        df_clean['score_B'] = 0
    
    # Победитель
    if 'map_winner' in df.columns:
        # map_winner: 1 = team_1, 2 = team_2
        df_clean['winner'] = np.where(
            df['map_winner'] == 1,
            df_clean['team_A'],
            df_clean['team_B']
        )
    else:
        df_clean['winner'] = np.where(
            df_clean['score_A'] > df_clean['score_B'],
            df_clean['team_A'],
            df_clean['team_B']
        )
    df_clean['winner_is_A'] = (df_clean['winner'] == df_clean['team_A']).astype(int)
    
    # Турнир
    if 'event_id' in df.columns:
        df_clean['event_id'] = df['event_id']
    df_clean['event'] = df.get('event', df.get('Event', df.get('event_name', 'Unknown')))
    
    # Match ID
    if 'match_id' in df.columns:
        df_clean['match_id'] = df['match_id']
    else:
        df_clean['match_id'] = range(1, len(df_clean) + 1)
    
    # Рейтинги команд
    if 'rank_1' in df.columns:
        df_clean['team_A_rank'] = pd.to_numeric(df['rank_1'], errors='coerce').fillna(50)
        df_clean['team_B_rank'] = pd.to_numeric(df['rank_2'], errors='coerce').fillna(50)
    else:
        df_clean['team_A_rank'] = 50
        df_clean['team_B_rank'] = 50
    
    df_clean['rank_diff'] = df_clean['team_A_rank'] - df_clean['team_B_rank']
    df_clean['abs_rank_diff'] = df_clean['rank_diff'].abs()
    
    # Map wins in the series
    if 'map_wins_1' in df.columns:
        df_clean['map_wins_A'] = df['map_wins_1']
        df_clean['map_wins_B'] = df['map_wins_2']
    
    # Убираем пустые записи
    df_clean = df_clean.dropna(subset=['team_A', 'team_B', 'date'])
    df_clean = df_clean[df_clean['team_A'] != '']
    df_clean = df_clean[df_clean['team_B'] != '']
    
    print(f"Уникальных карт: {df_clean['map'].nunique()}")
    print(f"Карты: {df_clean['map'].unique()[:10]}")
    
    return df_clean


def add_player_stats_columns(df, player_stats_df=None):
    """
    Добавить статистику игроков к датафрейму матчей.
    Если нет отдельного файла - используем заглушки.
    """
    
    if player_stats_df is not None:
        # Проверяем какие колонки есть
        available_cols = player_stats_df.columns.tolist()
        print(f"Колонки в player_stats: {available_cols[:15]}...")
        
        # Определяем колонки для агрегации
        agg_dict = {}
        if 'rating' in available_cols:
            agg_dict['rating'] = 'mean'
        if 'kddiff' in available_cols:
            agg_dict['kddiff'] = 'mean'
        elif 'kd' in available_cols:
            agg_dict['kd'] = 'mean'
        if 'adr' in available_cols:
            agg_dict['adr'] = 'mean'
        if 'kast' in available_cols:
            agg_dict['kast'] = 'mean'
        
        if not agg_dict:
            print("Не найдены нужные колонки для статистики игроков")
            df['team_A_avg_rating'] = 1.0
            df['team_B_avg_rating'] = 1.0
            return df
        
        # Агрегируем статистику по командам
        team_stats = player_stats_df.groupby('team').agg(agg_dict).reset_index()
        
        # Переименовываем колонки
        rename_map_A = {'team': 'team_A'}
        rename_map_B = {'team': 'team_B'}
        
        if 'rating' in agg_dict:
            rename_map_A['rating'] = 'team_A_avg_rating'
            rename_map_B['rating'] = 'team_B_avg_rating'
        if 'kddiff' in agg_dict:
            rename_map_A['kddiff'] = 'team_A_avg_kd'
            rename_map_B['kddiff'] = 'team_B_avg_kd'
        elif 'kd' in agg_dict:
            rename_map_A['kd'] = 'team_A_avg_kd'
            rename_map_B['kd'] = 'team_B_avg_kd'
        if 'adr' in agg_dict:
            rename_map_A['adr'] = 'team_A_avg_adr'
            rename_map_B['adr'] = 'team_B_avg_adr'
        if 'kast' in agg_dict:
            rename_map_A['kast'] = 'team_A_avg_kast'
            rename_map_B['kast'] = 'team_B_avg_kast'
        
        # Мержим с матчами
        team_stats_A = team_stats.rename(columns=rename_map_A)
        team_stats_B = team_stats.rename(columns=rename_map_B)
        
        df = df.merge(team_stats_A, on='team_A', how='left')
        df = df.merge(team_stats_B, on='team_B', how='left')
        
        # Заполняем пропуски
        for col in df.columns:
            if 'avg_rating' in col:
                df[col] = df[col].fillna(1.0)
            elif 'avg_kd' in col:
                df[col] = df[col].fillna(0.0)
            elif 'avg_adr' in col:
                df[col] = df[col].fillna(75.0)
            elif 'avg_kast' in col:
                df[col] = df[col].fillna(70.0)
        
        print(f"Добавлена статистика игроков для {len(team_stats)} команд")
    else:
        # Заглушки
        df['team_A_avg_rating'] = 1.0
        df['team_B_avg_rating'] = 1.0
        df['team_A_avg_kd'] = 0.0
        df['team_B_avg_kd'] = 0.0
        df['team_A_avg_adr'] = 75.0
        df['team_B_avg_adr'] = 75.0
    
    return df


def process_kaggle_dataset(kaggle_dir='data/kaggle', output_dir='data/processed_kaggle'):
    """Главная функция обработки"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем данные
    datasets = find_and_load_kaggle_data(kaggle_dir)
    
    if not datasets:
        return None
    
    # Объединяем все датасеты
    all_data = []
    player_stats = None
    
    for filepath, df in datasets:
        filename = os.path.basename(filepath).lower()
        
        # Определяем тип файла
        if 'player' in filename:
            player_stats = df
            print(f"Найден файл статистики игроков: {filepath}")
        elif 'match' in filename or 'result' in filename or 'game' in filename:
            converted = convert_mateusdmachado_dataset(df)
            if converted is not None and len(converted) > 0:
                all_data.append(converted)
                print(f"Конвертировано {len(converted)} матчей из {filepath}")
    
    if not all_data:
        print("Не удалось конвертировать данные")
        return None
    
    # Объединяем
    df_all = pd.concat(all_data, ignore_index=True)
    df_all = df_all.sort_values('date').reset_index(drop=True)
    
    # Добавляем статистику игроков
    df_all = add_player_stats_columns(df_all, player_stats)
    
    print(f"\nИтого: {len(df_all)} записей")
    print(f"Период: {df_all['date'].min()} - {df_all['date'].max()}")
    print(f"Команд: {df_all['team_A'].nunique()} + {df_all['team_B'].nunique()}")
    print(f"Карт: {df_all['map'].nunique()}")
    
    # Сохраняем
    df_all.to_csv(os.path.join(output_dir, 'kaggle_matches.csv'), index=False)
    print(f"\nСохранено в {output_dir}/kaggle_matches.csv")
    
    return df_all


if __name__ == "__main__":
    process_kaggle_dataset()
