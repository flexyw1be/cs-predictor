"""
Самописный генетический алгоритм для оптимизации гиперпараметров ML моделей.
Поддерживает Integer, Real (float) и Categorical параметры.
"""
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Callable, Union
from dataclasses import dataclass
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


@dataclass
class IntegerParam:
    """Целочисленный параметр в заданном диапазоне."""
    low: int
    high: int
    
    def sample(self) -> int:
        return random.randint(self.low, self.high)
    
    def mutate(self, value: int, mutation_strength: float = 0.3) -> int:
        """Мутация с гауссовым шумом."""
        range_size = self.high - self.low
        delta = int(np.random.normal(0, range_size * mutation_strength))
        new_value = value + delta
        return max(self.low, min(self.high, new_value))


@dataclass  
class RealParam:
    """Вещественный параметр в заданном диапазоне."""
    low: float
    high: float
    log_scale: bool = False  # Для learning_rate и подобных
    
    def sample(self) -> float:
        if self.log_scale:
            log_low, log_high = np.log(self.low), np.log(self.high)
            return float(np.exp(np.random.uniform(log_low, log_high)))
        return random.uniform(self.low, self.high)
    
    def mutate(self, value: float, mutation_strength: float = 0.3) -> float:
        """Мутация с гауссовым шумом."""
        if self.log_scale:
            log_val = np.log(value)
            log_range = np.log(self.high) - np.log(self.low)
            delta = np.random.normal(0, log_range * mutation_strength)
            new_val = np.exp(log_val + delta)
        else:
            range_size = self.high - self.low
            delta = np.random.normal(0, range_size * mutation_strength)
            new_val = value + delta
        return max(self.low, min(self.high, float(new_val)))


@dataclass
class CategoricalParam:
    """Категориальный параметр из списка значений."""
    choices: List[Any]
    
    def sample(self) -> Any:
        return random.choice(self.choices)
    
    def mutate(self, value: Any, mutation_strength: float = 0.3) -> Any:
        """Мутация — случайный выбор другого значения."""
        if random.random() < mutation_strength and len(self.choices) > 1:
            other_choices = [c for c in self.choices if c != value]
            return random.choice(other_choices)
        return value


ParamType = Union[IntegerParam, RealParam, CategoricalParam]


class Individual:
    """Особь в популяции — набор гиперпараметров."""
    
    def __init__(self, params: Dict[str, ParamType], genes: Dict[str, Any] = None):
        self.param_space = params
        if genes is None:
            self.genes = {name: param.sample() for name, param in params.items()}
        else:
            self.genes = genes
        self.fitness: float = None
        self.cv_scores: List[float] = None
    
    def mutate(self, mutation_rate: float = 0.2, mutation_strength: float = 0.3) -> 'Individual':
        """Создает мутировавшую копию особи."""
        new_genes = {}
        for name, value in self.genes.items():
            if random.random() < mutation_rate:
                new_genes[name] = self.param_space[name].mutate(value, mutation_strength)
            else:
                new_genes[name] = value
        return Individual(self.param_space, new_genes)
    
    def crossover(self, other: 'Individual') -> Tuple['Individual', 'Individual']:
        """Одноточечный кроссовер с другой особью."""
        keys = list(self.genes.keys())
        if len(keys) <= 1:
            return self.mutate(), other.mutate()
        
        crossover_point = random.randint(1, len(keys) - 1)
        
        child1_genes = {}
        child2_genes = {}
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                child1_genes[key] = self.genes[key]
                child2_genes[key] = other.genes[key]
            else:
                child1_genes[key] = other.genes[key]
                child2_genes[key] = self.genes[key]
        
        return (
            Individual(self.param_space, child1_genes),
            Individual(self.param_space, child2_genes)
        )
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, genes={self.genes})"


class GeneticOptimizer:
    """
    Генетический алгоритм для оптимизации гиперпараметров.
    
    Параметры:
    ----------
    param_space : Dict[str, ParamType]
        Пространство поиска гиперпараметров
    population_size : int
        Размер популяции (по умолчанию 20)
    generations : int
        Количество поколений (по умолчанию 30)
    mutation_rate : float
        Вероятность мутации гена (по умолчанию 0.2)
    mutation_strength : float
        Сила мутации (по умолчанию 0.3)
    crossover_rate : float
        Вероятность кроссовера (по умолчанию 0.8)
    elite_size : int
        Количество лучших особей, сохраняемых без изменений (по умолчанию 2)
    tournament_size : int
        Размер турнира для селекции (по умолчанию 3)
    early_stopping : int
        Остановка если нет улучшения N поколений (по умолчанию 10)
    random_state : int
        Seed для воспроизводимости
    verbose : int
        Уровень логирования (0, 1, 2)
    """
    
    def __init__(
        self,
        param_space: Dict[str, ParamType],
        population_size: int = 20,
        generations: int = 30,
        mutation_rate: float = 0.2,
        mutation_strength: float = 0.3,
        crossover_rate: float = 0.8,
        elite_size: int = 2,
        tournament_size: int = 3,
        early_stopping: int = 10,
        random_state: int = None,
        verbose: int = 1
    ):
        self.param_space = param_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elite_size = min(elite_size, population_size // 2)
        self.tournament_size = min(tournament_size, population_size)
        self.early_stopping = early_stopping
        self.verbose = verbose
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        self.population: List[Individual] = []
        self.best_individual: Individual = None
        self.history: List[Dict] = []
    
    def _create_population(self) -> List[Individual]:
        """Инициализация случайной популяции."""
        return [Individual(self.param_space) for _ in range(self.population_size)]
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Турнирная селекция."""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _select_parents(self, population: List[Individual]) -> List[Individual]:
        """Селекция родителей для следующего поколения."""
        parents = []
        for _ in range(self.population_size):
            parents.append(self._tournament_selection(population))
        return parents
    
    def _create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """Создание следующего поколения через кроссовер и мутацию."""
        # Сортируем по fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Элитизм — лучшие переходят без изменений
        next_gen = [deepcopy(ind) for ind in sorted_pop[:self.elite_size]]
        
        # Остальные — через кроссовер и мутацию
        while len(next_gen) < self.population_size:
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            if random.random() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            # Мутация
            child1 = child1.mutate(self.mutation_rate, self.mutation_strength)
            child2 = child2.mutate(self.mutation_rate, self.mutation_strength)
            
            next_gen.append(child1)
            if len(next_gen) < self.population_size:
                next_gen.append(child2)
        
        return next_gen
    
    def _evaluate_individual(
        self, 
        individual: Individual,
        model_class: type,
        X, y,
        cv_splitter,
        scoring_func: Callable,
        fit_params: Dict = None
    ) -> float:
        """Оценка одной особи через кросс-валидацию."""
        scores = []
        fit_params = fit_params or {}
        
        for train_idx, val_idx in cv_splitter.split(X, y):
            # Разделение данных
            if hasattr(X, 'iloc'):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # Создаём модель с параметрами особи
                model = model_class(**individual.genes)
                model.fit(X_train, y_train, **fit_params)
                
                # Оценка
                score = scoring_func(model, X_val, y_val)
                scores.append(score)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"  Ошибка при оценке {individual.genes}: {e}")
                scores.append(0.0)
        
        individual.cv_scores = scores
        individual.fitness = np.mean(scores)
        return individual.fitness
    
    def _evaluate_population(
        self,
        population: List[Individual],
        model_class: type,
        X, y,
        cv_splitter,
        scoring_func: Callable,
        fit_params: Dict = None
    ):
        """Оценка всей популяции с прогресс-баром."""
        to_evaluate = [ind for ind in population if ind.fitness is None]
        total = len(to_evaluate)
        
        for i, individual in enumerate(to_evaluate):
            self._evaluate_individual(
                individual, model_class, X, y, 
                cv_splitter, scoring_func, fit_params
            )
            # Прогресс-бар
            if self.verbose >= 1:
                progress = (i + 1) / total
                bar_len = 30
                filled = int(bar_len * progress)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"\r  Оценка: [{bar}] {i+1}/{total} (fitness={individual.fitness:.4f})", end='', flush=True)
        
        if self.verbose >= 1 and total > 0:
            print()  # Новая строка после прогресс-бара
    
    def optimize(
        self,
        model_class: type,
        X, y,
        cv_splitter,
        scoring_func: Callable = None,
        fit_params: Dict = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Запуск оптимизации.
        
        Параметры:
        ----------
        model_class : type
            Класс модели (RandomForestClassifier, CatBoostClassifier, ...)
        X : array-like
            Признаки
        y : array-like  
            Целевая переменная
        cv_splitter : sklearn CV splitter
            Кросс-валидатор (StratifiedKFold, GroupKFold, ...)
        scoring_func : Callable
            Функция оценки: (model, X_val, y_val) -> float
            По умолчанию — accuracy
        fit_params : Dict
            Дополнительные параметры для model.fit()
            
        Возвращает:
        -----------
        best_params : Dict
            Лучшие найденные гиперпараметры
        best_score : float
            Лучший CV score
        """
        if scoring_func is None:
            scoring_func = lambda model, X, y: model.score(X, y)
        
        # Инициализация популяции
        self.population = self._create_population()
        
        if self.verbose >= 1:
            print("="*60)
            print("🧬 ГЕНЕТИЧЕСКИЙ АЛГОРИТМ")
            print("="*60)
            print(f"Популяция: {self.population_size}")
            print(f"Поколений: {self.generations}")
            print(f"Пространство поиска: {list(self.param_space.keys())}")
            print("="*60)
        
        no_improvement_count = 0
        best_ever_fitness = -np.inf
        
        for gen in range(self.generations):
            # Оценка популяции
            self._evaluate_population(
                self.population, model_class, X, y,
                cv_splitter, scoring_func, fit_params
            )
            
            # Статистика поколения
            fitnesses = [ind.fitness for ind in self.population]
            gen_best = max(self.population, key=lambda x: x.fitness)
            gen_mean = np.mean(fitnesses)
            gen_std = np.std(fitnesses)
            
            # Обновляем лучшую особь
            if gen_best.fitness > best_ever_fitness:
                best_ever_fitness = gen_best.fitness
                self.best_individual = deepcopy(gen_best)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Сохраняем историю
            self.history.append({
                'generation': gen + 1,
                'best_fitness': gen_best.fitness,
                'mean_fitness': gen_mean,
                'std_fitness': gen_std,
                'best_params': gen_best.genes.copy(),
                'best_ever_fitness': best_ever_fitness
            })
            
            if self.verbose >= 1:
                print(f"Поколение {gen+1:3d}/{self.generations}: "
                      f"best={gen_best.fitness:.4f}, "
                      f"mean={gen_mean:.4f}±{gen_std:.4f}, "
                      f"best_ever={best_ever_fitness:.4f}")
            
            if self.verbose >= 2:
                print(f"  Лучшие параметры: {gen_best.genes}")
            
            # Early stopping
            if no_improvement_count >= self.early_stopping:
                if self.verbose >= 1:
                    print(f"\n⚡ Early stopping: нет улучшения {self.early_stopping} поколений")
                break
            
            # Создание следующего поколения (кроме последнего)
            if gen < self.generations - 1:
                self.population = self._create_next_generation(self.population)
        
        if self.verbose >= 1:
            print("\n" + "="*60)
            print("🏆 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
            print("="*60)
            print(f"Лучший CV Score: {self.best_individual.fitness:.4f}")
            print(f"Лучшие параметры:")
            for name, value in self.best_individual.genes.items():
                print(f"  {name}: {value}")
            print("="*60)
        
        return self.best_individual.genes, self.best_individual.fitness
    
    def get_cv_results(self) -> List[Dict]:
        """Возвращает историю оптимизации."""
        return self.history


def accuracy_score_func(model, X, y):
    """Стандартная функция оценки через accuracy."""
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)


def roc_auc_score_func(model, X, y):
    """Функция оценки через ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.predict(X)
    return roc_auc_score(y, y_prob)
