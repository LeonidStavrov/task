import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score
from enum import Enum

class Metrics(Enum):
    MSE = "mean_squared_error"  # Среднеквадратичная ошибка
    R2 = "r_squared"  # Коэффициент детерминации (R-квадрат)
    MAE = "mean_absolute_error"  # Средняя абсолютная ошибка

def compute_ap(y_true, y_pred, k=None):
    """Вычисляет среднюю точность (Average Precision, AP).
    
    Параметры:
    y_true -- истинные значения релевантности документов (0 или 1)
    y_pred -- предсказанные ранги документов
    k -- количество документов, для которых вычисляется метрика (по умолчанию используется весь список)
    
    Возвращает:
    ap -- средняя точность (Average Precision)
    """
    if k is None:
        k = len(y_true)
    else:
        k = min(k, len(y_true))
    
    sorted_indices = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)
    
    precision_at_k = 0.0
    num_relevant = 0
    for i in range(k):
        if y_true[sorted_indices[i]] == 1:
            num_relevant += 1
            precision_at_k += num_relevant / (i + 1)
    
    ap = precision_at_k / min(k, sum(y_true)) if sum(y_true) > 0 else 0
    
    return ap

def compute_map(y_true_list, y_pred_list, k=None):
    """Вычисляет среднюю точность среди запросов (Mean Average Precision, MAP).
    
    Параметры:
    y_true_list -- список списков истинных значений релевантности документов для каждого запроса
    y_pred_list -- список списков предсказанных рангов документов для каждого запроса
    k -- количество документов, для которых вычисляется метрика (по умолчанию используется весь список)
    
    Возвращает:
    map_score -- средняя точность среди запросов (Mean Average Precision)
    """
    ap_scores = [compute_ap(y_true, y_pred, k) for y_true, y_pred in zip(y_true_list, y_pred_list)]
    map_score = sum(ap_scores) / len(ap_scores)
    
    return map_score

df = pd.read_csv('intern_task.csv')

# Проверка наличия признаков с нулевой вариативностью
zero_variance_features = df.columns[df.nunique() == 1]
data_filtered = df.drop(columns=zero_variance_features)

# Разбиение данных на обучающий и тестовый наборы
train_data_filtered, test_data_filtered = train_test_split(data_filtered, test_size=0.2, random_state=42)

# Отбор признаков для моделирования
features_filtered = [col for col in data_filtered.columns if col.startswith('feature_')]

# Параметры модели LightGBM
params = {
    'boosting_type': 'gbdt',  # тип градиентного бустинга
    'objective': 'regression',  # тип задачи (регрессия)
    'metric': 'mse',  # метрика качества (среднеквадратичная ошибка)
    'num_leaves': 31,  # максимальное количество листьев в дереве
    'learning_rate': 0.1,  # скорость обучения
    'feature_fraction': 0.9,  # доля признаков, используемых для обучения каждого дерева
    'bagging_fraction': 0.8,  # доля данных, используемых для обучения каждого дерева
    'bagging_freq': 5,  # частота использования bagging
    'verbose': -1  # уровень вывода информации
}

# Обучение моделей LightGBM для каждого query_id
models_filtered = {}
for query_id, group in train_data_filtered.groupby('query_id'):
    if len(group) > 1:
        X_train = group[features_filtered]
        y_train = group['rank']
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        models_filtered[query_id] = model
    else:
        print(f"Недостаточно данных для обучения модели для query_id {query_id}")

# Пример применения модели для одного query_id
example_query_id = 10
X_example = test_data_filtered[test_data_filtered['query_id'] == example_query_id][features_filtered]
y_true = test_data_filtered[test_data_filtered['query_id'] == example_query_id]['rank']
y_pred = models_filtered[example_query_id].predict(X_example)

# Оценка качества модели на примере одного query_id
mse = mean_squared_error(y_true, y_pred)
print(f"MSE для query_id {example_query_id} (LightGBM): {mse}")

# Обучение линейных моделей для каждого query_id
linear_models = {}
for query_id, group in train_data_filtered.groupby('query_id'):
    if len(group) > 1:
        X_train = group[features_filtered]
        y_train = group['rank']
        model = LinearRegression()
        model.fit(X_train, y_train)
        linear_models[query_id] = model
    else:
        print(f"Недостаточно данных для обучения линейной модели для query_id {query_id}")

# Применение линейной модели для одного query_id
example_query_id = 10
X_example = test_data_filtered[test_data_filtered['query_id'] == example_query_id][features_filtered]
y_true = test_data_filtered[test_data_filtered['query_id'] == example_query_id]['rank']
y_pred_linear = linear_models[example_query_id].predict(X_example)

# Оценка качества модели на примере одного query_id
mse_linear = mean_squared_error(y_true, y_pred_linear)
print(f"MSE для query_id {example_query_id} (линейная регрессия): {mse_linear}")

# Создание словаря средних значений ранга для каждого query_id в обучающем наборе данных
mean_ranks = train_data_filtered.groupby('query_id')['rank'].mean().to_dict()

# Применение метода среднего значения для всех query_id
y_pred_final_simple = []
for query_id, group in test_data_filtered.groupby('query_id'):
    mean_rank = mean_ranks.get(query_id, train_data_filtered['rank'].mean())
    y_pred_final_simple.extend([mean_rank] * len(group))

# Оценка качества финальной модели метода среднего значения
mse_final_simple = mean_squared_error(test_data_filtered['rank'], y_pred_final_simple)
print(f"MSE финальной модели (метод среднего значения): {mse_final_simple}")

# Вычисление средней точности среди запросов (MAP)
y_true_list = []
y_pred_list = []
for query_id, group in test_data_filtered.groupby('query_id'):
    y_true_list.append(group['rank'].values)
    y_pred_list.append([mean_ranks.get(query_id, train_data_filtered['rank'].mean())] * len(group))

map_score = compute_map(y_true_list, y_pred_list)
print(f"Средняя точность среди запросов (MAP): {round(map_score * 100, 1)}%")


# Расчет NDCG_5 для модели LightGBM
ndcg_lgb = ndcg_score(y_true.values.reshape(1, -1), y_pred.reshape(1, -1), k=5)
print(f"NDCG_5 для модели LightGBM: {ndcg_lgb:.4f}")

# Расчет NDCG_5 для модели линейной регрессии
ndcg_linear = ndcg_score(y_true.values.reshape(1, -1), y_pred_linear.reshape(1, -1), k=5)
print(f"NDCG_5 для модели линейной регрессии: {ndcg_linear:.4f}")

# Расчет NDCG_5 для модели метода среднего значения
ndcg_mean = ndcg_score(test_data_filtered['rank'].values.reshape(1, -1), np.array(y_pred_final_simple).reshape(1, -1), k=5)
print(f"NDCG_5 для модели метода среднего значения: {ndcg_mean:.4f}")