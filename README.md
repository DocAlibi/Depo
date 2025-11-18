# 🎯 Customer Clustering & Churn Prediction System

Комплексная система для кластеризации клиентов банка, анализа миграций между кластерами и подготовки данных для прогнозирования оттока (churn).

## 📋 Содержание

- [Возможности](#возможности)
- [Архитектура](#архитектура)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Детальное использование](#детальное-использование)
- [Структура проекта](#структура-проекта)
- [Модули](#модули)
- [ML Pipeline](#ml-pipeline)
- [Визуализация](#визуализация)

## ✨ Возможности

### 1. **FM-кластеризация**
- Frequency-Monetary анализ клиентов
- Квантильная кластеризация (VIP, Лояльные, Обычные, Спящие)
- Помесячное отслеживание кластеров

### 2. **Анализ миграций**
- Матрица переходов между кластерами
- Расчет retention rate
- Анализ реактивации спящих клиентов
- Топ переходов

### 3. **ML подготовка для Churn Prediction**
- ✅ **Автоматическое создание таргета**: переход из верхнего кластера в нижний = churn
- ✅ **Feature Engineering**: 30+ признаков из транзакций
- ✅ **Trend Features**: тренды за последние 3 месяца
- ✅ **Train/Test Split**: временное разделение данных

### 4. **Визуализация**
- Динамика кластеров (6 графиков)
- Миграционный анализ (4 графика)
- Распределение churn

### 5. **Поддержка сегментов**
- MB_IP (Малый бизнес - ИП)
- MB_TOO (Малый бизнес - ТОО)
- SB (Средний бизнес)
- NEW_CLIENTS (Новые клиенты)
- SOC (Социальные платежи)
- CASH (Кассовые операции)

## 🏗️ Архитектура

```
┌─────────────────┐
│  Oracle DB      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Loader    │  ◄── config/settings.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessor   │  (Очистка, фильтрация, outliers)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FM Clustering  │  (Квантильная кластеризация)
└────────┬────────┘
         │
         ├──────────────────┬─────────────────┐
         ▼                  ▼                 ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Visualization│   │   Migration  │   │  ML Prep     │
│              │   │   Analyzer   │   │              │
└──────────────┘   └──────────────┘   └──────┬───────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │ Churn Target   │
                                      │ + Features     │
                                      │ + Train/Test   │
                                      └────────────────┘
```

## 🚀 Установка

### 1. Клонировать репозиторий
```bash
git clone <repository-url>
cd Depo
```

### 2. Установить зависимости
```bash
pip install -r requirements.txt
```

### 3. Настроить переменные окружения
```bash
cp .env.example .env
```

Отредактировать `.env`:
```env
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=MSBSPSS
DB_PORT=1521
DB_SERVICE_NAME=MSBSPSS
```

## ⚡ Быстрый старт

### Запуск полного анализа (все сегменты + ML)

```python
python main.py
```

Это выполнит:
1. ✅ Загрузку данных из Oracle DB
2. ✅ Кластеризацию для всех сегментов
3. ✅ Анализ миграций
4. ✅ Создание ML датасетов
5. ✅ Визуализацию

### Результаты

Файлы в директории `data/`:
- `summary_all_segments.csv` - сводная таблица
- `clusters_[SEGMENT].csv` - годовая кластеризация
- `monthly_clusters_[SEGMENT].csv` - помесячные кластеры
- `ml_dataset_[SEGMENT].csv` - полный ML датасет
- `train_[SEGMENT].csv` - обучающая выборка
- `test_[SEGMENT].csv` - тестовая выборка

Визуализации:
- `dynamics_[SEGMENT].png` - динамика кластеров
- `migration_[SEGMENT].png` - миграционный анализ
- `churn_[SEGMENT].png` - распределение churn

## 📚 Детальное использование

### Анализ конкретных сегментов

```python
from main import main

# Только MB_IP и NEW_CLIENTS
results, summary = main(
    currency='KZT',
    segments_to_analyze=['MB_IP', 'NEW_CLIENTS'],
    enable_ml=True
)
```

### Только кластеризация (без ML)

```python
results, summary = main(
    currency='KZT',
    enable_ml=False
)
```

### Доступ к результатам

```python
# ML датасет для MB_IP
ml_data = results['MB_IP']['ml_dataset']

# Список признаков
features = results['MB_IP']['feature_columns']

# Обучающая выборка
train_df = results['MB_IP']['train_df']

# Матрица переходов
transition_matrix = results['MB_IP']['transition_matrix']

# Retention rates
retention = results['MB_IP']['retention_rates']
```

### Использование отдельных модулей

#### 1. Database Loading

```python
from src.database import DatabaseConnector, DataLoader

with DatabaseConnector() as db_conn:
    loader = DataLoader(db_conn)
    data = loader.load_all_data()
```

#### 2. Preprocessing

```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_outliers(df)
df_clean = preprocessor.add_time_features(df_clean)
```

#### 3. Clustering

```python
from src.clustering import FMClustering

fm_clustering = FMClustering(use_log=True)
fm_data = fm_clustering.calculate_fm_metrics(df)
fm_data, boundaries = fm_clustering.create_clusters(fm_data)
```

#### 4. ML Preparation

```python
from src.ml_preparation import ChurnTargetCreator, FeatureEngineer

# Создать таргет
churn_creator = ChurnTargetCreator(churn_definition='downgrade')
churn_target = churn_creator.create_churn_target(monthly_clusters)

# Feature engineering
feature_engineer = FeatureEngineer(window_months=3)
features = feature_engineer.create_transaction_features(df, monthly_clusters)
```

## 📁 Структура проекта

```
Depo/
├── config/
│   ├── __init__.py
│   └── settings.py              # Конфигурация (DB, logging, ML)
├── src/
│   ├── __init__.py
│   ├── database.py              # Подключение к БД, загрузка данных
│   ├── preprocessing.py         # Предобработка, очистка outliers
│   ├── clustering.py            # FM-кластеризация, миграции
│   ├── visualization.py         # Графики и визуализация
│   └── ml_preparation.py        # Feature engineering, churn target
├── data/                        # Выходные данные и визуализации
├── logs/                        # Логи выполнения
├── notebooks/                   # Jupyter notebooks (для анализа)
├── main.py                      # Главный скрипт
├── requirements.txt             # Зависимости
├── .env.example                 # Пример конфигурации
└── README.md                    # Документация
```

## 🔧 Модули

### config/settings.py

Центральная конфигурация:
- **DB_CONFIG**: параметры подключения к БД
- **ANALYSIS_CONFIG**: параметры анализа (даты, квантили)
- **ML_CONFIG**: параметры ML (test_size, feature_window)
- **CLUSTER_NAMES**: маппинг кластеров
- **LOGGING_CONFIG**: настройки логирования

### src/database.py

**Классы:**
- `DatabaseConnector`: управление подключением к Oracle
- `DataLoader`: загрузка данных (транзакции, метаданные)

**Методы:**
- `load_main_transactions()`: основные RKO транзакции
- `load_business_metadata()`: метаданные клиентов
- `load_social_transactions()`: социальные платежи
- `load_cash_transactions()`: кассовые операции

### src/preprocessing.py

**Классы:**
- `DataPreprocessor`: предобработка данных
- `SegmentFilter`: фильтрация сегментов

**Методы:**
- `preprocess_segment()`: фильтрация по сегменту
- `clean_outliers()`: удаление выбросов (5-95 перцентили)
- `add_time_features()`: создание временных признаков

### src/clustering.py

**Классы:**
- `FMClustering`: Frequency-Monetary кластеризация
- `MigrationAnalyzer`: анализ миграций

**Методы:**
- `calculate_fm_metrics()`: расчет FM метрик
- `create_clusters()`: квантильная кластеризация
- `apply_clusters_to_monthly()`: применение к помесячным данным
- `analyze_transitions()`: анализ переходов
- `calculate_transition_matrix()`: матрица вероятностей переходов

### src/ml_preparation.py

**Классы:**
- `ChurnTargetCreator`: создание таргета churn
- `FeatureEngineer`: feature engineering
- `MLDatasetBuilder`: сборка финального датасета

**Ключевые методы:**

#### ChurnTargetCreator
```python
create_churn_target(monthly_clusters, prediction_horizon=1)
```
Создает бинарный таргет churn:
- **churn=1**: клиент перешел в худший кластер (downgrade)
- **churn=0**: клиент остался или улучшился

#### FeatureEngineer
```python
create_transaction_features(transactions, monthly_clusters)
```
Создает 30+ признаков:
- **Базовые**: total_amount, avg_amount, std_amount, transaction_count
- **Временные**: days_active, avg_days_between_transactions
- **Разнообразие**: unique_payers, unique_recipients, currency_count
- **Операционные**: по типам операций (oper_count_*)

```python
create_trend_features(monthly_clusters, lookback_months=3)
```
Тренды за N месяцев:
- frequency_growth, monetary_growth
- mean, std, min, max за период

### src/visualization.py

**Класс:**
- `ClusterVisualizer`: визуализация анализа

**Методы:**
- `plot_cluster_dynamics()`: 6 графиков динамики
- `plot_migration_analysis()`: 4 графика миграций
- `plot_churn_distribution()`: распределение churn

## 🤖 ML Pipeline

### 1. Определение Churn

**Downgrade Definition** (по умолчанию):
```
Кластеры: 0=VIP, 1=Лояльные, 2=Обычные, 3=Спящие

Churn = переход с меньшего номера на больший
Примеры:
  VIP (0) → Лояльные (1) = CHURN
  Лояльные (1) → Спящие (3) = CHURN
  Обычные (2) → VIP (0) = НЕ CHURN
```

### 2. Feature Categories

| Категория | Признаки | Количество |
|-----------|----------|------------|
| **Транзакционные** | Суммы, частота, разброс | 10+ |
| **Временные** | Дни активности, интервалы | 3 |
| **Операционные** | Типы операций | 5+ |
| **Контрагенты** | Уникальные плательщики/получатели | 2 |
| **Тренды** | Рост частоты/сумм за 3 мес | 10+ |

### 3. Train/Test Split

**Temporal Split** (рекомендуется):
```python
# Более ранние месяцы = train
# Более поздние месяцы = test
# Соотношение: 80/20
```

**Random Split** (опционально):
```python
ml_builder.split_train_test(temporal_split=False)
```

### 4. Пример обучения модели

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Загрузить данные
train_df = pd.read_csv('data/train_MB_IP.csv')
test_df = pd.read_csv('data/test_MB_IP.csv')

# Выделить X, y
X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']

X_test = test_df.drop('Churn', axis=1)
y_test = test_df['Churn']

# Обучить модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## 📊 Визуализация

### Cluster Dynamics (6 графиков)

1. **Динамика численности**: абсолютное количество по кластерам
2. **Доля кластеров**: процентное распределение (stacked area)
3. **Индекс роста**: рост относительно первого месяца
4. **Месячные изменения**: bar chart изменений
5. **Тепловая карта**: heatmap активности
6. **Доля спящих**: тренд sleeping customers

### Migration Analysis (4 графика)

1. **Матрица переходов**: вероятности transition (%)
2. **Топ-10 переходов**: самые частые миграции
3. **Retention rates**: удержание по кластерам
4. **Реактивация**: переходы из sleeping cluster

### Churn Distribution (4 графика)

1. **Overall churn rate**: pie chart
2. **Churn по кластерам**: bar chart по кластерам
3. **Churn во времени**: динамика по месяцам
4. **Frequency distribution**: boxplot по churn статусу

## ⚙️ Настройки

### Параметры кластеризации

```python
# config/settings.py

ANALYSIS_CONFIG = {
    'outlier_percentile_lower': 0.05,    # Нижний перцентиль
    'outlier_percentile_upper': 0.95,    # Верхний перцентиль
    'cluster_quantiles': [0.25, 0.5, 0.75]  # Квантили для кластеров
}
```

### Параметры ML

```python
ML_CONFIG = {
    'test_size': 0.2,                   # Размер test set
    'random_state': 42,                 # Random seed
    'churn_definition': 'downgrade',    # 'downgrade' or 'exit'
    'feature_window_months': 3,         # Окно для rolling features
}
```

## 🐛 Troubleshooting

### Проблема: Ошибка подключения к БД

```
ConnectionError: Database connection failed
```

**Решение:**
1. Проверить `.env` файл
2. Убедиться, что Oracle DB доступна
3. Проверить credentials

### Проблема: Недостаточно данных

```
WARNING: Insufficient data quality
```

**Решение:**
1. Уменьшить `outlier_percentile_lower/upper`
2. Увеличить период анализа в SQL запросах
3. Проверить фильтры сегмента

### Проблема: Низкий churn rate (<1%)

**Решение:**
1. Изменить определение churn в `ML_CONFIG`
2. Использовать SMOTE для балансировки классов
3. Увеличить `prediction_horizon` в `create_churn_target()`

## 🔍 Логирование

Логи сохраняются в `logs/clustering_analysis.log`

Уровни:
- **DEBUG**: детальная информация
- **INFO**: основные события (console + file)
- **WARNING**: предупреждения
- **ERROR**: ошибки с traceback

## 📈 Производительность

### Оптимизации:

1. **Векторизация**: использование `np.select` вместо циклов
2. **Batch loading**: загрузка данных одним запросом
3. **Эффективный SQL**: фильтрация на уровне БД
4. **Pandas оптимизация**: `groupby` + `agg` вместо `apply`

### Время выполнения (примерное):

| Операция | Время |
|----------|-------|
| Загрузка данных | 1-2 мин |
| Кластеризация (1 сегмент) | 10-20 сек |
| Feature engineering | 30-60 сек |
| Визуализация | 5-10 сек |

## 🤝 Contributing

1. Fork репозиторий
2. Создать feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit изменения (`git commit -m 'Add some AmazingFeature'`)
4. Push в branch (`git push origin feature/AmazingFeature`)
5. Открыть Pull Request

## 📝 License

Этот проект для внутреннего использования.

## 📧 Контакты

Вопросы и предложения: см. Issues в репозитории

---

**Версия**: 2.0.0
**Последнее обновление**: 2025-01-18
