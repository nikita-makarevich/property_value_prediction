import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

from data_cleaning import DataCleaning

# Путь к файлу данных
file_path = os.getenv('FILE_PATH')

# Чтение данных из CSV файла
data = pd.read_csv(file_path)

# Создание копии данных
data_copy = data.copy()

# Очистка данных с использованием класса DataCleaning
df = DataCleaning(data)
clean_data = df.main()

# Отделение целевой переменной (sold_price) от признаков
y = clean_data['sold_price']
X = clean_data.drop(['sold_price'], axis=1)

# Разделение данных на обучающую и валидационную выборки
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Выбор категориальных столбцов с количеством уникальных значений менее 1000
categorical_cols = [
    cname for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 1000 and X_train_full[cname].dtype == "object"
]

# Выбор числовых столбцов
numerical_cols = [
    cname for cname in X_train_full.columns
    if X_train_full[cname].dtype in ['int64', 'float64']
]

# Оставляем только выбранные столбцы
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Предобработка для числовых данных
numerical_transformer = SimpleImputer(strategy='constant', fill_value=0)

# Предобработка для категориальных данных
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Объединение предобработки для числовых и категориальных данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Создание модели LGBMRegressor
model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)

# Объединение предобработки и моделирования в один конвейер (Pipeline)
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Обучение модели
my_pipeline.fit(X_train, y_train)

# Предобработка валидационных данных и получение предсказаний
preds = my_pipeline.predict(X_valid)

# Оценка модели
score_MAE = mean_absolute_error(y_valid, preds)
score_MAPE = mean_absolute_percentage_error(y_valid, preds)

print('MAE:', score_MAE)
print('MAPE:', score_MAPE)

# Построение графиков фактических и предсказанных значений
plt.figure(figsize=(12, 6))

plt.plot(y_valid.values, label='Фактические значения', color='blue', alpha=0.7)
plt.plot(preds, label='Предсказанные значения', color='red', alpha=0.7)

plt.xlabel('Примеры')
plt.ylabel('Продажная цена')
plt.title('Сравнение фактических и предсказанных значений')
plt.legend()

# Добавление текста с оценками модели и названием модели
model_name = "LGBMRegressor"
plt.text(
    0.1, 0.9, f'{model_name}\nMAE: {score_MAE:.2f}\nMAPE: {score_MAPE:.2%}',
    transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white')
)

# Путь для сохранения графика
plot_path = os.path.join('..', 'graphical_results', 'LGBMR_comparison_plot.png')

# Сохранение графика в файл
plt.savefig(plot_path)