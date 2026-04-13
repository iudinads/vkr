import io
import pandas as pd
from datetime import datetime
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ALLOWED_EXTENSIONS = {'csv'}

# Шаблоны для распознавания дат
DATE_PATTERNS = [
    r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD
    r'\d{2}/\d{2}/\d{4}',       # MM/DD/YYYY
    r'\d{2}-\d{2}-\d{4}',       # DD-MM-YYYY
    r'\d{2}\.\d{2}\.\d{4}',     # DD.MM.YYYY
    r'\d{4}/\d{2}/\d{2}',       # YYYY/MM/DD
    r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',  # Общий шаблон
    r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # С временем
]

def is_date_column(series, threshold=0.7):
    """
    Определяет, содержит ли колонка даты.
    threshold - минимальная доля значений, которые должны соответствовать шаблону даты.
    """
    if series.dtype != 'object':
        return False
        
    sample = series.dropna().head(1000)  # Проверяем первые 1000 непустых значений
    
    if len(sample) == 0:
        return False
        
    match_count = 0
    
    for value in sample:
        str_value = str(value)
        for pattern in DATE_PATTERNS:
            if re.fullmatch(pattern, str_value):
                match_count += 1
                break
                
    return (match_count / len(sample)) >= threshold

def auto_convert_dates(df):
    """
    Преобразует только колонки с датами из object в datetime
    """
    for col in df.columns:
        if df[col].dtype == 'object' and is_date_column(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Невалидные -> NaT
            except (ValueError, TypeError):
                continue
                
    return df

def process_chunk_with_dates(chunk):
    """Обрабатывает чанк данных с преобразованием только дат"""
    return auto_convert_dates(chunk)

def read_csv_in_chunks(file, chunk_size=10000):
    """Чтение CSV файла чанками"""
    return pd.read_csv(file, chunksize=chunk_size)


def allowed_file(filename: str) -> bool:
    """Проверяет, разрешён ли формат файла."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_chunk(chunk):
    """Обработка одного чанка данных"""
    return chunk


def read_file(file) -> pd.DataFrame:
    """
    Считывает файл и возвращает DataFrame.
    Поддерживаются форматы CSV, XLS, XLSX.
    Для CSV-файлов автоматически определяется разделитель (',' или ';').
    """
    filename = file.filename.lower()
    content = file.read()

    if filename.endswith('.csv'):
        try:
            s = content.decode('utf-8')
            first_line = s.splitlines()[0]
            if first_line.count(";") > first_line.count(","):
                delimiter = ";"
            else:
                delimiter = ","
            return pd.read_csv(io.StringIO(s), sep=delimiter)
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV: {str(e)}")
    elif filename.endswith('.xls') or filename.endswith('.xlsx'):
        try:
            return pd.read_excel(io.BytesIO(content))
        except Exception as e:
            raise ValueError(f"Ошибка чтения Excel: {str(e)}")
    else:
        raise ValueError("Неподдерживаемый формат файла")


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """
    Выполняет анализ DataFrame и возвращает словарь с характеристиками:
      - данные: предварительный просмотр (первые 5 строк)
      - столбцы: список названий столбцов
      - общее количество строк и столбцов
      - типы данных столбцов
      - количество пропущенных значений по столбцам
      - количество дублирующих значений по столбцам
    """
    data = df.head(5).values.tolist()
    columns = df.columns.tolist()
    total_rows = df.shape[0]
    total_columns = df.shape[1]
    data_types = df.dtypes.astype(str).to_dict()
    missing_data = df.isna().sum().to_dict()
    amount_nulls = df.isnull().sum().sum()
    amount_str = df.isnull().any(axis=1).sum()
    total_duplicates = int(df.duplicated().sum())

    return {
        "data": data,
        "columns": columns,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "data_types": data_types,
        "missing_data": missing_data,
        "amount_nulls": amount_nulls,
        "amount_str": amount_str,
        "total_duplicates": total_duplicates
    }

# НОВОЕ
def remove_missing_from_df(df: pd.DataFrame):
    """
    Удаляет все строки с пропущенными значениями из DataFrame.
    
    После удаления проверяет, что в результирующем DataFrame отсутствуют пропуски.
    
    Возвращает обновлённый DataFrame и сообщение об успехе, либо сообщение об ошибке, 
    если в DataFrame всё ещё остаются пропуски.
    
    :param df: pandas.DataFrame, исходный DataFrame
    :return: tuple (обновлённый DataFrame, сообщение)
    """

    # Удаляем строки с пропущенными значениями
    df = df.dropna()
    
    # Проверяем, что в обновлённом DataFrame нет пропусков
    if df.isnull().sum().sum() == 0:
        message = "Пропуски удалены"
    else:
        message = "Ошибка: в обновлённом DataFrame остаются пропуски"
    
    return df, message

def remove_duplicates_from_df(df: pd.DataFrame):
    """
    Удаляет все строки-дубликаты из DataFrame.
    
    После удаления проверяет, что в результирующем DataFrame отсутствуют дубликаты.
    
    Возвращает обновлённый DataFrame и сообщение об успехе, либо сообщение об ошибке, 
    если в DataFrame всё ещё остаются дубликаты.
    
    :param df: pandas.DataFrame, исходный DataFrame
    :return: tuple (обновлённый DataFrame, сообщение)
    """

    id_column = None
    if 'id' in df.columns:
        id_column = 'id'
    elif 'ID' in df.columns:
        id_column = 'ID'

    if id_column is not None:
        df = df.drop_duplicates(subset=id_column, keep='last')

    df = df.drop_duplicates()
    
    if df.duplicated().sum() == 0:
        message = "Дубликаты удалены"
    else:
        message = "Ошибка: в обновлённом DataFrame остаются дубликаты"
    
    return df, message



def fill_missing_values(
    df: pd.DataFrame,
    categorical_method: str = 'most_frequent',
    quantitative_method: str = 'mean',
    n_neighbors: int = 5,
    **kwargs
) -> pd.DataFrame:
    """
    Универсальная функция для заполнения пропусков с поддержкой KNN-импутации
    
    Параметры:
        df: исходный DataFrame
        categorical_method: метод для категориальных данных 
            ('unknown', 'most_frequent')
        quantitative_method: метод для числовых данных
            ('mean', 'median', 'zero', 'knn')
        n_neighbors: количество соседей для KNN (только при quantitative_method='knn')
        **kwargs: дополнительные параметры для StandardScaler
    
    Возвращает:
        DataFrame с заполненными пропусками
    """
    # Создаем копию DataFrame для безопасной обработки
    df_filled = df.copy()
    
    # 1. Обработка категориальных признаков
    cat_cols = df_filled.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_filled[col].isnull().any():
            if categorical_method == 'unknown':
                df_filled[col] = df_filled[col].fillna('Неизвестен')
            elif categorical_method == 'most_frequent' or categorical_method == 'knn':
                mode = df_filled[col].mode()
                df_filled[col] = df_filled[col].fillna(
                    mode.iloc[0] if not mode.empty else 'Неизвестен'
                )
    
    # 2. Обработка числовых признаков
    num_cols = df_filled.select_dtypes(include=np.number).columns
    
    if not num_cols.empty:
        if quantitative_method == 'knn':
            # ===== KNN Imputation =====
            X_num = df_filled[num_cols].values.astype(float)
            
            # Масштабирование
            scaler = StandardScaler(**kwargs)
            X_scaled = scaler.fit_transform(X_num)
            
            # Маска пропусков
            mask_nan = np.isnan(X_scaled)
            
            # Временное заполнение средними для построения дерева
            col_means = np.nanmean(X_scaled, axis=0)
            X_init = np.where(mask_nan, col_means, X_scaled)
            
            # Построение Ball-Tree
            nn = NearestNeighbors(
                n_neighbors=n_neighbors + 1,
                algorithm='ball_tree',
                metric='euclidean'
            )
            nn.fit(X_init)
            
            # Поиск соседей
            _, indices = nn.kneighbors(X_init)
            
            # Заполнение пропусков
            X_filled = X_scaled.copy()
            for i in range(X_scaled.shape[0]):
                miss = mask_nan[i]
                if not miss.any():
                    continue
                
                nbrs = indices[i, 1:]  # Исключаем саму точку
                for j in np.where(miss)[0]:
                    X_filled[i, j] = np.nanmean(X_init[nbrs, j])
            
            # Обратное преобразование
            df_filled[num_cols] = scaler.inverse_transform(X_filled)
            
        else:
            # Стандартные методы заполнения
            for col in num_cols:
                if quantitative_method == 'median':
                    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
                elif quantitative_method == 'mean':
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                elif quantitative_method == 'zero':
                    df_filled[col] = df_filled[col].fillna(0)
    
    return df_filled