import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, cramervonmises
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_regression, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from io import BytesIO
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# Настройка современных тонких шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.titleweight'] = 'light'
plt.rcParams['axes.labelweight'] = 'light'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Цветовая палитра
COLORS = {
    'primary': '#FFD1DC',  # Нежно-розовый
    'accent': '#FFB6C1',   # Розовый
    'dark_pink': '#FF69B4', # Темно-розовый
    'soft_blue': '#B5EAD7', # Мягкий бирюзовый
    'light_blue': '#C7CEEA', # Светло-голубой
    'text_dark': '#2F4F4F',  # Темно-серый текст
    'text_light': '#6c757d', # Светло-серый текст
    'background': "#FFFFFF",  # Белый фон
    'grid': "#e9ecef"       # Светло-серый для сетки
}

def get_column_statistics(df):
    """
    Получение статистики для каждого столбца
    """
    numeric_stats = {}
    categorical_stats = {}
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_stats[column] = {
                'count': df[column].count(),
                'mean': df[column].mean(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                '25%': df[column].quantile(0.25),
                '50%': df[column].quantile(0.50),
                '75%': df[column].quantile(0.75)
            }
        else:
            value_counts = df[column].value_counts()
            categorical_stats[column] = {
                'unique_count': df[column].nunique(),
                'top_values': value_counts.head(5).to_dict()
            }
    
    return numeric_stats, categorical_stats

def create_histogram_plot(df, column):
    """
    Создание минималистичной гистограммы с тонким шрифтом
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Гистограмма с прозрачностью
    n, bins, patches = ax.hist(df[column].dropna(), bins=30, 
                              color=COLORS['primary'], 
                              edgecolor=COLORS['accent'],
                              linewidth=0.8,
                              alpha=0.8)
    
    # Заголовок и метки с тонким шрифтом
    ax.set_title(f'{column}', fontsize=12, 
                fontweight='light', color=COLORS['text_dark'], pad=15)
    ax.set_xlabel('Значения', fontsize=10, color=COLORS['text_light'], weight='light')
    ax.set_ylabel('Частота', fontsize=10, color=COLORS['text_light'], weight='light')
    
    # Минималистичная сетка
    ax.grid(True, color=COLORS['grid'], alpha=0.6, linestyle='-', linewidth=0.5)
    
    # Убираем рамку
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Тонкие тики
    ax.tick_params(colors=COLORS['text_light'], which='both', labelsize=9)
    
    plt.tight_layout()
    
    # Конвертация
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), transparent=False)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64

def create_bar_plot(df, column):
    """
    Создание минималистичной столбчатой диаграммы с подписями категорий
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    top_values = df[column].value_counts().head(8)
    x_pos = np.arange(len(top_values))
    
    bars = ax.bar(x_pos, top_values.values, 
                 color=COLORS['soft_blue'], 
                 edgecolor=COLORS['light_blue'],
                 linewidth=0.8,
                 alpha=0.8,
                 width=0.7)
    
    ax.set_title(f'{column}', fontsize=12, fontweight='light', 
                color=COLORS['text_dark'], pad=15)
    ax.set_ylabel('Частота', fontsize=10, color=COLORS['text_light'], weight='light')
    ax.set_xlabel('Категории', fontsize=10, color=COLORS['text_light'], weight='light')
    
    # Добавляем подписи категорий с поворотом для читаемости
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_values.index, rotation=45, ha='right', 
                      fontsize=9, color=COLORS['text_light'], weight='light')
    
    # Минималистичная сетка
    ax.grid(True, color=COLORS['grid'], alpha=0.6, linestyle='-', 
           linewidth=0.5, axis='y')
    
    # Убираем рамку
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Тонкие тики
    ax.tick_params(colors=COLORS['text_light'], which='both', labelsize=9)
    
    # Автоматическая регулировка layout для подписей
    plt.tight_layout()
    
    # Конвертация
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64

def create_boxplot(df, column):
    """
    Создание минималистичного boxplot с тонким шрифтом
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    boxprops = dict(facecolor=COLORS['primary'], color=COLORS['accent'], 
                   linewidth=1.2, alpha=0.8)
    whiskerprops = dict(color=COLORS['text_light'], linewidth=1.2, alpha=0.8)
    capprops = dict(color=COLORS['text_light'], linewidth=1.2, alpha=0.8)
    medianprops = dict(color=COLORS['dark_pink'], linewidth=1.5, alpha=0.9)
    
    ax.boxplot(df[column].dropna(), patch_artist=True,
               boxprops=boxprops,
               whiskerprops=whiskerprops,
               capprops=capprops,
               medianprops=medianprops,
               widths=0.6)
    
    ax.set_title(f'{column}', fontsize=12, 
                fontweight='light', color=COLORS['text_dark'], pad=15)
    ax.set_ylabel('Значения', fontsize=10, color=COLORS['text_light'], weight='light')
    
    # Минималистичная сетка
    ax.grid(True, color=COLORS['grid'], alpha=0.6, linestyle='-', 
           linewidth=0.5, axis='y')
    
    # Убираем рамку
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Тонкие тики
    ax.tick_params(colors=COLORS['text_light'], which='both', labelsize=9)
    
    plt.tight_layout()
    
    # Конвертация
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64

def detect_outliers(df, column):
    """
    Обнаружение выбросов с использованием Isolation Forest
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {
            'outlier_count': 0,
            'lower_bound': None,
            'upper_bound': None,
            'outliers_indices': [],
            'method': 'Not applicable for categorical data'
        }
    
    data = df[column].dropna()
    if len(data) < 4:
        return {
            'outlier_count': 0,
            'lower_bound': None,
            'upper_bound': None,
            'outliers_indices': [],
            'method': 'Insufficient data'
        }
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers_pred = iso_forest.fit_predict(data.values.reshape(-1, 1))
    outlier_indices = data[outliers_pred == -1].index.tolist()
    outlier_count = len(outlier_indices)
    
    # Также вычисляем IQR границы для справки
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return {
        'outlier_count': outlier_count,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers_indices': outlier_indices,
        'method': 'Isolation Forest'
    }

def remove_outliers(df, column):
    """
    Удаление выбросов заменой на 1 и 99 процентиль
    """
    lower_percentile = df[column].quantile(0.01)
    upper_percentile = df[column].quantile(0.99)
    
    # Заменяем выбросы
    df_cleaned = df.copy()
    df_cleaned[column] = np.where(
        df_cleaned[column] < lower_percentile, 
        lower_percentile, 
        np.where(
            df_cleaned[column] > upper_percentile, 
            upper_percentile, 
            df_cleaned[column]
        )
    )
    
    return df_cleaned

def shapiro_wilk_test(df, target_column):
    """
    Тест Шапиро-Уилка на нормальность распределения
    """
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return None
    
    data = df[target_column].dropna()
    if len(data) < 3:
        return None
    
    stat, p_value = stats.shapiro(data)
    
    return {
        'statistic': stat,
        'p_value': p_value,
        'is_normal': p_value > 0.05
    }

def chi_square_test(df, col1, col2):
    """
    Критерий хи-квадрат для категориальных переменных
    """
    if col1 == col2:
        return None
    
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    if contingency_table.size < 4:
        return None
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        cramers_v = np.sqrt(chi2 / (contingency_table.sum().sum() * (min(contingency_table.shape) - 1)))
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'is_significant': p_value < 0.05,
            'contingency_table': contingency_table
        }
    except:
        return None

def cramer_v_coefficient(df, col1, col2):
    """
    Коэффициент Крамера для измерения связи между категориальными переменными
    """
    chi_square_result = chi_square_test(df, col1, col2)
    if chi_square_result:
        return chi_square_result['cramers_v']
    return None

def spearman_correlation(df, col1, col2):
    """
    Коэффициент корреляции Спирмена
    """
    if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2])):
        return None
    
    data1 = df[col1].dropna()
    data2 = df[col2].dropna()
    
    if len(data1) != len(data2) or len(data1) < 3:
        return None
    
    try:
        correlation, p_value = spearmanr(data1, data2)
        return {
            'correlation': correlation,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
    except:
        return None

def anova_test(df, categorical_col, numeric_col):
    """
    ANOVA для смешанных типов данных (категориальный + числовой)
    """
    if not (pd.api.types.is_numeric_dtype(df[numeric_col]) and 
            not pd.api.types.is_numeric_dtype(df[categorical_col])):
        return None
    
    groups = []
    group_names = []
    
    for category in df[categorical_col].dropna().unique():
        group_data = df[df[categorical_col] == category][numeric_col].dropna()
        if len(group_data) > 0:
            groups.append(group_data)
            group_names.append(str(category))
    
    if len(groups) < 2:
        return None
    
    try:
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Вычисляем eta-squared (мера эффекта)
        ss_between = sum(len(group) * (group.mean() - df[numeric_col].mean())**2 for group in groups)
        ss_total = ((df[numeric_col] - df[numeric_col].mean())**2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'is_significant': p_value < 0.05,
            'groups': {name: {'mean': group.mean(), 'std': group.std(), 'count': len(group)} 
                      for name, group in zip(group_names, groups)}
        }
    except:
        return None

def f_statistic_test(df, numeric_columns, target_column=None):
    """
    F-статистика для дисперсионного анализа
    """
    if not target_column or target_column not in numeric_columns:
        return None
    
    try:
        X = df[numeric_columns].dropna()
        y = df[target_column].dropna()
        
        if len(X) != len(y) or len(X) < 3:
            return None
        
        # F-статистика для регрессии
        f_scores, p_values = f_regression(X, y)
        
        results = []
        for feature, f_score, p_value in zip(numeric_columns, f_scores, p_values):
            if feature != target_column:
                results.append({
                    'feature': feature,
                    'f_statistic': f_score,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05
                })
        
        return sorted(results, key=lambda x: x['f_statistic'], reverse=True)
    except:
        return None

def calculate_vif(df, numeric_columns):
    """
    VIF (Variance Inflation Factor) для мультиколлинеарности
    """
    numeric_df = df[numeric_columns].dropna()
    
    if len(numeric_df) < 10 or len(numeric_columns) < 2:
        return None
    
    try:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_columns
        vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) 
                          for i in range(len(numeric_columns))]
        
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        return {
            'vif_scores': vif_data.to_dict('records'),
            'high_vif_features': vif_data[vif_data['VIF'] > 10]['Feature'].tolist()
        }
    except:
        return None

def categorical_relationship_analysis(df):
    """
    Анализ связей между категориальными признаками
    """
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    
    if len(categorical_columns) < 2:
        return None
    
    relationships = []
    
    for i, col1 in enumerate(categorical_columns):
        for col2 in categorical_columns[i+1:]:
            chi_result = chi_square_test(df, col1, col2)
            if chi_result:
                relationships.append({
                    'feature1': col1,
                    'feature2': col2,
                    'chi2_statistic': chi_result['chi2_statistic'],
                    'p_value': chi_result['p_value'],
                    'cramers_v': chi_result['cramers_v'],
                    'is_significant': chi_result['is_significant']
                })
    
    return sorted(relationships, key=lambda x: x['cramers_v'], reverse=True)

def mixed_type_analysis(df):
    """
    Смешанный анализ (числовой + категориальный)
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    
    if len(numeric_columns) == 0 or len(categorical_columns) == 0:
        return None
    
    mixed_results = []
    
    for cat_col in categorical_columns:
        for num_col in numeric_columns:
            anova_result = anova_test(df, cat_col, num_col)
            if anova_result:
                mixed_results.append({
                    'categorical_feature': cat_col,
                    'numeric_feature': num_col,
                    'f_statistic': anova_result['f_statistic'],
                    'p_value': anova_result['p_value'],
                    'eta_squared': anova_result['eta_squared'],
                    'is_significant': anova_result['is_significant'],
                    'groups_summary': anova_result['groups']
                })
    
    return sorted(mixed_results, key=lambda x: x['f_statistic'], reverse=True)

def multiple_correlation_analysis(df, numeric_columns):
    """
    Множественная корреляция
    """
    numeric_df = df[numeric_columns].dropna()
    
    if len(numeric_df) < 3:
        return None
    
    try:
        # Корреляционная матрица
        corr_matrix = numeric_df.corr()
        
        # Множественная корреляция для каждой переменной с остальными
        multiple_correlations = {}
        for col in numeric_columns:
            other_cols = [c for c in numeric_columns if c != col]
            if len(other_cols) > 0:
                # R-квадрат множественной корреляции
                corr_with_others = corr_matrix[col][other_cols].abs()
                max_corr = corr_with_others.max()
                mean_corr = corr_with_others.mean()
                
                multiple_correlations[col] = {
                    'max_correlation': max_corr,
                    'mean_correlation': mean_corr,
                    'highly_correlated': corr_with_others[corr_with_others > 0.7].to_dict()
                }
        
        return multiple_correlations
    except:
        return None

def check_multicollinearity(df, target_column=None):
    """
    Проверка на мультиколлинеарность и корреляции (расширенная версия)
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_df = df[numeric_columns]
    
    # Корреляционная матрица
    corr_matrix = numeric_df.corr()
    
    # Поиск сильно коррелирующих пар (коэффициент > 0.8)
    multicollinear_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                multicollinear_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    # Топ 10 признаков, коррелирующих с целевой переменной
    top_correlations = {}
    if target_column and target_column in numeric_columns:
        target_correlations = corr_matrix[target_column].drop(target_column, errors='ignore')
        if not target_correlations.empty:
            top_correlations = target_correlations.abs().sort_values(ascending=False).head(10).to_dict()
    
    # VIF анализ
    vif_results = calculate_vif(df, numeric_columns)
    
    # Множественная корреляция
    multiple_corr_results = multiple_correlation_analysis(df, numeric_columns)
    
    return {
        'correlation_matrix': corr_matrix,
        'multicollinear_pairs': multicollinear_pairs,
        'top_correlations': top_correlations,
        'vif_analysis': vif_results,
        'multiple_correlations': multiple_corr_results
    }

def create_correlation_heatmap(corr_matrix):
    """
    Создание минималистичной тепловой карты корреляционной матрицы
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor(COLORS['background'])
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Минималистичная тепловая карта
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, ax=ax, fmt='.2f',
                annot_kws={'size': 8, 'weight': 'light'},
                cbar_kws={'shrink': 0.8})
    
    ax.set_title('Матрица корреляций', fontsize=12, 
                fontweight='light', color=COLORS['text_dark'], pad=20)
    
    # Тонкие тики
    ax.tick_params(colors=COLORS['text_light'], which='both', labelsize=8)
    
    plt.tight_layout()
    
    # Конвертация в base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return image_base64

def perform_statistical_analysis(df, target_column=None):
    """
    Основная функция для выполнения полного статистического анализа (расширенная версия)
    """
    results = {}
    
    # 1. Базовая статистика
    numeric_stats, categorical_stats = get_column_statistics(df)
    results['numeric_stats'] = numeric_stats
    results['categorical_stats'] = categorical_stats
    
    # 2. Гистограммы для числовых признаков
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    results['histograms'] = {}
    for col in numeric_columns:
        results['histograms'][col] = create_histogram_plot(df, col)
    
    # 3. Столбчатые диаграммы для категориальных признаков
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    results['bar_plots'] = {}
    for col in categorical_columns:
        results['bar_plots'][col] = create_bar_plot(df, col)
    
    # 4. Boxplots и обнаружение выбросов (с Isolation Forest)
    results['boxplots'] = {}
    results['outliers'] = {}
    for col in numeric_columns:
        results['boxplots'][col] = create_boxplot(df, col)
        results['outliers'][col] = detect_outliers(df, col)
    
    # 5. Тест Шапиро-Уилка
    if target_column and target_column in numeric_columns:
        results['shapiro_wilk'] = shapiro_wilk_test(df, target_column)
    
    # 6. Проверка мультиколлинеарности (расширенная)
    multicollinearity_results = check_multicollinearity(df, target_column)
    results.update(multicollinearity_results)
    
    # 7. Тепловая карта корреляций (если признаков <= 10)
    if len(numeric_columns) <= 10:
        results['correlation_heatmap'] = create_correlation_heatmap(
            multicollinearity_results['correlation_matrix']
        )
    
    # 8. НОВЫЕ СТАТИСТИЧЕСКИЕ ТЕСТЫ
    
    # Критерий хи-квадрат и анализ связей между категориальными переменными
    if len(categorical_columns) >= 2:
        results['categorical_relationships'] = categorical_relationship_analysis(df)
    
    # ANOVA для смешанных типов данных
    if len(numeric_columns) > 0 and len(categorical_columns) > 0:
        results['mixed_type_analysis'] = mixed_type_analysis(df)
    
    # Коэффициент Спирмена для числовых переменных
    if len(numeric_columns) >= 2:
        spearman_results = {}
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                spearman_result = spearman_correlation(df, col1, col2)
                if spearman_result:
                    spearman_results[f"{col1}_vs_{col2}"] = spearman_result
        results['spearman_correlations'] = spearman_results
    
    # F-статистика дисперсионного анализа
    if target_column and target_column in numeric_columns:
        results['f_statistics'] = f_statistic_test(df, numeric_columns, target_column)
    
    return results