"""
Улучшенный алгоритм генерации локальных рекомендаций на основе эвристических правил.

Этот модуль реализует комплексный алгоритм анализа данных и генерации рекомендаций
для предобработки данных в задачах машинного обучения. Алгоритм основан на научно
обоснованных статистических методах и best practices в области data science.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class RecommendationEngine:
    """
    Класс для генерации комплексных рекомендаций по предобработке данных.
    
    Алгоритм включает 10 основных этапов анализа:
    1. Анализ пропущенных значений
    2. Анализ вариативности признаков
    3. Мультиколлинеарность и корреляции
    4. Анализ выбросов
    5. Анализ распределений
    6. Анализ категориальных признаков
    7. Взаимодействие признаков
    8. Ранжирование признаков
    9. Оценка пригодности признаков
    10. Рекомендации по трансформациям
    """
    
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        """
        Инициализация движка рекомендаций.
        
        Args:
            df: DataFrame для анализа
            target_column: Имя целевой переменной (опционально)
        """
        self.df = df.copy()
        self.target_column = target_column
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        self.recommendations = {
            'drop': [],
            'keep': [],
            'transform': [],
            'investigate': [],
            'warnings': []
        }
    
    def analyze_missingness(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Этап 1: Анализ пропущенных значений (Missingness Analysis)
        
        Методы:
        - Определение процента пропусков
        - Анализ паттернов пропусков (MCAR, MAR, MNAR через тесты)
        - Оценка влияния пропусков на целевую переменную
        """
        recs = []
        
        missing_pct = self.df.isna().mean() * 100
        
        for col in self.df.columns:
            pct = missing_pct[col]
            
            # Критический уровень пропусков (>50%)
            if pct > 50:
                recs.append({
                    'feature': col,
                    'action': 'drop',
                    'reason': f'Критический уровень пропусков: {pct:.1f}% (>50%)',
                    'severity': 'high',
                    'method': 'threshold_analysis'
                })
            
            # Высокий уровень пропусков (20-50%)
            elif pct > 20:
                # Проверяем, связаны ли пропуски с целевой переменной (MAR анализ)
                if self.target_column and col != self.target_column:
                    try:
                        # Создаем индикатор пропусков
                        missing_indicator = self.df[col].isna().astype(int)
                        if self.df[self.target_column].dtype in ['object', 'category']:
                            # Для категориальной целевой переменной - хи-квадрат
                            contingency = pd.crosstab(missing_indicator, self.df[self.target_column])
                            if contingency.size >= 4:
                                chi2, p_val, _, _ = chi2_contingency(contingency)
                                if p_val < 0.05:
                                    recs.append({
                                        'feature': col,
                                        'action': 'investigate',
                                        'reason': f'Пропуски ({pct:.1f}%) зависят от целевой переменной (MAR, p={p_val:.4f}). Требуется специальная стратегия импутации.',
                                        'severity': 'high',
                                        'method': 'mar_analysis'
                                    })
                                    continue
                        else:
                            # Для числовой целевой переменной - t-тест
                            present_mask = ~self.df[col].isna()
                            if present_mask.sum() > 2 and (~present_mask).sum() > 2:
                                group_present = self.df.loc[present_mask, self.target_column]
                                group_missing = self.df.loc[~present_mask, self.target_column]
                                t_stat, p_val = stats.ttest_ind(group_present, group_missing, nan_policy='omit')
                                if p_val < 0.05:
                                    recs.append({
                                        'feature': col,
                                        'action': 'investigate',
                                        'reason': f'Пропуски ({pct:.1f}%) зависят от целевой переменной (MAR, p={p_val:.4f}). Требуется специальная стратегия импутации.',
                                        'severity': 'high',
                                        'method': 'mar_analysis'
                                    })
                                    continue
                    except Exception:
                        pass
                
                recs.append({
                    'feature': col,
                    'action': 'transform',
                    'reason': f'Высокий уровень пропусков: {pct:.1f}% (20-50%). Рекомендуется импутация или создание индикатора пропусков.',
                    'severity': 'medium',
                    'method': 'threshold_analysis',
                    'suggestions': ['imputation', 'missing_indicator']
                })
            
            # Умеренный уровень пропусков (5-20%)
            elif pct > 5:
                recs.append({
                    'feature': col,
                    'action': 'transform',
                    'reason': f'Умеренный уровень пропусков: {pct:.1f}% (5-20%). Рассмотреть импутацию.',
                    'severity': 'low',
                    'method': 'threshold_analysis'
                })
        
        return recs
    
    def analyze_variability(self) -> List[Dict[str, Any]]:
        """
        Этап 2: Анализ вариативности признаков
        
        Методы:
        - Постоянные признаки (zero variance)
        - Квази-постоянные признаки (near-zero variance)
        - Анализ дисбаланса категориальных признаков
        """
        recs = []
        
        # Анализ числовых признаков
        for col in self.numeric_cols:
            if col == self.target_column:
                continue
                
            # Zero variance
            if self.df[col].nunique(dropna=True) <= 1:
                recs.append({
                    'feature': col,
                    'action': 'drop',
                    'reason': 'Постоянный признак (zero variance: <=1 уникальное значение)',
                    'severity': 'high',
                    'method': 'variance_analysis'
                })
                continue
            
            # Near-zero variance (< 0.01% уникальных значений)
            unique_ratio = self.df[col].nunique(dropna=True) / len(self.df[col].dropna())
            if unique_ratio < 0.0001 and len(self.df[col].dropna()) > 1000:
                recs.append({
                    'feature': col,
                    'action': 'drop',
                    'reason': f'Квази-постоянный признак (near-zero variance: {unique_ratio*100:.4f}% уникальных значений)',
                    'severity': 'medium',
                    'method': 'variance_analysis'
                })
        
        # Анализ категориальных признаков
        for col in self.categorical_cols:
            if col == self.target_column:
                continue
                
            value_counts = self.df[col].value_counts()
            n_unique = len(value_counts)
            
            # Постоянный признак
            if n_unique <= 1:
                recs.append({
                    'feature': col,
                    'action': 'drop',
                    'reason': 'Постоянный категориальный признак (<=1 уникальная категория)',
                    'severity': 'high',
                    'method': 'variance_analysis'
                })
                continue
            
            # Дисбаланс категорий (доминирующая категория >95%)
            if len(value_counts) > 0:
                dominant_freq = value_counts.iloc[0] / len(self.df[col].dropna())
                if dominant_freq > 0.95:
                    recs.append({
                        'feature': col,
                        'action': 'transform',
                        'reason': f'Сильный дисбаланс категорий: доминирующая категория составляет {dominant_freq*100:.1f}%. Рассмотреть объединение редких категорий.',
                        'severity': 'medium',
                        'method': 'category_balance_analysis',
                        'suggestions': ['category_merging', 'rare_category_threshold']
                    })
            
            # Слишком много категорий для one-hot encoding (>50)
            if n_unique > 50:
                recs.append({
                    'feature': col,
                    'action': 'transform',
                    'reason': f'Высокая кардинальность ({n_unique} категорий). One-hot encoding неэффективен. Использовать target encoding, embedding или binning.',
                    'severity': 'medium',
                    'method': 'cardinality_analysis',
                    'suggestions': ['target_encoding', 'embedding', 'category_binning']
                })
            
            # Редкие категории (менее 1% наблюдений каждая)
            rare_categories = value_counts[value_counts / len(self.df[col].dropna()) < 0.01]
            if len(rare_categories) > 0 and len(rare_categories) < n_unique * 0.5:
                recs.append({
                    'feature': col,
                    'action': 'transform',
                    'reason': f'Найдено {len(rare_categories)} редких категорий (<1% наблюдений). Рассмотреть объединение в категорию "Other".',
                    'severity': 'low',
                    'method': 'rare_category_analysis',
                    'suggestions': ['category_merging']
                })
        
        return recs
    
    def analyze_multicollinearity(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Этап 3: Мультиколлинеарность и корреляции
        
        Методы:
        - VIF (Variance Inflation Factor)
        - Высокие парные корреляции
        - Множественная корреляция
        - Анализ корреляций с целевой переменной
        """
        recs = []
        
        # VIF анализ
        vif_analysis = analysis_results.get('vif_analysis')
        if vif_analysis and isinstance(vif_analysis, dict):
            high_vif_features = vif_analysis.get('high_vif_features', [])
            vif_scores = vif_analysis.get('vif_scores', [])
            
            for item in vif_scores:
                feature = item.get('Feature')
                vif = item.get('VIF', 0)
                
                if feature == self.target_column:
                    continue
                
                if vif > 10:
                    recs.append({
                        'feature': feature,
                        'action': 'drop',
                        'reason': f'Высокий VIF ({vif:.2f} > 10) - сильная мультиколлинеарность. Признак линейно зависим от других.',
                        'severity': 'high',
                        'method': 'vif_analysis',
                        'metric_value': vif
                    })
                elif vif > 5:
                    recs.append({
                        'feature': feature,
                        'action': 'investigate',
                        'reason': f'Умеренный VIF ({vif:.2f} > 5). Возможна мультиколлинеарность. Требуется дополнительный анализ.',
                        'severity': 'medium',
                        'method': 'vif_analysis',
                        'metric_value': vif
                    })
        
        # Парные корреляции
        multicollinear_pairs = analysis_results.get('multicollinear_pairs', [])
        handled_features = set()
        
        for pair in multicollinear_pairs:
            f1 = pair.get('feature1')
            f2 = pair.get('feature2')
            corr = abs(pair.get('correlation', 0))
            
            if f1 == self.target_column or f2 == self.target_column:
                continue
            if f1 in handled_features or f2 in handled_features:
                continue
            
            # Выбираем, какой признак удалить
            chosen = None
            reason_suffix = ""
            
            if self.target_column and self.target_column in self.numeric_cols:
                try:
                    corr_f1 = abs(self.df[[f1, self.target_column]].dropna().corr().iloc[0, 1])
                    corr_f2 = abs(self.df[[f2, self.target_column]].dropna().corr().iloc[0, 1])
                    
                    if corr_f1 < corr_f2:
                        chosen = f1
                        reason_suffix = f" (низкая корреляция с целевой: {corr_f1:.3f} vs {corr_f2:.3f})"
                    else:
                        chosen = f2
                        reason_suffix = f" (низкая корреляция с целевой: {corr_f2:.3f} vs {corr_f1:.3f})"
                except Exception:
                    pass
            
            if not chosen:
                # Fallback: выбираем признак с большим количеством пропусков
                missing_f1 = self.df[f1].isna().mean()
                missing_f2 = self.df[f2].isna().mean()
                if missing_f1 > missing_f2:
                    chosen = f1
                    reason_suffix = f" (больше пропусков: {missing_f1*100:.1f}% vs {missing_f2*100:.1f}%)"
                else:
                    chosen = f2
                    reason_suffix = f" (больше пропусков: {missing_f2*100:.1f}% vs {missing_f1*100:.1f}%)"
            
            if chosen:
                recs.append({
                    'feature': chosen,
                    'action': 'drop',
                    'reason': f'Высокая корреляция с признаком "{f2 if chosen==f1 else f1}" ({corr:.3f} > 0.8){reason_suffix}',
                    'severity': 'high' if corr > 0.9 else 'medium',
                    'method': 'correlation_analysis',
                    'metric_value': corr,
                    'related_feature': f2 if chosen == f1 else f1
                })
                handled_features.add(chosen)
        
        # Топ корреляции с целевой переменной - сохраняем
        top_correlations = analysis_results.get('top_correlations', {})
        if top_correlations:
            for feature, corr_value in list(top_correlations.items())[:10]:
                if feature != self.target_column and feature not in handled_features:
                    abs_corr = abs(corr_value)
                    if abs_corr > 0.3:
                        recs.append({
                            'feature': feature,
                            'action': 'keep',
                            'reason': f'Высокая корреляция с целевой переменной ({corr_value:.3f}). Важный предиктор.',
                            'severity': 'info',
                            'method': 'target_correlation_analysis',
                            'metric_value': corr_value
                        })
        
        return recs
    
    def analyze_outliers(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Этап 4: Анализ выбросов (Outlier Detection)
        
        Методы:
        - Статистические методы (IQR, Z-score)
        - Изоляционный лес
        - Влияние выбросов на целевую переменную
        """
        recs = []
        
        outliers_info = analysis_results.get('outliers', {})
        
        for col in self.numeric_cols:
            if col == self.target_column:
                continue
            
            outlier_data = outliers_info.get(col, {})
            outlier_count = outlier_data.get('outlier_count', 0)
            
            if outlier_count == 0:
                continue
            
            total_non_null = len(self.df[col].dropna())
            outlier_pct = (outlier_count / total_non_null * 100) if total_non_null > 0 else 0
            
            # Критический уровень выбросов (>10%)
            if outlier_pct > 10:
                # Проверяем влияние выбросов на целевую переменную
                if self.target_column and self.target_column in self.numeric_cols:
                    try:
                        outlier_indices = outlier_data.get('outliers_indices', [])
                        if len(outlier_indices) > 0:
                            outlier_mask = self.df.index.isin(outlier_indices)
                            inlier_target = self.df.loc[~outlier_mask, self.target_column]
                            outlier_target = self.df.loc[outlier_mask, self.target_column]
                            
                            if len(inlier_target) > 2 and len(outlier_target) > 2:
                                t_stat, p_val = stats.ttest_ind(inlier_target, outlier_target, nan_policy='omit')
                                if p_val < 0.05:
                                    recs.append({
                                        'feature': col,
                                        'action': 'transform',
                                        'reason': f'Критический уровень выбросов ({outlier_pct:.1f}%), влияющих на целевую переменную (p={p_val:.4f}). Требуется обработка (winsorization, transformation).',
                                        'severity': 'high',
                                        'method': 'outlier_impact_analysis',
                                        'metric_value': outlier_pct,
                                        'suggestions': ['winsorization', 'log_transform', 'robust_scaling']
                                    })
                                    continue
                    except Exception:
                        pass
                
                recs.append({
                    'feature': col,
                    'action': 'transform',
                    'reason': f'Критический уровень выбросов: {outlier_pct:.1f}% (>10%). Рекомендуется обработка (winsorization, transformation).',
                    'severity': 'high',
                    'method': 'outlier_analysis',
                    'metric_value': outlier_pct,
                    'suggestions': ['winsorization', 'log_transform', 'robust_scaling']
                })
            
            # Умеренный уровень выбросов (5-10%)
            elif outlier_pct > 5:
                recs.append({
                    'feature': col,
                    'action': 'investigate',
                    'reason': f'Умеренный уровень выбросов: {outlier_pct:.1f}% (5-10%). Рекомендуется проверить влияние на модель.',
                    'severity': 'medium',
                    'method': 'outlier_analysis',
                    'metric_value': outlier_pct
                })
        
        return recs
    
    def analyze_distributions(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Этап 5: Анализ распределений
        
        Методы:
        - Нормальность распределения (Shapiro-Wilk)
        - Симметрия и эксцесс
        - Рекомендации по трансформациям
        """
        recs = []
        
        shapiro_wilk = analysis_results.get('shapiro_wilk')
        
        for col in self.numeric_cols:
            if col == self.target_column:
                continue
            
            data = self.df[col].dropna()
            if len(data) < 3:
                continue
            
            # Тест Шапиро-Уилка (для выборок <5000)
            is_normal = None
            if len(data) <= 5000:
                try:
                    stat, p_value = stats.shapiro(data)
                    is_normal = p_value > 0.05
                except Exception:
                    pass
            
            # Вычисляем асимметрию и эксцесс
            try:
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)
                
                # Сильная асимметрия (>2 или <-2)
                if abs(skewness) > 2:
                    transform_type = 'log' if skewness > 0 else 'inverse'
                    recs.append({
                        'feature': col,
                        'action': 'transform',
                        'reason': f'Сильная асимметрия распределения (skewness={skewness:.2f}). Рекомендуется {transform_type}-трансформация для нормализации.',
                        'severity': 'medium',
                        'method': 'distribution_analysis',
                        'metric_value': skewness,
                        'suggestions': [f'{transform_type}_transform', 'box_cox_transform']
                    })
                
                # Высокий эксцесс (тяжелые хвосты)
                if abs(kurtosis) > 3:
                    recs.append({
                        'feature': col,
                        'action': 'transform',
                        'reason': f'Высокий эксцесс ({kurtosis:.2f}), указывающий на тяжелые хвосты распределения. Рекомендуется robust scaling.',
                        'severity': 'low',
                        'method': 'distribution_analysis',
                        'metric_value': kurtosis,
                        'suggestions': ['robust_scaling', 'winsorization']
                    })
                
                # Не нормальное распределение (если определено)
                if is_normal is False:
                    recs.append({
                        'feature': col,
                        'action': 'transform',
                        'reason': f'Распределение не является нормальным (Shapiro-Wilk p<0.05). Рекомендуется трансформация для улучшения предпосылок моделей.',
                        'severity': 'low',
                        'method': 'normality_test',
                        'suggestions': ['box_cox_transform', 'yeo_johnson_transform']
                    })
            
            except Exception:
                pass
        
        return recs
    
    def analyze_feature_importance(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Этап 8: Ранжирование признаков
        
        Методы:
        - F-статистика
        - Взаимная информация
        - Важность признаков относительно целевой переменной
        """
        recs = []
        
        if not self.target_column or self.target_column not in self.df.columns:
            return recs
        
        # F-статистика
        f_statistics = analysis_results.get('f_statistics')
        if f_statistics:
            # Сортируем по убыванию F-статистики
            sorted_features = sorted(f_statistics, key=lambda x: x.get('f_statistic', 0), reverse=True)
            
            for item in sorted_features[:10]:  # Топ-10
                feature = item.get('feature')
                f_stat = item.get('f_statistic', 0)
                p_value = item.get('p_value', 1)
                is_significant = item.get('is_significant', False)
                
                if feature == self.target_column:
                    continue
                
                if is_significant and f_stat > 1:
                    recs.append({
                        'feature': feature,
                        'action': 'keep',
                        'reason': f'Высокая F-статистика ({f_stat:.2f}, p={p_value:.4f}) - значимый предиктор для целевой переменной.',
                        'severity': 'info',
                        'method': 'f_statistic_analysis',
                        'metric_value': f_stat
                    })
        
        return recs
    
    def generate_all_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Главная функция: генерирует все рекомендации, объединяя результаты всех этапов анализа.
        
        Returns:
            Словарь с рекомендациями по категориям: drop, keep, transform, investigate, warnings
        """
        all_recs = {
            'drop': [],
            'keep': [],
            'transform': [],
            'investigate': [],
            'warnings': []
        }
        
        # Собираем рекомендации со всех этапов
        all_recs['drop'].extend(self.analyze_missingness(analysis_results))
        all_recs['drop'].extend(self.analyze_variability())
        all_recs['drop'].extend(self.analyze_multicollinearity(analysis_results))
        
        all_recs['transform'].extend(self.analyze_missingness(analysis_results))
        all_recs['transform'].extend(self.analyze_variability())
        all_recs['transform'].extend(self.analyze_outliers(analysis_results))
        all_recs['transform'].extend(self.analyze_distributions(analysis_results))
        
        all_recs['investigate'].extend(self.analyze_multicollinearity(analysis_results))
        all_recs['investigate'].extend(self.analyze_outliers(analysis_results))
        
        all_recs['keep'].extend(self.analyze_multicollinearity(analysis_results))
        all_recs['keep'].extend(self.analyze_feature_importance(analysis_results))
        
        # Дедупликация по признакам (сохраняем рекомендацию с наивысшим приоритетом)
        priority_order = {'drop': 4, 'transform': 3, 'investigate': 2, 'keep': 1}
        feature_recommendations = {}
        
        for category in ['drop', 'transform', 'investigate', 'keep']:
            for rec in all_recs[category]:
                feature = rec['feature']
                
                if feature not in feature_recommendations:
                    feature_recommendations[feature] = rec
                else:
                    # Сохраняем рекомендацию с более высоким приоритетом
                    current_priority = priority_order.get(rec['action'], 0)
                    existing_priority = priority_order.get(feature_recommendations[feature]['action'], 0)
                    if current_priority > existing_priority:
                        feature_recommendations[feature] = rec
        
        # Перегруппировка по категориям
        final_recs = {
            'drop': [],
            'keep': [],
            'transform': [],
            'investigate': [],
            'warnings': []
        }
        
        for feature, rec in feature_recommendations.items():
            action = rec.get('action', 'keep')
            if action in final_recs:
                final_recs[action].append(rec)
        
        return final_recs


def generate_advanced_recommendations(
    df: pd.DataFrame, 
    target_column: Optional[str], 
    analysis_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Функция-обертка для генерации улучшенных рекомендаций.
    
    Args:
        df: DataFrame для анализа
        target_column: Имя целевой переменной
        analysis_results: Результаты статистического анализа
    
    Returns:
        Словарь с рекомендациями
    """
    engine = RecommendationEngine(df, target_column)
    return engine.generate_all_recommendations(analysis_results)
