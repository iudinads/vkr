import os
import json
import math
import time
from typing import Dict, Any, Optional, List

import requests
import pandas as pd

from statistic_utils import perform_statistical_analysis


# DeepSeek API configuration
DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1/chat/completions').strip()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-a76883ed56594a129842c0c81e96790b').strip()


def detect_task_type(df: pd.DataFrame, target_column: str) -> str:
    """Определяем тип задачи: 'regression' или 'classification' (binary/multiclass).
    Простая эвристика:
    - если целевая колонка числовая и число уникальных значений > 20 -> regression
    - если числовая и уникальных <= 20 -> classification (категориальная числовая)
    - если не числовая -> classification
    """
    if target_column not in df.columns:
        return 'unknown'

    series = df[target_column]
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.dropna().nunique()
        if n_unique == 0:
            return 'unknown'
        if n_unique > 20:
            return 'regression'
        return 'classification'
    else:
        return 'classification'


def _safe_pct(x: float) -> float:
    try:
        return round(float(x) * 100.0, 2)
    except Exception:
        return 0.0


def local_recommendations(df: pd.DataFrame, target_column: Optional[str], analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate local heuristic recommendations about which features to drop or keep.
    Returns a dictionary with lists of recommendations and reasons.
    """
    recs = {'drop': [], 'keep': [], 'transform': []}

    # 1. Анализ пропусков (Missingness)
    missing = df.isna().mean()
    for col, frac in missing.items():
        pct = _safe_pct(frac)
        if pct > 50.0:
            recs['drop'].append({'feature': col, 'reason': f'missingness {pct}% > 50%'})
        elif pct > 20.0:
            # Исправление: добавлена проверка на наличие целевой переменной
            if target_column and target_column in df.columns:
                recs['transform'].append({'feature': col, 'reason': f'high missingness {pct}%, consider special imputation (depends on target)'})
            else:
                recs['transform'].append({'feature': col, 'reason': f'high missingness {pct}%, consider imputation'})
        else:  # pct <= 20%
            recs['transform'].append({'feature': col, 'reason': f'moderate missingness {pct}%, consider imputation or missing indicator'})

    # 2. Анализ вариативности (Constant or near-constant)
    for col in df.columns:
        try:
            n = len(df[col].dropna())
            if n == 0:
                recs['drop'].append({'feature': col, 'reason': 'all values missing'})
                continue
                
            nunique = df[col].nunique(dropna=True)
            unique_ratio = nunique / n if n > 0 else 0
            
            # Правило: количество уникальных значений < 2
            if nunique <= 1:
                recs['drop'].append({'feature': col, 'reason': 'constant feature (<=1 unique values)'})
            # Правило: доля уникальных значений < 0.01%
            elif unique_ratio < 0.0001:  # 0.01% = 0.0001
                recs['drop'].append({'feature': col, 'reason': f'very low unique ratio {unique_ratio:.6f} < 0.01%'})
            
            # Правило: доминирующая категория > 95% (для категориальных признаков)
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                value_counts = df[col].value_counts(normalize=True)
                if not value_counts.empty and value_counts.iloc[0] > 0.95:
                    recs['transform'].append({'feature': col, 'reason': f'dominant category >95%, consider merging rare categories'})
            
            # Правило: кардинальность категорий > 50
            if (df[col].dtype == 'object' or df[col].dtype.name == 'category') and nunique > 50:
                recs['transform'].append({'feature': col, 'reason': f'high cardinality ({nunique} categories), use special encoders'})
                
        except Exception as e:
            continue

    # 3. Анализ мультиколлинеарности (VIF)
    vif = analysis_results.get('vif_analysis') if isinstance(analysis_results, dict) else None
    if vif and isinstance(vif, dict):
        high_vif = vif.get('high_vif_features', [])
        for f in high_vif:
            recs['drop'].append({'feature': f, 'reason': 'high VIF (>10) - multicollinearity'})
        
        # Исправление: добавлена обработка VIF от 5 до 10
        moderate_vif = vif.get('moderate_vif_features', [])
        for f in moderate_vif:
            recs['transform'].append({'feature': f, 'reason': 'moderate VIF (5-10), investigate further'})

    # 4. Парная корреляция (Highly correlated pairs -> drop one)
    pairs = analysis_results.get('multicollinear_pairs', []) if isinstance(analysis_results, dict) else []
    for p in pairs:
        f1 = p['feature1']
        f2 = p['feature2']
        corr = abs(p.get('correlation', 0))
        
        # Проверяем, что корреляция > 0.8 (правило из таблицы)
        if corr > 0.8:
            chosen = None
            if target_column and target_column in df.columns:
                try:
                    corr_f1 = abs(df[[f1, target_column]].dropna().corr().iloc[0, 1])
                except Exception:
                    corr_f1 = 0
                try:
                    corr_f2 = abs(df[[f2, target_column]].dropna().corr().iloc[0, 1])
                except Exception:
                    corr_f2 = 0
                chosen = f1 if corr_f1 < corr_f2 else f2
            else:
                m1 = df[f1].isna().mean()
                m2 = df[f2].isna().mean()
                chosen = f1 if m1 > m2 else f2

            if chosen:
                recs['drop'].append({'feature': chosen, 'reason': f'pair correlation {corr:.2f} > 0.8 with {f2 if chosen==f1 else f1}'})

    # 5. Анализ выбросов (нужно добавить, если есть в analysis_results)
    outliers = analysis_results.get('outlier_analysis') if isinstance(analysis_results, dict) else None
    if outliers and isinstance(outliers, dict):
        for col, outlier_pct in outliers.items():
            if outlier_pct >= 10:
                if target_column and col != target_column:
                    # Проверяем влияние на целевую переменную
                    recs['transform'].append({'feature': col, 'reason': f'high outliers {outlier_pct}% >=10%, special treatment needed (check target impact)'})
                else:
                    recs['transform'].append({'feature': col, 'reason': f'high outliers {outlier_pct}% >=10%, handle outliers'})
            elif outlier_pct >= 5:
                recs['transform'].append({'feature': col, 'reason': f'moderate outliers {outlier_pct}% (5-10%), investigate impact'})

    # 6. Анализ распределений (нужно добавить, если есть в analysis_results)
    distributions = analysis_results.get('distribution_analysis') if isinstance(analysis_results, dict) else None
    if distributions and isinstance(distributions, dict):
        for col, dist_stats in distributions.items():
            skewness = dist_stats.get('skewness', 0)
            kurtosis = dist_stats.get('kurtosis', 0)
            
            if abs(skewness) >= 2:
                recs['transform'].append({'feature': col, 'reason': f'high skewness {skewness:.2f} >= 2, apply log/inverse transformation'})
            
            if abs(kurtosis) >= 3:
                recs['transform'].append({'feature': col, 'reason': f'high kurtosis {kurtosis:.2f} >= 3, use robust scaling'})

    # 7. Ранжирование признаков (Top correlations with target)
    if isinstance(analysis_results, dict) and 'top_correlations' in analysis_results and analysis_results['top_correlations']:
        top = list(analysis_results['top_correlations'].keys())[:10]
        for f in top:
            if f not in [r['feature'] for r in recs['drop']]:
                recs['keep'].append({'feature': f, 'reason': 'high correlation with target (statistically significant)'})

    # Deduplicate recommendations (by feature, keeping highest-priority reason)
    def unique_by_feature(lst: List[Dict[str, str]]):
        seen = {}
        for item in lst:
            feat = item['feature']
            if feat not in seen:
                seen[feat] = item
        return list(seen.values())

    recs['drop'] = unique_by_feature(recs['drop'])
    recs['keep'] = unique_by_feature(recs['keep'])
    recs['transform'] = unique_by_feature(recs['transform'])

    return recs



def build_prompt_summary(
    df: pd.DataFrame,
    target_column: Optional[str],
    analysis_results: Dict[str, Any],
    local_recs: Dict[str, Any],
    preprocessing_log: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build a concise prompt to send to the Deepseek API with context for a final report."""
    n_rows, n_cols = df.shape
    task_type = detect_task_type(df, target_column) if target_column else 'unknown'

    summary: Dict[str, Any] = {
        'rows': n_rows,
        'columns': n_cols,
        'target_column': target_column,
        'task_type': task_type,
        'numeric_count': int(len(df.select_dtypes(include=["number"]).columns)),
        'categorical_count': int(len(df.select_dtypes(exclude=["number"]).columns)),
        'top_local_recommendations': local_recs,
    }

    # attach top correlations if available (trimmed)
    if isinstance(analysis_results, dict) and analysis_results.get('top_correlations'):
        summary['top_correlations'] = {
            k: analysis_results['top_correlations'][k]
            for k in list(analysis_results['top_correlations'].keys())[:10]
        }

    # При наличии журнала предобработки добавляем его (обрезаем до последних шагов, чтобы не раздувать промпт)
    if preprocessing_log:
        # Берём последние 20 шагов, чтобы сохранить хронологию и не перегружать контекст
        summary['preprocessing_log'] = preprocessing_log[-20:]

    prompt = (
        "Ты — помощник по data science. На основе статистики датасета и результатов анализа создай финальный отчет по предобработке данных для машинного обучения. "
        "Сосредоточься на том, какие признаки удалить, а какие оставить, рекомендованных трансформациях (масштабирование, кодирование, заполнение пропусков) и советах по модели в зависимости от того, является ли задача регрессией или классификацией. "
        "Вывод должен быть структурирован: 1) краткое резюме (3-6 предложений), 2) список признаков для УДАЛЕНИЯ с причинами, 3) список признаков для СОХРАНЕНИЯ с причинами, 4) трансформации для применения, 5) следующие шаги и быстрые проверки. "
        "Отвечай строго на русском языке. Вот сводка датасета и анализа (JSON):\n" + json.dumps(summary, ensure_ascii=False, indent=2)
    )

    return prompt


def call_deepseek(prompt: str, max_tokens: int = 2000, timeout: int = 60) -> Optional[str]:
    """Send prompt to Deepseek API and return textual response.
    Uses OpenAI-compatible ChatCompletions format.
    """
    if not DEEPSEEK_API_URL or not DEEPSEEK_API_KEY:
        return None

    # DeepSeek uses OpenAI-compatible ChatCompletions API format
    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {
                'role': 'system',
                'content': 'You are a data science assistant. Provide clear, structured recommendations in Russian.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'max_tokens': max_tokens,
        'temperature': 0.7
    }
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            # Extract message content from OpenAI-compatible response
            if 'choices' in data and isinstance(data['choices'], list) and len(data['choices']) > 0:
                choice = data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    return choice['message']['content']
                elif 'text' in choice:
                    return choice['text']
            # Fallback: try other common keys
            for key in ('text', 'response', 'output', 'content'):
                if key in data:
                    return data[key] if isinstance(data[key], str) else json.dumps(data[key], ensure_ascii=False)
            # Last resort: return whole json as string
            return json.dumps(data, ensure_ascii=False, indent=2)
        else:
            error_text = resp.text
            return f"Deepseek API вернул статус {resp.status_code}: {error_text}"
    except requests.exceptions.Timeout:
        return "Превышено время ожидания ответа от Deepseek API"
    except Exception as e:
        return f"Ошибка при запросе к Deepseek API: {str(e)}"


def generate_agent_report(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    use_deepseek: bool = True,
    preprocessing_log: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Main entry: run analysis, make local recs and (optionally) consult Deepseek for a final narrative report.
    Returns a dictionary with keys: analysis_results, local_recommendations, deepseek_report (optional/string).
    """
    # Засекаем время работы модуля AI-агента:
    # от входа в generate_agent_report до формирования итогового словаря результата.
    t_start = time.perf_counter()

    analysis_results = perform_statistical_analysis(df, target_column)

    local_recs = local_recommendations(df, target_column, analysis_results)

    deepseek_report = None
    deepseek_start = None
    deepseek_end = None

    if use_deepseek:
        prompt = build_prompt_summary(
            df,
            target_column,
            analysis_results,
            local_recs,
            preprocessing_log=preprocessing_log
        )
        #-----test part
        # Для тестирования можно включить вывод финального промпта в логи:
        # export LOG_FINAL_PROMPT=true
        if os.getenv("LOG_FINAL_PROMPT", "false").strip().lower() in ("1", "true", "yes", "y", "on"):
            print("#-----test part")
            print(prompt)
            print("#-----test part")
        #-----test part

        # Отдельно измеряем время внешнего вызова DeepSeek:
        # от непосредственного отправления запроса до получения ответа.
        deepseek_start = time.perf_counter()
        deepseek_report = call_deepseek(prompt)
        deepseek_end = time.perf_counter()

    t_end = time.perf_counter()

    # Логируем время работы модуля AI-агента.
    # Общее время: весь путь внутри generate_agent_report (анализ + эвристики + DeepSeek, если включён).
    total_time = t_end - t_start
    print(f"[AI_AGENT_TIMING] total_generate_agent_report_time_sec={total_time:.3f}")

    # Если DeepSeek вызывался — логируем отдельное время сетевого вызова.
    if deepseek_start is not None and deepseek_end is not None:
        deepseek_time = deepseek_end - deepseek_start
        print(f"[AI_AGENT_TIMING] deepseek_call_time_sec={deepseek_time:.3f}")

    return {
        'analysis_results': analysis_results,
        'local_recommendations': local_recs,
        'deepseek_report': deepseek_report,
        'preprocessing_log': preprocessing_log
    }


if __name__ == '__main__':
    # Quick demo when invoked directly. Uses TEST_DATA_PATH env or test_data.csv in repo root.
    import sys

    data_path = os.getenv('TEST_DATA_PATH', 'test_data.csv')
    target = os.getenv('TARGET_COLUMN', '') or None

    if not os.path.exists(data_path):
        print(f"Demo data not found at {data_path}. Set TEST_DATA_PATH env or place test_data.csv in project root.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    # If target not specified, try last column
    if not target:
        target = df.columns[-1]

    result = generate_agent_report(df, target, use_deepseek=False)
    print(json.dumps({'target_column': target, 'local_recommendations': result['local_recommendations']}, ensure_ascii=False, indent=2))
