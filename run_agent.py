import os
import json
import pandas as pd

from ai_agent import generate_agent_report


def main():
    data_path = os.getenv('TEST_DATA_PATH', 'test_data.csv')
    target = os.getenv('TARGET_COLUMN', '') or None

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    if not target:
        target = df.columns[-1]

    print(f"Running AI agent on {data_path} with target '{target}'...\n")
    # В офлайн-режиме журнал предобработки не используется, поэтому передаём None
    result = generate_agent_report(df, target, use_deepseek=False, preprocessing_log=None)

    out_path = 'final_report.json'

    def _serialize(obj):
        # Convert numpy/pandas types to native Python types for JSON
        try:
            import numpy as _np
            import pandas as _pd
        except Exception:
            _np = None
            _pd = None

        if _pd is not None and isinstance(obj, _pd.DataFrame):
            return obj.to_dict()
        if _pd is not None and isinstance(obj, _pd.Series):
            return obj.tolist()
        if _np is not None and isinstance(obj, (_np.integer, _np.int64, _np.int32)):
            return int(obj)
        if _np is not None and isinstance(obj, (_np.floating, _np.float64, _np.float32)):
            return float(obj)
        if isinstance(obj, bytes):
            try:
                return obj.decode('utf-8')
            except Exception:
                return str(obj)
        # Fallback to str
        return str(obj)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=_serialize)

    print(f"Local recommendations saved to {out_path}. Summary:\n")
    print(json.dumps({'target': target, 'local_recommendations': result['local_recommendations']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
