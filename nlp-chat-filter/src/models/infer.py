import json
import joblib
from typing import List, Optional, Union
from .types import Prediction

def _safe_predict_proba(model, texts: List[str]) -> Optional[List[List[float]]]:
    try:
        return model.predict_proba(texts).tolist()
    except Exception:
        return None
    
def load_router(model_path: str = "models/artifacts/best_text_router.joblib",
                label_map_path: str = "models/artifacts/label_map.json"):
    mdl = joblib.load(model_path)
    with open(label_map_path, "r", encoding="utf-8") as f:
        lbl = {int(k): v for k, v in json.load(f).items()}
        
    def infer(texts: Union[str, List[str]]) -> List[Prediction]:
        batch = [texts] if isinstance(texts, str) else list(texts)
        y = mdl.predict(batch)
        proba = _safe_predict_proba(mdl, batch)
        out = []
        for i, yi in enumerate(y):
            out.append(Prediction(int(yi), lbl[int(yi)], proba[i] if proba else None))
        return out

    return infer