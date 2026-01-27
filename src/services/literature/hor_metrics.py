import re
from typing import Dict, Any, List, Optional, Tuple

VALUE_RE = r"[-+]?\d+(?:\.\d+)?"

METRIC_PATTERNS = {
    "exchange_current_density": [
        rf"(exchange current density|j0|j_0)\s*(?:=|:|is)?\s*({VALUE_RE})\s*(mA\s*/\s*cm2|mA\s*cm-2|A\s*/\s*cm2|A\s*cm-2)",
    ],
    "mass_activity": [
        rf"(mass activity|ma)\s*(?:=|:|is)?\s*({VALUE_RE})\s*(A\s*/\s*mg|A\s*mg-1|A\s*/\s*g|A\s*g-1|mA\s*/\s*mg)",
    ],
    "specific_activity": [
        rf"(specific activity|jk|j_k)\s*(?:=|:|is)?\s*({VALUE_RE})\s*(mA\s*/\s*cm2|mA\s*cm-2|A\s*/\s*cm2|A\s*cm-2)",
    ],
    "overpotential": [
        rf"(overpotential)\s*(?:=|:|is)?\s*({VALUE_RE})\s*(mV)",
    ],
    "tafel_slope": [
        rf"(tafel slope)\s*(?:=|:|is)?\s*({VALUE_RE})\s*(mV\s*/\s*dec|mV\s*dec-1)",
    ],
}

FORMULA_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*){1,6}\b")


def _to_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _match_metric(text: str, patterns: List[str]) -> Optional[Tuple[float, str]]:
    text = text or ""
    for pat in patterns:
        for match in re.finditer(pat, text, flags=re.IGNORECASE):
            val = _to_float(match.group(2))
            unit = match.group(3)
            if val is not None:
                return val, unit
    return None


def extract_hor_metrics(text: str) -> Dict[str, Any]:
    """Extract HOR metrics from text (best-effort regex)."""
    out: Dict[str, Any] = {}
    if not text:
        return out
    for key, patterns in METRIC_PATTERNS.items():
        found = _match_metric(text, patterns)
        if found:
            val, unit = found
            out[key] = {"value": val, "unit": unit}
    return out


def extract_formulas(text: str, allowed_elements: Optional[set] = None, min_elements: int = 2) -> List[str]:
    """Extract candidate formulas containing only allowed elements."""
    if not text:
        return []
    allowed_elements = allowed_elements or set()
    formulas = []
    for match in FORMULA_RE.finditer(text):
        formula = match.group(0)
        parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        if not parts:
            continue
        elements = {p[0] for p in parts}
        if allowed_elements and not elements.issubset(allowed_elements):
            continue
        if min_elements and len(elements) < min_elements:
            continue
        formulas.append(formula)
    # de-dup preserving order
    seen = set()
    unique = []
    for f in formulas:
        if f in seen:
            continue
        seen.add(f)
        unique.append(f)
    return unique
