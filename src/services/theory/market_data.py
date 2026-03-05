"""
IMCs Market Data Service (V6 - Phase II)

Provides elemental abundance and spot price data for calculating catalyst costs.
Essential for Multi-Objective Pareto optimization (Activity vs. Cost).
"""

import math
from typing import Dict
from src.core.logger import get_logger

logger = get_logger(__name__)

# Mocked or static fallback spot prices (USD/kg) as of ~2024
# For a production system this could fetch from an external commodity API
ELEMENT_PRICES_USD_KG: Dict[str, float] = {
    "Pt": 30000.0,  # Highly expensive PGM
    "Pd": 32000.0,
    "Ir": 150000.0, # Ir is extremely expensive
    "Ru": 14000.0,
    "Rh": 145000.0,
    "Au": 65000.0,
    
    "Ni": 18.0,     # Transition metals are cheap
    "Co": 30.0,
    "Fe": 0.1,
    "Cu": 8.5,
    "Zn": 2.5,
    "Ag": 750.0,
    
    "Cr": 9.0,
    "Mn": 2.0,
    "Mo": 45.0,
    "W": 35.0,
    "Ti": 10.0,
    "V": 25.0,
    "C": 0.5,       # Carbon support
    "N": 0.2
}

class MarketDataService:
    def __init__(self):
        self.prices = ELEMENT_PRICES_USD_KG
        logger.info("MarketDataService initialized with elemental price dict.")

    def get_element_price(self, el: str) -> float:
        """Return price in USD/kg for a single element. Defaults to an average transition metal price if unknown."""
        return self.prices.get(el, 50.0)

    def estimate_formula_cost(self, comp_dict: Dict[str, float]) -> float:
        """
        Estimate the raw material cost of a generic formula (e.g., {'Pt': 0.75, 'Ni': 0.25})
        comp_dict represents the atomic fractions (or counts).
        Returns a normalized cost score.
        """
        if not comp_dict:
            return 99999.0 # Infinity penalty
            
        total_atoms = sum(comp_dict.values())
        if total_atoms <= 0:
            return 99999.0
            
        cost = 0.0
        for el, amt in comp_dict.items():
            cost += self.get_element_price(el) * (amt / total_atoms)
            
        return cost

    def get_cost_penalty(self, comp_dict: Dict[str, float]) -> float:
        """
        Calculate a logarithmic cost penalty from 0.0 (cheap) to 1.0 (extremely expensive).
        Used by the Bayesian AL acquisition function.
        """
        raw_cost = self.estimate_formula_cost(comp_dict)
        # Log scaling: base 10
        # Assuming Fe is ~0.1 USD/kg (penalty 0), Ir is 150000 USD/kg (penalty 1)
        try:
            log_cost = math.log10(raw_cost + 1.0) # +1 to avoid log(0) or negatives
            # Max expected log is log10(150000) ~ 5.17
            penalty = min(max(log_cost / 5.2, 0.0), 1.0)
            return penalty
        except Exception:
            return 1.0

def get_market_data() -> MarketDataService:
    return MarketDataService()
