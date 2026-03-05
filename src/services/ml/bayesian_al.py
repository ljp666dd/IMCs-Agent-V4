"""
IMCs Bayesian Active Learning Service (V5 - Phase A)

Implements acquisition-function-driven active learning:
1. Gaussian Process surrogate model for uncertainty estimation
2. Expected Improvement (EI) and Probability of Improvement (PI) acquisition functions
3. Optimal exploration-exploitation trade-off for catalyst discovery

Replaces the simple P90 percentile threshold used in previous versions.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from src.core.logger import get_logger

logger = get_logger(__name__)

# Attempt to import sklearn GP
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    from scipy.stats import norm
    HAS_GP = True
except ImportError:
    HAS_GP = False
    logger.warning("sklearn/scipy not available for Bayesian AL. Falling back to heuristic.")


@dataclass
class ALCandidate:
    """A candidate scored by the acquisition function."""
    material_id: str
    formula: str = ""
    predicted_activity: float = 0.0
    uncertainty: float = 0.0
    ei_score: float = 0.0
    pi_score: float = 0.0
    acquisition_score: float = 0.0
    cost_penalty: float = 0.0
    is_pareto_optimal: bool = False
    reason: str = ""


class BayesianALService:
    """
    Bayesian Active Learning Service.

    Uses a Gaussian Process to model the performance landscape and
    acquisition functions to select the most informative candidates
    for experimental validation.
    """

    def __init__(self, strategy: str = "EI", xi: float = 0.01):
        """
        Args:
            strategy: Acquisition function - 'EI' (Expected Improvement) or 'PI' (Probability of Improvement)
            xi: Exploration-exploitation trade-off parameter.
                  Higher xi = more exploration, lower xi = more exploitation.
        """
        self.strategy = strategy
        self.xi = xi
        self.gp: Optional[GaussianProcessRegressor] = None
        self.is_fitted = False
        self.best_observed = -np.inf

        if HAS_GP:
            kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                normalize_y=True,
                alpha=1e-6
            )
            logger.info(f"BayesianALService initialized with {strategy} strategy (xi={xi}).")
        else:
            logger.warning("BayesianALService running in fallback mode (no GP).")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the GP surrogate model on observed data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,), e.g. measured activity
        """
        if not HAS_GP or self.gp is None:
            logger.warning("Cannot fit GP: sklearn not available.")
            return

        if len(X) < 2:
            logger.warning("Need at least 2 data points to fit GP.")
            return

        self.gp.fit(X, y)
        self.best_observed = np.max(y)
        self.is_fitted = True
        logger.info(f"GP fitted on {len(X)} samples. Best observed: {self.best_observed:.4f}")

    def score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        feature_extractor=None
    ) -> List[ALCandidate]:
        """
        Score candidates using acquisition function.

        Args:
            candidates: List of candidate dicts with 'material_id', 'formula', 'properties'.
            feature_extractor: Optional callable(candidate) -> np.array of features.

        Returns:
            Sorted list of ALCandidates (highest acquisition score first).
        """
        if not self.is_fitted or not HAS_GP:
            return self._fallback_scoring(candidates)

        from src.services.theory.market_data import get_market_data
        md_service = get_market_data()

        results = []
        for cand in candidates:
            mat_id = cand.get("material_id", "unknown")
            formula = cand.get("formula", "")
            props = cand.get("properties", {})

            # Extract features
            if feature_extractor:
                x = feature_extractor(cand)
            else:
                x = self._default_features(props)

            if x is None:
                continue

            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            mu = mu[0]
            sigma = max(sigma[0], 1e-9)

            ei = self._expected_improvement(mu, sigma)
            pi = self._probability_of_improvement(mu, sigma)

            acq = ei if self.strategy == "EI" else pi

            comp_dict = self._parse_formula(formula)
            cost_pen = md_service.get_cost_penalty(comp_dict)
            
            # Combine or keep separate for MO
            # We will use acquisition_score as the pure activity score, 
            # and use pareto front to flag true multi-objective winners.
            # 
            # A simple scalarized acq is: acq = acq * (1.0 - cost_pen * 0.5)
            # but we keep acq pristine for Pareto calculation.

            results.append(ALCandidate(
                material_id=mat_id,
                formula=formula,
                predicted_activity=float(mu),
                uncertainty=float(sigma),
                ei_score=float(ei),
                pi_score=float(pi),
                acquisition_score=float(acq),
                cost_penalty=float(cost_pen),
                reason=self._generate_reason(mu, sigma, ei, pi, cost_pen)
            ))

        # Calculate Pareto Front (Max Acq, Min Cost)
        results = self._mark_pareto_front(results)
        
        # Sort primarily by Pareto rank, then by acquisition score
        results.sort(key=lambda c: (not c.is_pareto_optimal, -c.acquisition_score))
        return results

    def select_for_experiment(
        self,
        candidates: List[Dict[str, Any]],
        n_select: int = 5,
        feature_extractor=None
    ) -> List[ALCandidate]:
        """
        Select top-N candidates for experimental validation.

        Args:
            candidates: Candidate list.
            n_select: Number to select.
            feature_extractor: Optional feature extractor.

        Returns:
            Top-N ALCandidates.
        """
        scored = self.score_candidates(candidates, feature_extractor)
        
        # In V6, we prefer Pareto optimal candidates
        pareto_front = [c for c in scored if c.is_pareto_optimal]
        non_pareto = [c for c in scored if not c.is_pareto_optimal]
        
        selected = (pareto_front + non_pareto)[:n_select]
        logger.info(f"Selected {len(selected)} candidates via MO-AL ({len(pareto_front)} in Pareto front).")
        return selected

    # ---------- Acquisition Functions ----------

    def _expected_improvement(self, mu: float, sigma: float) -> float:
        """
        Expected Improvement (EI).
        EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
        """
        improvement = mu - self.best_observed - self.xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return max(ei, 0.0)

    def _probability_of_improvement(self, mu: float, sigma: float) -> float:
        """
        Probability of Improvement (PI).
        PI(x) = Phi((mu - f_best - xi) / sigma)
        """
        Z = (mu - self.best_observed - self.xi) / sigma
        return float(norm.cdf(Z))

    # ---------- Helpers ----------

    def _default_features(self, props: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract a feature vector from material properties."""
        feature_keys = [
            "formation_energy", "d_band_center", "predicted_activity",
            "hydrogen_binding_energy", "overpotential", "ml_score"
        ]
        values = []
        for k in feature_keys:
            v = props.get(k)
            if v is not None:
                try:
                    values.append(float(v))
                except (ValueError, TypeError):
                    values.append(0.0)
            else:
                values.append(0.0)

        if all(v == 0.0 for v in values):
            return None

        return np.array(values)

    def _generate_reason(self, mu: float, sigma: float, ei: float, pi: float) -> str:
        """Generate a human-readable reason for selection."""
        parts = []
        if ei > 0.1:
            parts.append(f"预期改进量显著 (EI={ei:.4f})")
        if sigma > 0.3:
            parts.append(f"模型不确定度高 (σ={sigma:.4f})，值得探索")
        if mu > self.best_observed:
            parts.append(f"预测活性 ({mu:.4f}) 超过当前最优 ({self.best_observed:.4f})")
        if pi > 0.7:
            parts.append(f"改进概率高 (PI={pi:.2%})")

        return "；".join(parts) if parts else f"综合采集得分: EI={ei:.4f}, PI={pi:.4f}"

    def _parse_formula(self, formula: str) -> Dict[str, float]:
        """Simple regex-based formula parser for cost estimation (e.g. Pt3Ni -> {'Pt':3, 'Ni':1})."""
        import re
        comp = {}
        # Match element followed by optional numbers/decimals
        matches = re.findall(r'([A-Z][a-z]*)([\d\.]*)', formula)
        for el, amt in matches:
            if not amt:
                val = 1.0
            else:
                try:
                    val = float(amt)
                except ValueError:
                    val = 1.0
            comp[el] = comp.get(el, 0.0) + val
        return comp

    def _mark_pareto_front(self, candidates: List[ALCandidate]) -> List[ALCandidate]:
        """
        Identify candidates on the Pareto front.
        Objective 1: Maximize acquisition_score (Activity + Uncertainty)
        Objective 2: Minimize cost_penalty
        """
        for i, c1 in enumerate(candidates):
            c1.is_pareto_optimal = True
            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                # c2 dominates c1 if c2 is better or equal in ALL objectives, AND strictly better in AT LEAST ONE.
                better_or_eq_acq = c2.acquisition_score >= c1.acquisition_score
                better_or_eq_cost = c2.cost_penalty <= c1.cost_penalty
                
                strictly_better_acq = c2.acquisition_score > c1.acquisition_score
                strictly_better_cost = c2.cost_penalty < c1.cost_penalty
                
                if better_or_eq_acq and better_or_eq_cost and (strictly_better_acq or strictly_better_cost):
                    c1.is_pareto_optimal = False
                    break
                    
        return candidates

    def _fallback_scoring(self, candidates: List[Dict[str, Any]]) -> List[ALCandidate]:
        """Fallback when GP is not available: use simple uncertainty ranking."""
        results = []
        for cand in candidates:
            props = cand.get("properties", {})
            uncertainty = props.get("uncertainty", 0)
            activity = props.get("predicted_activity", 0)

            results.append(ALCandidate(
                material_id=cand.get("material_id", "unknown"),
                formula=cand.get("formula", ""),
                predicted_activity=float(activity),
                uncertainty=float(uncertainty),
                acquisition_score=float(uncertainty),
                reason=f"启发式模式: 不确定度={uncertainty:.4f}"
            ))

        results.sort(key=lambda c: c.acquisition_score, reverse=True)
        return results


def get_bayesian_al_service(strategy: str = "EI") -> BayesianALService:
    """Factory function."""
    return BayesianALService(strategy=strategy)
