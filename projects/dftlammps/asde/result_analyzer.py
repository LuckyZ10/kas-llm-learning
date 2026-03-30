"""
Result Analyzer Module

Provides statistical analysis, hypothesis testing, and effect size calculation
for experimental results. Implements comprehensive statistical validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Callable
import warnings

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class TestType(Enum):
    """Types of statistical tests."""
    T_TEST = auto()
    PAIRED_T_TEST = auto()
    WELCH_T_TEST = auto()
    MANN_WHITNEY_U = auto()
    WILCOXON_SIGNED_RANK = auto()
    ANOVA = auto()
    CHI_SQUARE = auto()
    FISHER_EXACT = auto()
    CORRELATION = auto()
    REGRESSION = auto()


class EffectSizeType(Enum):
    """Types of effect size measures."""
    COHENS_D = auto()
    HEDGES_G = auto()
    PEARSON_R = auto()
    R_SQUARED = auto()
    ETA_SQUARED = auto()
    OMEGA_SQUARED = auto()
    CLIFFS_DELTA = auto()
    ODDS_RATIO = auto()


@dataclass
class StatisticalTest:
    """Represents a statistical test configuration."""
    name: str
    test_type: TestType
    null_hypothesis: str
    alternative_hypothesis: str
    alpha: float = 0.05


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    effect_size: Optional[EffectSizeResult] = None
    confidence_interval: Optional[tuple[float, float]] = None
    degrees_of_freedom: Optional[float] = None
    sample_size: Optional[int] = None
    power: Optional[float] = None
    additional_stats: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "significant": self.significant,
            "effect_size": self.effect_size.to_dict() if self.effect_size else None,
            "confidence_interval": self.confidence_interval,
            "degrees_of_freedom": self.degrees_of_freedom,
            "sample_size": self.sample_size,
            "power": self.power,
            **self.additional_stats
        }


@dataclass
class EffectSizeResult:
    """Result of effect size calculation."""
    measure: str
    value: float
    interpretation: str
    confidence_interval: Optional[tuple[float, float]] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "value": self.value,
            "interpretation": self.interpretation,
            "confidence_interval": self.confidence_interval
        }


@dataclass
class DatasetSummary:
    """Summary statistics for a dataset."""
    n: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    
    @classmethod
    def from_array(cls, data: Array) -> DatasetSummary:
        """Create summary from numpy array."""
        from scipy import stats
        
        return cls(
            n=len(data),
            mean=float(np.mean(data)),
            std=float(np.std(data, ddof=1)),
            median=float(np.median(data)),
            min=float(np.min(data)),
            max=float(np.max(data)),
            q25=float(np.percentile(data, 25)),
            q75=float(np.percentile(data, 75)),
            skewness=float(stats.skew(data)),
            kurtosis=float(stats.kurtosis(data))
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "q25": self.q25,
            "q75": self.q75,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis
        }


class ResultAnalyzer:
    """
    Comprehensive statistical analyzer for experimental results.
    """
    
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        
        try:
            from scipy import stats
            self.stats = stats
        except ImportError:
            raise ImportError("SciPy is required for statistical analysis")
    
    def summarize(self, data: Array) -> DatasetSummary:
        """Calculate comprehensive summary statistics."""
        return DatasetSummary.from_array(data)
    
    def t_test(
        self,
        group1: Array,
        group2: Array,
        paired: bool = False,
        equal_var: bool = True,
        alternative: str = "two-sided"
    ) -> TestResult:
        """
        Perform t-test between two groups.
        
        Args:
            group1: First group data
            group2: Second group data
            paired: Whether samples are paired
            equal_var: Assume equal variances (for independent t-test)
            alternative: 'two-sided', 'less', or 'greater'
        """
        if paired:
            if len(group1) != len(group2):
                raise ValueError("Paired samples must have same size")
            
            statistic, p_value = self.stats.ttest_rel(group1, group2, alternative=alternative)
            df = len(group1) - 1
            test_type = TestType.PAIRED_T_TEST
            test_name = "Paired t-test"
        elif equal_var:
            statistic, p_value = self.stats.ttest_ind(
                group1, group2, equal_var=True, alternative=alternative
            )
            df = len(group1) + len(group2) - 2
            test_type = TestType.T_TEST
            test_name = "Independent t-test"
        else:
            statistic, p_value = self.stats.ttest_ind(
                group1, group2, equal_var=False, alternative=alternative
            )
            # Welch-Satterthwaite degrees of freedom
            s1_sq = np.var(group1, ddof=1)
            s2_sq = np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
            test_type = TestType.WELCH_T_TEST
            test_name = "Welch's t-test"
        
        # Calculate Cohen's d
        effect_size = self._cohens_d(group1, group2, paired=paired)
        
        # Calculate confidence interval for effect size
        ci = self._cohens_d_ci(effect_size.value, len(group1), len(group2))
        effect_size.confidence_interval = ci
        
        # Calculate power
        power = self._calculate_power(effect_size.value, len(group1), len(group2))
        
        return TestResult(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            degrees_of_freedom=float(df),
            sample_size=len(group1) + len(group2),
            power=power
        )
    
    def mann_whitney_u(
        self,
        group1: Array,
        group2: Array,
        alternative: str = "two-sided"
    ) -> TestResult:
        """Mann-Whitney U test for non-parametric comparison."""
        statistic, p_value = self.stats.mannwhitneyu(
            group1, group2, alternative=alternative
        )
        
        # Calculate Cliff's delta (effect size for non-parametric)
        effect_size = self._cliffs_delta(group1, group2)
        
        return TestResult(
            test_name="Mann-Whitney U test",
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            sample_size=len(group1) + len(group2)
        )
    
    def wilcoxon_signed_rank(
        self,
        group1: Array,
        group2: Array,
        alternative: str = "two-sided"
    ) -> TestResult:
        """Wilcoxon signed-rank test for paired non-parametric data."""
        statistic, p_value = self.stats.wilcoxon(
            group1, group2, alternative=alternative
        )
        
        # Use rank-biserial correlation as effect size
        effect_size = self._rank_biserial_correlation(group1, group2)
        
        return TestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            sample_size=len(group1)
        )
    
    def anova(
        self,
        *groups: Array
    ) -> TestResult:
        """One-way ANOVA for comparing multiple groups."""
        statistic, p_value = self.stats.f_oneway(*groups)
        
        # Calculate eta-squared (effect size)
        effect_size = self._eta_squared(groups, statistic)
        
        # Total sample size
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        df_between = k - 1
        df_within = n_total - k
        
        # Post-hoc tests (Tukey HSD)
        if p_value < self.alpha and k >= 2:
            from scipy.stats import tukey_hsd
            post_hoc = tukey_hsd(*groups)
            post_hoc_result = {
                "post_hoc_test": "Tukey HSD",
                "p_values": post_hoc.pvalue.tolist() if hasattr(post_hoc, 'pvalue') else None
            }
        else:
            post_hoc_result = {}
        
        return TestResult(
            test_name="One-way ANOVA",
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            degrees_of_freedom=(float(df_between), float(df_within)),
            sample_size=n_total,
            additional_stats=post_hoc_result
        )
    
    def correlation(
        self,
        x: Array,
        y: Array,
        method: str = "pearson"
    ) -> TestResult:
        """
        Calculate correlation between two variables.
        
        Methods: 'pearson', 'spearman', 'kendall'
        """
        if method == "pearson":
            statistic, p_value = self.stats.pearsonr(x, y)
            effect_size = EffectSizeResult(
                measure="Pearson r",
                value=float(statistic),
                interpretation=self._interpret_correlation(abs(statistic))
            )
            test_name = "Pearson correlation"
        elif method == "spearman":
            statistic, p_value = self.stats.spearmanr(x, y)
            effect_size = EffectSizeResult(
                measure="Spearman rho",
                value=float(statistic),
                interpretation=self._interpret_correlation(abs(statistic))
            )
            test_name = "Spearman rank correlation"
        elif method == "kendall":
            statistic, p_value = self.stats.kendalltau(x, y)
            effect_size = EffectSizeResult(
                measure="Kendall tau",
                value=float(statistic),
                interpretation=self._interpret_correlation(abs(statistic))
            )
            test_name = "Kendall's tau correlation"
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Calculate confidence interval
        ci = self._correlation_ci(float(statistic), len(x))
        effect_size.confidence_interval = ci
        
        return TestResult(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            sample_size=len(x)
        )
    
    def regression_analysis(
        self,
        X: Array,
        y: Array,
        add_intercept: bool = True
    ) -> dict[str, Any]:
        """
        Linear regression analysis.
        
        Args:
            X: Predictor variables (n_samples, n_features)
            y: Outcome variable (n_samples,)
            add_intercept: Whether to add intercept term
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if add_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        
        # Fit regression
        n, k = X.shape
        
        # Normal equation: beta = (X'X)^(-1) X'y
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"error": "Matrix is singular"}
        
        # Predictions and residuals
        y_pred = X @ beta
        residuals = y - y_pred
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Adjusted R-squared
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
        
        # Standard errors
        mse = ss_res / (n - k)
        var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
        se_beta = np.sqrt(var_beta)
        
        # t-statistics and p-values
        t_stats = beta / se_beta
        p_values = 2 * (1 - self.stats.t.cdf(np.abs(t_stats), n - k))
        
        # F-statistic
        ms_reg = (ss_tot - ss_res) / (k - 1) if k > 1 else 0
        f_stat = ms_reg / mse if mse > 0 else 0
        f_pvalue = 1 - self.stats.f.cdf(f_stat, k - 1, n - k) if k > 1 else 1
        
        return {
            "coefficients": beta.tolist(),
            "standard_errors": se_beta.tolist(),
            "t_statistics": t_stats.tolist(),
            "p_values": p_values.tolist(),
            "r_squared": float(r_squared),
            "adjusted_r_squared": float(adj_r_squared),
            "f_statistic": float(f_stat),
            "f_pvalue": float(f_pvalue),
            "n": n,
            "n_features": k - (1 if add_intercept else 0),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse))
        }
    
    def normality_test(self, data: Array) -> TestResult:
        """Test if data follows normal distribution (Shapiro-Wilk or D'Agostino)."""
        if len(data) <= 5000:
            # Shapiro-Wilk test
            statistic, p_value = self.stats.shapiro(data)
            test_name = "Shapiro-Wilk normality test"
        else:
            # D'Agostino's normality test
            statistic, p_value = self.stats.normaltest(data)
            test_name = "D'Agostino normality test"
        
        return TestResult(
            test_name=test_name,
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            significant=p_value < self.alpha,  # Significant = NOT normal
            sample_size=len(data),
            additional_stats={"is_normal": p_value >= self.alpha}
        )
    
    def homoscedasticity_test(
        self,
        group1: Array,
        group2: Array
    ) -> TestResult:
        """Test for equal variances (Levene's test)."""
        statistic, p_value = self.stats.levene(group1, group2)
        
        return TestResult(
            test_name="Levene's test for equal variances",
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            significant=p_value < self.alpha,  # Significant = unequal variances
            additional_stats={"equal_variances": p_value >= self.alpha}
        )
    
    def comprehensive_analysis(
        self,
        *groups: Array,
        group_names: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """
        Perform comprehensive statistical analysis on multiple groups.
        """
        if group_names is None:
            group_names = [f"Group {i+1}" for i in range(len(groups))]
        
        results: dict[str, Any] = {
            "group_names": group_names,
            "n_groups": len(groups),
            "alpha": self.alpha
        }
        
        # Descriptive statistics
        results["descriptive"] = {
            name: self.summarize(group).to_dict()
            for name, group in zip(group_names, groups)
        }
        
        # Normality tests
        results["normality"] = {
            name: self.normality_test(group).to_dict()
            for name, group in zip(group_names, groups)
        }
        
        # Overall test
        if len(groups) == 2:
            # Check normality and homoscedasticity
            g1_normal = results["normality"][group_names[0]]["additional_stats"]["is_normal"]
            g2_normal = results["normality"][group_names[1]]["additional_stats"]["is_normal"]
            
            if g1_normal and g2_normal:
                # Check equal variances
                var_test = self.homoscedasticity_test(groups[0], groups[1])
                results["variance_test"] = var_test.to_dict()
                
                # Use appropriate t-test
                results["comparison"] = self.t_test(
                    groups[0], groups[1],
                    equal_var=var_test.additional_stats["equal_variances"]
                ).to_dict()
            else:
                # Non-parametric
                results["comparison"] = self.mann_whitney_u(
                    groups[0], groups[1]
                ).to_dict()
        elif len(groups) > 2:
            # Check normality for all groups
            all_normal = all(
                results["normality"][name]["additional_stats"]["is_normal"]
                for name in group_names
            )
            
            if all_normal:
                results["comparison"] = self.anova(*groups).to_dict()
            else:
                # Kruskal-Wallis test
                statistic, p_value = self.stats.kruskal(*groups)
                results["comparison"] = {
                    "test_name": "Kruskal-Wallis H test",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": p_value < self.alpha
                }
        
        return results
    
    # Effect size calculations
    
    def _cohens_d(
        self,
        group1: Array,
        group2: Array,
        paired: bool = False
    ) -> EffectSizeResult:
        """Calculate Cohen's d effect size."""
        if paired:
            d = np.mean(group1 - group2) / np.std(group1 - group2, ddof=1)
        else:
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(
                ((n1 - 1) * np.var(group1, ddof=1) +
                 (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2)
            )
            d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return EffectSizeResult(
            measure="Cohen's d",
            value=float(d),
            interpretation=self._interpret_cohens_d(abs(d))
        )
    
    def _cohens_d_ci(
        self,
        d: float,
        n1: int,
        n2: int,
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for Cohen's d."""
        # Standard error of d
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        
        # Critical value
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        
        return (float(d - z * se_d), float(d + z * se_d))
    
    def _cliffs_delta(self, group1: Array, group2: Array) -> EffectSizeResult:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(group1), len(group2)
        
        # Count pairs
        dom = sum(1 for x in group1 for y in group2 if x > y)
        ties = sum(1 for x in group1 for y in group2 if x == y)
        
        delta = (dom - (n1 * n2 - dom - ties)) / (n1 * n2)
        
        return EffectSizeResult(
            measure="Cliff's delta",
            value=float(delta),
            interpretation=self._interpret_cliffs_delta(abs(delta))
        )
    
    def _rank_biserial_correlation(
        self,
        group1: Array,
        group2: Array
    ) -> EffectSizeResult:
        """Calculate rank-biserial correlation as effect size for Wilcoxon test."""
        n = len(group1)
        # Calculate proportion of positive differences
        positive = np.sum(group1 > group2)
        negative = np.sum(group1 < group2)
        
        r = (positive - negative) / (positive + negative) if (positive + negative) > 0 else 0
        
        return EffectSizeResult(
            measure="Rank-biserial r",
            value=float(r),
            interpretation=self._interpret_correlation(abs(r))
        )
    
    def _eta_squared(
        self,
        groups: tuple[Array, ...],
        f_stat: float
    ) -> EffectSizeResult:
        """Calculate eta-squared from ANOVA."""
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        
        eta_sq = (f_stat * (k - 1)) / (f_stat * (k - 1) + (n_total - k))
        
        return EffectSizeResult(
            measure="Eta-squared",
            value=float(eta_sq),
            interpretation=self._interpret_eta_squared(eta_sq)
        )
    
    def _correlation_ci(
        self,
        r: float,
        n: int,
        confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for correlation."""
        # Fisher z-transformation
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        
        from scipy.stats import norm
        z_crit = norm.ppf((1 + confidence) / 2)
        
        ci_low = z - z_crit * se
        ci_high = z + z_crit * se
        
        # Transform back
        ci_low = (np.exp(2 * ci_low) - 1) / (np.exp(2 * ci_low) + 1)
        ci_high = (np.exp(2 * ci_high) - 1) / (np.exp(2 * ci_high) + 1)
        
        return (float(ci_low), float(ci_high))
    
    def _calculate_power(
        self,
        effect_size: float,
        n1: int,
        n2: int,
        alpha: float = 0.05
    ) -> float:
        """Calculate statistical power for t-test."""
        try:
            from statsmodels.stats.power import TTestIndPower
            analysis = TTestIndPower()
            return float(analysis.power(
                effect_size=abs(effect_size),
                nobs1=n1,
                alpha=alpha,
                ratio=n2/n1
            ))
        except ImportError:
            # Approximate power calculation
            n = (n1 + n2) / 2
            z_alpha = 1.96  # For alpha=0.05 two-tailed
            z_beta = abs(effect_size) * np.sqrt(n / 2) - z_alpha
            from scipy.stats import norm
            return float(norm.cdf(z_beta))
    
    # Interpretation helpers
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's delta."""
        if delta < 0.147:
            return "negligible"
        elif delta < 0.33:
            return "small"
        elif delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient."""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        elif r < 0.7:
            return "large"
        else:
            return "very large"
    
    def _interpret_eta_squared(self, eta: float) -> str:
        """Interpret eta-squared."""
        if eta < 0.01:
            return "negligible"
        elif eta < 0.06:
            return "small"
        elif eta < 0.14:
            return "medium"
        else:
            return "large"


def demo():
    """Demo statistical analysis."""
    np.random.seed(42)
    
    # Generate sample data
    group_a = np.random.normal(100, 15, 50)  # Control
    group_b = np.random.normal(110, 15, 50)  # Treatment
    group_c = np.random.normal(105, 20, 50)  # Alternative treatment
    
    analyzer = ResultAnalyzer(alpha=0.05)
    
    print("=== Descriptive Statistics ===")
    for name, data in [("Control", group_a), ("Treatment", group_b), ("Alt Treatment", group_c)]:
        summary = analyzer.summarize(data)
        print(f"\n{name}:")
        print(f"  N: {summary.n}")
        print(f"  Mean: {summary.mean:.2f} ± {summary.std:.2f}")
        print(f"  Median: {summary.median:.2f}")
        print(f"  Range: [{summary.min:.2f}, {summary.max:.2f}]")
    
    print("\n=== Normality Tests ===")
    for name, data in [("Control", group_a), ("Treatment", group_b)]:
        result = analyzer.normality_test(data)
        print(f"{name}: statistic={result.statistic:.4f}, p={result.p_value:.4f}, "
              f"normal={result.additional_stats['is_normal']}")
    
    print("\n=== T-Test (Control vs Treatment) ===")
    t_result = analyzer.t_test(group_a, group_b)
    print(f"t-statistic: {t_result.statistic:.4f}")
    print(f"p-value: {t_result.p_value:.4f}")
    print(f"Significant: {t_result.significant}")
    print(f"Effect size ({t_result.effect_size.measure}): {t_result.effect_size.value:.4f} "
          f"({t_result.effect_size.interpretation})")
    print(f"Power: {t_result.power:.2%}")
    
    print("\n=== ANOVA (All Groups) ===")
    anova_result = analyzer.anova(group_a, group_b, group_c)
    print(f"F-statistic: {anova_result.statistic:.4f}")
    print(f"p-value: {anova_result.p_value:.4f}")
    print(f"Effect size ({anova_result.effect_size.measure}): {anova_result.effect_size.value:.4f} "
          f"({anova_result.effect_size.interpretation})")
    
    print("\n=== Correlation Analysis ===")
    x = np.random.normal(0, 1, 100)
    y = 0.7 * x + np.random.normal(0, 0.5, 100)
    corr_result = analyzer.correlation(x, y, method="pearson")
    print(f"Pearson r: {corr_result.statistic:.4f}")
    print(f"p-value: {corr_result.p_value:.4f}")
    print(f"Interpretation: {corr_result.effect_size.interpretation}")


if __name__ == "__main__":
    demo()
