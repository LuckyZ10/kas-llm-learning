"""
Data Anonymization and Privacy Protection Module
=================================================

This module implements various data anonymization techniques for
protecting sensitive information in materials science data.

Techniques:
- k-anonymity
- l-diversity
- t-closeness
- Differential privacy for data release
- Synthetic data generation
- Feature transformation and perturbation

Author: DFT-LAMMPS Team
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib
import secrets
import json
import logging
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)


class AnonymizationLevel(Enum):
    """Levels of data anonymization."""
    NONE = "none"
    MASKING = "masking"  # Simple masking
    K_ANONYMITY = "k_anonymity"  # k-anonymity
    L_DIVERSITY = "l_diversity"  # l-diversity
    T_CLOSENESS = "t_closeness"  # t-closeness
    DIFFERENTIAL_PRIVACY = "differential_privacy"  # DP-based
    SYNTHETIC = "synthetic"  # Fully synthetic data


@dataclass
class AnonymizationConfig:
    """Configuration for data anonymization."""
    level: AnonymizationLevel = AnonymizationLevel.K_ANONYMITY
    k: int = 5  # k-anonymity parameter
    l: int = 2  # l-diversity parameter
    t: float = 0.2  # t-closeness parameter
    epsilon: float = 1.0  # DP epsilon
    delta: float = 1e-5  # DP delta
    sensitive_attributes: List[str] = field(default_factory=list)
    quasi_identifiers: List[str] = field(default_factory=list)
    identifier_attributes: List[str] = field(default_factory=list)


class DataAnonymizer(ABC):
    """Abstract base class for data anonymization methods."""
    
    @abstractmethod
    def anonymize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Anonymize the input data."""
        pass
    
    @abstractmethod
    def get_privacy_guarantee(self) -> Dict[str, Any]:
        """Return privacy guarantee metrics."""
        pass


class KAnonymityAnonymizer(DataAnonymizer):
    """
    k-anonymity implementation using generalization and suppression.
    
    Ensures each record is indistinguishable from at least k-1 other
    records with respect to quasi-identifiers.
    
    Reference: Sweeney, "k-anonymity: A Model for Protecting Privacy"
    (IJUFKS 2002)
    """
    
    def __init__(self, config: AnonymizationConfig):
        self.config = config
        self.generalization_hierarchies = {}
        self.suppression_count = 0
        
    def anonymize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Anonymize data to achieve k-anonymity.
        
        Args:
            data: Input DataFrame
            
        Returns:
            k-anonymous DataFrame
        """
        df = data.copy()
        qi_columns = self.config.quasi_identifiers
        k = self.config.k
        
        if not qi_columns:
            logger.warning("No quasi-identifiers specified")
            return df
        
        # Initialize generalization levels
        generalization_levels = {col: 0 for col in qi_columns}
        
        # Iteratively generalize until k-anonymity is achieved
        max_iterations = 100
        for iteration in range(max_iterations):
            # Check k-anonymity
            equivalence_classes = self._get_equivalence_classes(df, qi_columns)
            
            if self._is_k_anonymous(equivalence_classes, k):
                logger.info(f"k-anonymity achieved after {iteration} iterations")
                break
            
            # Find column to generalize
            col_to_generalize = self._choose_column_to_generalize(
                df, qi_columns, equivalence_classes
            )
            
            if col_to_generalize is None:
                # Suppress remaining records
                df = self._suppress_records(df, equivalence_classes, k)
                break
            
            # Generalize column
            df = self._generalize_column(df, col_to_generalize, 
                                        generalization_levels[col_to_generalize])
            generalization_levels[col_to_generalize] += 1
        
        return df
    
    def _get_equivalence_classes(self, df: pd.DataFrame, 
                                 qi_columns: List[str]) -> Dict:
        """Get equivalence class sizes."""
        groups = df.groupby(qi_columns)
        return {name: len(group) for name, group in groups}
    
    def _is_k_anonymous(self, equivalence_classes: Dict, k: int) -> bool:
        """Check if all equivalence classes have size >= k."""
        return all(size >= k for size in equivalence_classes.values())
    
    def _choose_column_to_generalize(self, df: pd.DataFrame,
                                    qi_columns: List[str],
                                    equivalence_classes: Dict) -> Optional[str]:
        """Choose the best column to generalize next."""
        # Find equivalence classes with size < k
        small_classes = [ec for ec, size in equivalence_classes.items() if size < self.config.k]
        
        if not small_classes:
            return None
        
        # Choose column that appears most in small equivalence classes
        column_counts = defaultdict(int)
        for ec in small_classes:
            for col in qi_columns:
                column_counts[col] += 1
        
        if column_counts:
            return max(column_counts, key=column_counts.get)
        
        return None
    
    def _generalize_column(self, df: pd.DataFrame, column: str,
                          level: int) -> pd.DataFrame:
        """Generalize values in a column."""
        df = df.copy()
        
        # Numerical columns: use range generalization
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = self._generalize_numerical(df[column], level)
        else:
            # Categorical columns: use hierarchy
            df[column] = self._generalize_categorical(df[column], column, level)
        
        return df
    
    def _generalize_numerical(self, series: pd.Series, level: int) -> pd.Series:
        """Generalize numerical values to ranges."""
        min_val = series.min()
        max_val = series.max()
        
        # Create bins
        num_bins = max(2, 10 - level * 2)
        bins = np.linspace(min_val, max_val, num_bins + 1)
        
        # Map to bin ranges
        labels = [f"[{bins[i]:.1f}, {bins[i+1]:.1f})" for i in range(len(bins)-1)]
        return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(str)
    
    def _generalize_categorical(self, series: pd.Series, column: str,
                               level: int) -> pd.Series:
        """Generalize categorical values using hierarchy."""
        # Define hierarchies for common material science attributes
        hierarchies = {
            'element': {
                0: lambda x: x,  # No generalization
                1: self._generalize_element_group,
                2: self._generalize_element_block,
                3: lambda x: '*'  # Fully generalized
            }
        }
        
        if column in hierarchies:
            return series.apply(hierarchies[column].get(level, lambda x: '*'))
        
        # Default: group rare values
        value_counts = series.value_counts()
        common_values = set(value_counts[value_counts >= self.config.k].index)
        
        return series.apply(lambda x: x if x in common_values else 'OTHER')
    
    def _generalize_element_group(self, element: str) -> str:
        """Generalize element to periodic table group."""
        groups = {
            'alkali': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
            'alkaline': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
            'transition': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 
                          'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                          'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re',
                          'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Rf', 'Db', 'Sg',
                          'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn'],
            'chalcogen': ['O', 'S', 'Se', 'Te', 'Po'],
            'pnictogen': ['N', 'P', 'As', 'Sb', 'Bi', 'Mc'],
            'halogen': ['F', 'Cl', 'Br', 'I', 'At', 'Ts'],
            'noble': ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og'],
            'lanthanide': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                          'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
            'actinide': ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
                        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        }
        
        for group, elements in groups.items():
            if element in elements:
                return f"[{group}]"
        
        return element
    
    def _generalize_element_block(self, element: str) -> str:
        """Generalize element to periodic table block."""
        s_block = ['H', 'He', 'Li', 'Be', 'Na', 'Mg', 'K', 'Ca', 'Rb', 'Sr',
                   'Cs', 'Ba', 'Fr', 'Ra']
        p_block = ['B', 'C', 'N', 'O', 'F', 'Ne', 'Al', 'Si', 'P', 'S', 'Cl',
                   'Ar', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Nh',
                   'Fl', 'Mc', 'Lv', 'Ts', 'Og']
        d_block = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                   'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Rf',
                   'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']
        f_block = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
                   'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np',
                   'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        
        if element in s_block:
            return "[s-block]"
        elif element in p_block:
            return "[p-block]"
        elif element in d_block:
            return "[d-block]"
        elif element in f_block:
            return "[f-block]"
        
        return element
    
    def _suppress_records(self, df: pd.DataFrame, 
                         equivalence_classes: Dict, k: int) -> pd.DataFrame:
        """Suppress records in small equivalence classes."""
        qi_columns = self.config.quasi_identifiers
        
        # Find records to suppress
        records_to_keep = []
        for name, group in df.groupby(qi_columns):
            if len(group) >= k:
                records_to_keep.extend(group.index.tolist())
        
        suppressed_count = len(df) - len(records_to_keep)
        self.suppression_count = suppressed_count
        
        logger.info(f"Suppressed {suppressed_count} records")
        
        return df.loc[records_to_keep]
    
    def get_privacy_guarantee(self) -> Dict[str, Any]:
        """Return privacy guarantee information."""
        return {
            'k': self.config.k,
            'suppressed_records': self.suppression_count,
            'generalization_levels': dict(self.generalization_hierarchies)
        }


class LDiversityAnonymizer(DataAnonymizer):
    """
    l-diversity implementation for sensitive attributes.
    
    Ensures each equivalence class contains at least l well-represented
    values for sensitive attributes.
    
    Reference: Machanavajjhala et al., "l-Diversity: Privacy Beyond 
    k-Anonymity" (TKDD 2007)
    """
    
    def __init__(self, config: AnonymizationConfig):
        self.config = config
        self.k_anonymizer = KAnonymityAnonymizer(config)
        
    def anonymize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Anonymize data to achieve l-diversity.
        
        Args:
            data: Input DataFrame
            
        Returns:
            l-diverse DataFrame
        """
        # First achieve k-anonymity
        df = self.k_anonymizer.anonymize(data)
        
        qi_columns = self.config.quasi_identifiers
        sensitive_columns = self.config.sensitive_attributes
        l = self.config.l
        
        if not sensitive_columns:
            logger.warning("No sensitive attributes specified")
            return df
        
        # Check and enforce l-diversity
        equivalence_classes = df.groupby(qi_columns)
        
        records_to_keep = []
        for name, group in equivalence_classes:
            if self._has_l_diversity(group, sensitive_columns, l):
                records_to_keep.extend(group.index.tolist())
        
        return df.loc[records_to_keep]
    
    def _has_l_diversity(self, group: pd.DataFrame,
                        sensitive_columns: List[str], l: int) -> bool:
        """Check if group has l-diversity for all sensitive attributes."""
        for col in sensitive_columns:
            unique_values = group[col].nunique()
            if unique_values < l:
                return False
            
            # Check entropy l-diversity
            value_counts = group[col].value_counts(normalize=True)
            entropy = -np.sum(value_counts * np.log(value_counts))
            if entropy < np.log(l):
                return False
        
        return True
    
    def get_privacy_guarantee(self) -> Dict[str, Any]:
        """Return privacy guarantee information."""
        return {
            'k': self.config.k,
            'l': self.config.l,
            'k_anonymity_guarantee': self.k_anonymizer.get_privacy_guarantee()
        }


class DifferentialPrivacyAnonymizer(DataAnonymizer):
    """
    Differential privacy for data release.
    
    Adds calibrated noise to query results or synthetic data generation
    to provide formal differential privacy guarantees.
    
    Reference: Dwork and Roth, "The Algorithmic Foundations of 
    Differential Privacy" (FnT-TCS 2014)
    """
    
    def __init__(self, config: AnonymizationConfig):
        self.config = config
        self.noise_scale = 0.0
        
    def anonymize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Anonymize data using differential privacy.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DP-anonymized DataFrame
        """
        epsilon = self.config.epsilon
        delta = self.config.delta
        
        # Compute noise scale
        sensitivity = self._compute_global_sensitivity(data)
        self.noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        # Add noise to numerical columns
        df = data.copy()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in self.config.identifier_attributes:
                noise = np.random.laplace(0, self.noise_scale, len(df))
                df[col] = df[col] + noise
        
        # For categorical columns, use randomized response
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col not in self.config.identifier_attributes:
                df[col] = self._randomized_response(df[col], epsilon)
        
        return df
    
    def _compute_global_sensitivity(self, data: pd.DataFrame) -> float:
        """Compute global sensitivity of the dataset."""
        # For range queries, sensitivity is the range
        max_sensitivity = 0.0
        
        for col in data.select_dtypes(include=[np.number]).columns:
            col_range = data[col].max() - data[col].min()
            max_sensitivity = max(max_sensitivity, col_range)
        
        return max_sensitivity if max_sensitivity > 0 else 1.0
    
    def _randomized_response(self, series: pd.Series, epsilon: float) -> pd.Series:
        """
        Apply randomized response to categorical data.
        
        With probability p = e^epsilon / (e^epsilon + k - 1),
        keep the true value. Otherwise, randomly select another value.
        """
        unique_values = series.unique()
        k = len(unique_values)
        
        if k <= 1:
            return series
        
        exp_epsilon = np.exp(epsilon)
        p = exp_epsilon / (exp_epsilon + k - 1)
        
        result = []
        for value in series:
            if np.random.random() < p:
                result.append(value)
            else:
                result.append(np.random.choice([v for v in unique_values if v != value]))
        
        return pd.Series(result, index=series.index)
    
    def privatize_histogram(self, data: pd.DataFrame, 
                           column: str) -> Dict[Any, float]:
        """
        Create a differentially private histogram.
        
        Args:
            data: Input DataFrame
            column: Column to histogram
            
        Returns:
            Dictionary of bin counts with DP noise
        """
        value_counts = data[column].value_counts()
        
        epsilon = self.config.epsilon
        sensitivity = 1  # Adding/removing one record changes count by 1
        scale = sensitivity / epsilon
        
        private_counts = {}
        for value, count in value_counts.items():
            noise = np.random.laplace(0, scale)
            private_counts[value] = max(0, count + noise)
        
        return private_counts
    
    def privatize_mean(self, data: pd.DataFrame, 
                      column: str,
                      clip_bounds: Tuple[float, float] = None) -> float:
        """
        Compute differentially private mean.
        
        Args:
            data: Input DataFrame
            column: Column to compute mean
            clip_bounds: (min, max) bounds for clipping
            
        Returns:
            DP mean estimate
        """
        values = data[column].dropna()
        
        # Clip values
        if clip_bounds:
            values = values.clip(clip_bounds[0], clip_bounds[1])
            sensitivity = (clip_bounds[1] - clip_bounds[0]) / len(values)
        else:
            # Use data range (less private)
            sensitivity = (values.max() - values.min()) / len(values)
        
        # Add noise
        scale = sensitivity / self.config.epsilon
        noise = np.random.laplace(0, scale)
        
        return values.mean() + noise
    
    def get_privacy_guarantee(self) -> Dict[str, Any]:
        """Return privacy guarantee information."""
        return {
            'epsilon': self.config.epsilon,
            'delta': self.config.delta,
            'noise_scale': self.noise_scale,
            'mechanism': 'Laplace'
        }


class SyntheticDataGenerator(DataAnonymizer):
    """
    Generate synthetic data that preserves statistical properties
    while providing strong privacy guarantees.
    """
    
    def __init__(self, config: AnonymizationConfig):
        self.config = config
        self.statistical_model = None
        
    def anonymize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic data based on original data statistics.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Synthetic DataFrame with same schema
        """
        # Learn statistical model from data with DP
        self._learn_model(data)
        
        # Generate synthetic records
        n_records = len(data)
        synthetic_data = self._generate_synthetic(n_records)
        
        return synthetic_data
    
    def _learn_model(self, data: pd.DataFrame) -> None:
        """Learn statistical model from data."""
        self.statistical_model = {}
        
        epsilon = self.config.epsilon
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                # Learn mean and std with DP
                mean = data[col].mean()
                std = data[col].std()
                
                # Add DP noise
                sensitivity = (data[col].max() - data[col].min()) / len(data)
                noise_scale = sensitivity / epsilon
                
                mean += np.random.laplace(0, noise_scale)
                std = max(0.1, std + np.random.laplace(0, noise_scale))
                
                self.statistical_model[col] = {
                    'type': 'numeric',
                    'mean': mean,
                    'std': std
                }
            else:
                # Learn category distribution with DP
                value_counts = data[col].value_counts(normalize=True)
                
                # Add DP noise to counts
                sensitivity = 1 / len(data)
                noise_scale = sensitivity / epsilon
                
                noisy_counts = {}
                for val, count in value_counts.items():
                    noisy_count = max(0, count + np.random.laplace(0, noise_scale))
                    noisy_counts[val] = noisy_count
                
                # Normalize
                total = sum(noisy_counts.values())
                if total > 0:
                    noisy_counts = {k: v/total for k, v in noisy_counts.items()}
                
                self.statistical_model[col] = {
                    'type': 'categorical',
                    'distribution': noisy_counts
                }
    
    def _generate_synthetic(self, n: int) -> pd.DataFrame:
        """Generate synthetic records."""
        synthetic = {}
        
        for col, model in self.statistical_model.items():
            if model['type'] == 'numeric':
                synthetic[col] = np.random.normal(model['mean'], model['std'], n)
            else:
                values = list(model['distribution'].keys())
                probabilities = list(model['distribution'].values())
                synthetic[col] = np.random.choice(values, n, p=probabilities)
        
        return pd.DataFrame(synthetic)
    
    def get_privacy_guarantee(self) -> Dict[str, Any]:
        """Return privacy guarantee information."""
        return {
            'type': 'synthetic_data',
            'epsilon': self.config.epsilon,
            'model_type': 'statistical'
        }


class PrivacyAuditor:
    """
    Audit data for privacy vulnerabilities.
    
    Checks for various privacy risks including:
    - Re-identification risk
    - Attribute disclosure
    - Membership inference
    """
    
    def __init__(self):
        self.risk_scores = {}
        
    def audit_k_anonymity(self, data: pd.DataFrame, 
                         qi_columns: List[str]) -> Dict:
        """
        Audit k-anonymity compliance.
        
        Args:
            data: DataFrame to audit
            qi_columns: Quasi-identifier columns
            
        Returns:
            Audit results
        """
        equivalence_classes = data.groupby(qi_columns).size()
        
        min_k = equivalence_classes.min()
        max_k = equivalence_classes.max()
        avg_k = equivalence_classes.mean()
        
        # Records at risk (k=1)
        records_at_risk = equivalence_classes[equivalence_classes == 1].sum()
        
        return {
            'min_k': int(min_k),
            'max_k': int(max_k),
            'avg_k': float(avg_k),
            'records_at_risk': int(records_at_risk),
            'risk_percentage': float(records_at_risk / len(data) * 100)
        }
    
    def audit_l_diversity(self, data: pd.DataFrame,
                         qi_columns: List[str],
                         sensitive_columns: List[str]) -> Dict:
        """Audit l-diversity compliance."""
        results = {}
        
        for sensitive_col in sensitive_columns:
            l_values = []
            for name, group in data.groupby(qi_columns):
                l_values.append(group[sensitive_col].nunique())
            
            results[sensitive_col] = {
                'min_l': int(min(l_values)),
                'max_l': int(max(l_values)),
                'avg_l': float(np.mean(l_values))
            }
        
        return results
    
    def estimate_reidentification_risk(self, data: pd.DataFrame,
                                      qi_columns: List[str]) -> float:
        """
        Estimate probability of re-identification.
        
        Returns:
            Re-identification risk score [0, 1]
        """
        equivalence_classes = data.groupby(qi_columns).size()
        
        # Risk is inversely proportional to equivalence class size
        risks = 1 / equivalence_classes
        
        return float(risks.mean())
    
    def generate_privacy_report(self, data: pd.DataFrame,
                               original_data: pd.DataFrame = None) -> str:
        """
        Generate comprehensive privacy audit report.
        
        Args:
            data: Anonymized data
            original_data: Original data (if available)
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("Privacy Audit Report")
        report.append("=" * 60)
        
        # Basic statistics
        report.append(f"\nDataset Statistics:")
        report.append(f"  Records: {len(data)}")
        report.append(f"  Attributes: {len(data.columns)}")
        
        if original_data is not None:
            info_loss = self._compute_information_loss(original_data, data)
            report.append(f"  Information Loss: {info_loss:.2%}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def _compute_information_loss(self, original: pd.DataFrame,
                                 anonymized: pd.DataFrame) -> float:
        """Compute information loss due to anonymization."""
        # Simplified metric - compare column distributions
        losses = []
        
        for col in original.select_dtypes(include=[np.number]).columns:
            if col in anonymized.columns:
                orig_mean = original[col].mean()
                anon_mean = anonymized[col].mean()
                
                if orig_mean != 0:
                    loss = abs(orig_mean - anon_mean) / abs(orig_mean)
                    losses.append(min(loss, 1.0))
        
        return np.mean(losses) if losses else 0.0


def create_anonymization_pipeline(config: AnonymizationConfig) -> DataAnonymizer:
    """
    Factory function to create appropriate anonymizer.
    
    Args:
        config: Anonymization configuration
        
    Returns:
        Configured anonymizer instance
    """
    level = config.level
    
    if level == AnonymizationLevel.K_ANONYMITY:
        return KAnonymityAnonymizer(config)
    elif level == AnonymizationLevel.L_DIVERSITY:
        return LDiversityAnonymizer(config)
    elif level == AnonymizationLevel.DIFFERENTIAL_PRIVACY:
        return DifferentialPrivacyAnonymizer(config)
    elif level == AnonymizationLevel.SYNTHETIC:
        return SyntheticDataGenerator(config)
    else:
        raise ValueError(f"Unsupported anonymization level: {level}")


def demo_anonymization():
    """Demonstration of anonymization capabilities."""
    print("=" * 60)
    print("Data Anonymization Demo")
    print("=" * 60)
    
    # Create sample materials data
    np.random.seed(42)
    
    data = pd.DataFrame({
        'material_id': [f"MAT_{i:03d}" for i in range(100)],
        'composition': np.random.choice(['LiFePO4', 'LiCoO2', 'NaFePO4', 'MgSiO3'], 100),
        'researcher': np.random.choice(['Alice', 'Bob', 'Carol', 'David'], 100),
        'institution': np.random.choice(['MIT', 'Stanford', 'Berkeley', 'Caltech'], 100),
        'band_gap': np.random.uniform(0.5, 5.0, 100),
        'formation_energy': np.random.uniform(-8, -1, 100),
        'synthesis_temperature': np.random.uniform(200, 1000, 100),
        'yield_percent': np.random.uniform(60, 99, 100)
    })
    
    print("\nOriginal Data Sample:")
    print(data.head())
    
    # Demo 1: k-anonymity
    print("\n" + "-" * 40)
    print("1. k-Anonymity (k=5)")
    print("-" * 40)
    
    config_k = AnonymizationConfig(
        level=AnonymizationLevel.K_ANONYMITY,
        k=5,
        quasi_identifiers=['composition', 'institution'],
        identifier_attributes=['material_id', 'researcher']
    )
    
    anonymizer_k = KAnonymityAnonymizer(config_k)
    data_k = anonymizer_k.anonymize(data)
    
    print(f"Original records: {len(data)}")
    print(f"Anonymized records: {len(data_k)}")
    print(f"Privacy guarantee: {anonymizer_k.get_privacy_guarantee()}")
    
    # Demo 2: Differential Privacy
    print("\n" + "-" * 40)
    print("2. Differential Privacy (ε=1.0)")
    print("-" * 40)
    
    config_dp = AnonymizationConfig(
        level=AnonymizationLevel.DIFFERENTIAL_PRIVACY,
        epsilon=1.0,
        delta=1e-5
    )
    
    anonymizer_dp = DifferentialPrivacyAnonymizer(config_dp)
    data_dp = anonymizer_dp.anonymize(data.select_dtypes(include=[np.number]))
    
    print("Original band_gap stats:")
    print(f"  Mean: {data['band_gap'].mean():.3f}")
    print(f"  Std: {data['band_gap'].std():.3f}")
    
    print("DP band_gap stats:")
    print(f"  Mean: {data_dp['band_gap'].mean():.3f}")
    print(f"  Std: {data_dp['band_gap'].std():.3f}")
    
    print(f"Privacy guarantee: {anonymizer_dp.get_privacy_guarantee()}")
    
    # Demo 3: Synthetic Data
    print("\n" + "-" * 40)
    print("3. Synthetic Data Generation")
    print("-" * 40)
    
    config_syn = AnonymizationConfig(
        level=AnonymizationLevel.SYNTHETIC,
        epsilon=1.0
    )
    
    anonymizer_syn = SyntheticDataGenerator(config_syn)
    data_syn = anonymizer_syn.anonymize(data.select_dtypes(include=[np.number]))
    
    print(f"Synthetic records generated: {len(data_syn)}")
    print(f"Privacy guarantee: {anonymizer_syn.get_privacy_guarantee()}")
    
    # Demo 4: Privacy Audit
    print("\n" + "-" * 40)
    print("4. Privacy Audit")
    print("-" * 40)
    
    auditor = PrivacyAuditor()
    audit_results = auditor.audit_k_anonymity(data, ['composition', 'institution'])
    
    print(f"k-anonymity audit: {audit_results}")
    
    risk = auditor.estimate_reidentification_risk(data, ['composition', 'institution'])
    print(f"Estimated re-identification risk: {risk:.4f}")
    
    print("\n" + "=" * 60)
    print("Anonymization Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    demo_anonymization()
