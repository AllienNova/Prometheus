"""
Statistical analysis module for scientific research.

This module provides advanced statistical analysis capabilities for scientific data,
including hypothesis testing, correlation analysis, and statistical modeling.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class StatTestResult:
    """Data class for storing statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    significant: bool = None
    
    def __post_init__(self):
        if self.significant is None:
            self.significant = self.p_value < self.alpha

class StatisticalAnalyzer:
    """
    Class for performing statistical analysis on scientific data.
    
    This class provides methods for hypothesis testing, correlation analysis,
    distribution fitting, and other statistical analyses commonly used in
    scientific research.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the StatisticalAnalyzer.
        
        Args:
            alpha: Significance level for hypothesis tests (default: 0.05)
        """
        self.alpha = alpha
        logger.info(f"Initialized StatisticalAnalyzer with alpha={alpha}")
    
    def test_normality(self, data: Union[pd.Series, np.ndarray], 
                       test: str = 'shapiro') -> StatTestResult:
        """
        Test whether a data sample comes from a normal distribution.
        
        Args:
            data: Data sample to test
            test: Test to use ('shapiro', 'ks', 'anderson')
            
        Returns:
            StatTestResult object with test results
        """
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            data = data.dropna().values
        
        if len(data) < 3:
            raise ValueError("At least 3 data points required for normality test")
        
        if test == 'shapiro':
            # Shapiro-Wilk test (better for smaller samples, n < 2000)
            statistic, p_value = stats.shapiro(data)
            logger.info(f"Performed Shapiro-Wilk normality test: statistic={statistic:.4f}, p-value={p_value:.4f}")
            return StatTestResult(test_name="Shapiro-Wilk normality test", 
                                 statistic=statistic, 
                                 p_value=p_value,
                                 alpha=self.alpha)
        
        elif test == 'ks':
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
            logger.info(f"Performed Kolmogorov-Smirnov normality test: statistic={statistic:.4f}, p-value={p_value:.4f}")
            return StatTestResult(test_name="Kolmogorov-Smirnov normality test", 
                                 statistic=statistic, 
                                 p_value=p_value,
                                 alpha=self.alpha)
        
        elif test == 'anderson':
            # Anderson-Darling test
            result = stats.anderson(data, dist='norm')
            statistic = result.statistic
            
            # Find the appropriate critical value based on our alpha
            critical_values = result.critical_values
            significance_levels = [15, 10, 5, 2.5, 1]  # Percentages
            
            # Find the closest significance level to our alpha
            closest_idx = np.abs(np.array(significance_levels) - (self.alpha * 100)).argmin()
            critical_value = critical_values[closest_idx]
            
            # Determine significance
            significant = statistic > critical_value
            
            logger.info(f"Performed Anderson-Darling normality test: statistic={statistic:.4f}, critical_value={critical_value:.4f}")
            return StatTestResult(test_name="Anderson-Darling normality test", 
                                 statistic=statistic, 
                                 p_value=significance_levels[closest_idx]/100,  # Approximate p-value
                                 alpha=self.alpha,
                                 significant=significant)
        
        else:
            raise ValueError(f"Unknown normality test: {test}. Use 'shapiro', 'ks', or 'anderson'.")
    
    def test_ttest(self, sample1: Union[pd.Series, np.ndarray], 
                  sample2: Optional[Union[pd.Series, np.ndarray]] = None,
                  paired: bool = False,
                  equal_var: bool = True,
                  alternative: str = 'two-sided',
                  popmean: Optional[float] = None) -> StatTestResult:
        """
        Perform t-test for the means of one or two independent samples of scores.
        
        Args:
            sample1: First sample data
            sample2: Second sample data (for independent or paired test)
            paired: Whether to perform a paired test
            equal_var: Whether to assume equal variances (for independent test)
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            popmean: Expected population mean (for one-sample test)
            
        Returns:
            StatTestResult object with test results
        """
        # Convert to numpy arrays if pandas Series
        if isinstance(sample1, pd.Series):
            sample1 = sample1.dropna().values
        
        if sample2 is not None and isinstance(sample2, pd.Series):
            sample2 = sample2.dropna().values
        
        # Determine which t-test to perform
        if sample2 is None and popmean is not None:
            # One-sample t-test
            statistic, p_value = stats.ttest_1samp(sample1, popmean, alternative=alternative)
            test_name = "One-sample t-test"
            logger.info(f"Performed one-sample t-test: statistic={statistic:.4f}, p-value={p_value:.4f}")
        
        elif sample2 is not None and paired:
            # Paired t-test
            if len(sample1) != len(sample2):
                raise ValueError("Paired t-test requires samples of equal length")
            
            statistic, p_value = stats.ttest_rel(sample1, sample2, alternative=alternative)
            test_name = "Paired t-test"
            logger.info(f"Performed paired t-test: statistic={statistic:.4f}, p-value={p_value:.4f}")
        
        elif sample2 is not None and not paired:
            # Independent t-test
            statistic, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var, alternative=alternative)
            test_name = "Independent t-test"
            var_type = "equal" if equal_var else "unequal"
            logger.info(f"Performed independent t-test ({var_type} variances): statistic={statistic:.4f}, p-value={p_value:.4f}")
        
        else:
            raise ValueError("Invalid combination of parameters for t-test")
        
        return StatTestResult(test_name=test_name, 
                             statistic=statistic, 
                             p_value=p_value,
                             alpha=self.alpha)
    
    def test_anova(self, *samples: Union[pd.Series, np.ndarray]) -> StatTestResult:
        """
        Perform one-way ANOVA test on two or more independent samples.
        
        Args:
            *samples: Two or more sample data arrays
            
        Returns:
            StatTestResult object with test results
        """
        if len(samples) < 2:
            raise ValueError("ANOVA requires at least two samples")
        
        # Convert to numpy arrays if pandas Series
        samples_clean = []
        for sample in samples:
            if isinstance(sample, pd.Series):
                samples_clean.append(sample.dropna().values)
            else:
                samples_clean.append(sample)
        
        # Perform one-way ANOVA
        statistic, p_value = stats.f_oneway(*samples_clean)
        logger.info(f"Performed one-way ANOVA: statistic={statistic:.4f}, p-value={p_value:.4f}")
        
        return StatTestResult(test_name="One-way ANOVA", 
                             statistic=statistic, 
                             p_value=p_value,
                             alpha=self.alpha)
    
    def test_correlation(self, x: Union[pd.Series, np.ndarray], 
                        y: Union[pd.Series, np.ndarray],
                        method: str = 'pearson') -> StatTestResult:
        """
        Calculate correlation between two variables.
        
        Args:
            x: First variable
            y: Second variable
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            StatTestResult object with test results
        """
        # Convert to numpy arrays if pandas Series
        if isinstance(x, pd.Series):
            x = x.dropna().values
        
        if isinstance(y, pd.Series):
            y = y.dropna().values
        
        # Check that arrays have the same length
        if len(x) != len(y):
            raise ValueError("Arrays must have the same length")
        
        # Remove pairs with NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 2:
            raise ValueError("Not enough valid data points for correlation")
        
        if method == 'pearson':
            # Pearson correlation
            statistic, p_value = stats.pearsonr(x_clean, y_clean)
            test_name = "Pearson correlation"
        
        elif method == 'spearman':
            # Spearman rank correlation
            statistic, p_value = stats.spearmanr(x_clean, y_clean)
            test_name = "Spearman rank correlation"
        
        elif method == 'kendall':
            # Kendall's tau correlation
            statistic, p_value = stats.kendalltau(x_clean, y_clean)
            test_name = "Kendall's tau correlation"
        
        else:
            raise ValueError(f"Unknown correlation method: {method}. Use 'pearson', 'spearman', or 'kendall'.")
        
        logger.info(f"Performed {test_name}: coefficient={statistic:.4f}, p-value={p_value:.4f}")
        
        return StatTestResult(test_name=test_name, 
                             statistic=statistic, 
                             p_value=p_value,
                             alpha=self.alpha)
    
    def test_chi2(self, observed: Union[pd.DataFrame, np.ndarray], 
                 expected: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> StatTestResult:
        """
        Perform chi-square test of independence or goodness of fit.
        
        Args:
            observed: Observed frequencies (contingency table for independence test)
            expected: Expected frequencies (for goodness of fit test)
            
        Returns:
            StatTestResult object with test results
        """
        # Convert to numpy arrays if pandas DataFrame
        if isinstance(observed, pd.DataFrame):
            observed = observed.values
        
        if expected is not None and isinstance(expected, pd.DataFrame):
            expected = expected.values
        
        if expected is None:
            # Chi-square test of independence
            statistic, p_value, dof, expected = stats.chi2_contingency(observed)
            test_name = "Chi-square test of independence"
            logger.info(f"Performed chi-square test of independence: statistic={statistic:.4f}, p-value={p_value:.4f}, dof={dof}")
        
        else:
            # Chi-square goodness of fit test
            if observed.shape != expected.shape:
                raise ValueError("Observed and expected arrays must have the same shape")
            
            statistic, p_value = stats.chisquare(observed.flatten(), expected.flatten())
            test_name = "Chi-square goodness of fit test"
            logger.info(f"Performed chi-square goodness of fit test: statistic={statistic:.4f}, p-value={p_value:.4f}")
        
        return StatTestResult(test_name=test_name, 
                             statistic=statistic, 
                             p_value=p_value,
                             alpha=self.alpha)
    
    def test_mann_whitney(self, sample1: Union[pd.Series, np.ndarray], 
                         sample2: Union[pd.Series, np.ndarray],
                         alternative: str = 'two-sided') -> StatTestResult:
        """
        Perform Mann-Whitney U test (non-parametric test for independent samples).
        
        Args:
            sample1: First sample data
            sample2: Second sample data
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns:
            StatTestResult object with test results
        """
        # Convert to numpy arrays if pandas Series
        if isinstance(sample1, pd.Series):
            sample1 = sample1.dropna().values
        
        if isinstance(sample2, pd.Series):
            sample2 = sample2.dropna().values
        
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
        logger.info(f"Performed Mann-Whitney U test: statistic={statistic:.4f}, p-value={p_value:.4f}")
        
        return StatTestResult(test_name="Mann-Whitney U test", 
                             statistic=statistic, 
                             p_value=p_value,
                             alpha=self.alpha)
    
    def test_wilcoxon(self, sample1: Union[pd.Series, np.ndarray], 
                     sample2: Optional[Union[pd.Series, np.ndarray]] = None,
                     alternative: str = 'two-sided') -> StatTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric test for paired samples).
        
        Args:
            sample1: First sample data or differences if sample2 is None
            sample2: Second sample data (optional)
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            
        Returns:
            StatTestResult object with test results
        """
        # Convert to numpy arrays if pandas Series
        if isinstance(sample1, pd.Series):
            sample1 = sample1.dropna().values
        
        if sample2 is not None:
            if isinstance(sample2, pd.Series):
                sample2 = sample2.dropna().values
            
            # Calculate differences for paired test
            if len(sample1) != len(sample2):
                raise ValueError("Paired test requires samples of equal length")
            
            differences = sample1 - sample2
        else:
            differences = sample1
        
        # Remove zeros as they are discarded in the Wilcoxon test
        differences = differences[differences != 0]
        
        if len(differences) < 10:
            logger.warning("Wilcoxon test may not be reliable with less than 10 non-zero differences")
        
        # Perform Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(differences, alternative=alternative)
        logger.info(f"Performed Wilcoxon signed-rank test: statistic={statistic:.4f}, p-value={p_value:.4f}")
        
        return StatTestResult(test_name="Wilcoxon signed-rank test", 
                             statistic=statistic, 
                             p_value=p_value,
                             alpha=self.alpha)
    
    def test_kruskal(self, *samples: Union[pd.Series, np.ndarray]) -> StatTestResult:
        """
        Perform Kruskal-Wallis H test (non-parametric version of one-way ANOVA).
        
        Args:
            *samples: Two or more sample data arrays
            
        Returns:
            StatTestResult object with test results
        """
        if len(samples) < 2:
            raise ValueError("Kruskal-Wallis test requires at least two samples")
        
        # Convert to numpy arrays if pandas Series
        samples_clean = []
        for sample in samples:
            if isinstance(sample, pd.Series):
                samples_clean.append(sample.dropna().values)
            else:
                samples_clean.append(sample)
        
        # Perform Kruskal-Wallis H test
        statistic, p_value = stats.kruskal(*samples_clean)
        logger.info(f"Performed Kruskal-Wallis H test: statistic={statistic:.4f}, p-value={p_value:.4f}")
        
        return StatTestResult(test_name="Kruskal-Wallis H test", 
                             statistic=statistic, 
                             p_value=p_value,
                             alpha=self.alpha)
    
    def fit_distribution(self, data: Union[pd.Series, np.ndarray], 
                        dist_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fit various probability distributions to the data and find the best fit.
        
        Args:
            data: Data sample to fit
            dist_names: List of distribution names to try (default: common distributions)
            
        Returns:
            Dictionary with best fit distribution and parameters
        """
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            data = data.dropna().values
        
        if len(data) < 3:
            raise ValueError("At least 3 data points required for distribution fitting")
        
        # Default distributions to try
        if dist_names is None:
            dist_names = ['norm', 'lognorm', 'expon', 'gamma', 'beta', 'weibull_min']
        
        # Initialize results
        results = {}
        best_sse = np.inf
        best_dist = None
        best_params = None
        
        # Try each distribution
        for dist_name in dist_names:
            try:
                # Get distribution from scipy.stats
                distribution = getattr(stats, dist_name)
                
                # Fit distribution to data
                params = distribution.fit(data)
                
                # Calculate fitted PDF and error with actual data
                arg_params = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Get PDF values
                x = np.linspace(min(data), max(data), 100)
                pdf_fitted = distribution.pdf(x, loc=loc, scale=scale, *arg_params)
                
                # Calculate histogram of actual data
                hist, bin_edges = np.histogram(data, bins=50, density=True)
                bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
                
                # Interpolate PDF values at bin centers
                from scipy.interpolate import interp1d
                pdf_interpolated = interp1d(x, pdf_fitted, bounds_error=False, fill_value=0)(bin_centers)
                
                # Calculate sum of squared errors
                sse = np.sum((hist - pdf_interpolated)**2)
                
                # Store results
                results[dist_name] = {
                    'params': params,
                    'sse': sse
                }
                
                # Update best fit
                if sse < best_sse:
                    best_sse = sse
                    best_dist = dist_name
                    best_params = params
                
                logger.info(f"Fitted {dist_name} distribution: SSE={sse:.4f}")
                
            except Exception as e:
                logger.warning(f"Error fitting {dist_name} distribution: {str(e)}")
        
        # Prepare result with best fit
        if best_dist is not None:
            distribution = getattr(stats, best_dist)
            arg_params = best_params[:-2]
            loc = best_params[-2]
            scale = best_params[-1]
            
            # Calculate goodness of fit with Kolmogorov-Smirnov test
            ks_statistic, ks_p_value = stats.kstest(
                data, 
                lambda x: distribution.cdf(x, loc=loc, scale=scale, *arg_params)
            )
            
            best_fit = {
                'distribution': best_dist,
                'params': best_params,
                'loc': loc,
                'scale': scale,
                'arg_params': arg_params,
                'sse': best_sse,
                'ks_statistic': ks_statistic,
                'ks_p_value': ks_p_value
            }
            
            logger.info(f"Best fit distribution: {best_dist} with SSE={best_sse:.4f}")
            
            return {
                'best_fit': best_fit,
                'all_fits': results
            }
        else:
            logger.warning("No distribution could be fitted to the data")
            return {
                'best_fit': None,
                'all_fits': results
            }
    
    def power_analysis(self, test_type: str, 
                      effect_size: float,
                      alpha: Optional[float] = None,
                      power: float = 0.8,
                      sample_size: Optional[int] = None,
                      nobs: Optional[int] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Perform statistical power analysis for various tests.
        
        Args:
            test_type: Type of test ('t', 'paired-t', 'z', 'f', 'chi2')
            effect_size: Expected effect size (Cohen's d, f, etc.)
            alpha: Significance level (default: self.alpha)
            power: Desired power (default: 0.8)
            sample_size: Sample size (to calculate power)
            nobs: Number of observations (for some tests)
            **kwargs: Additional test-specific parameters
            
        Returns:
            Dictionary with power analysis results
        """
        try:
            from statsmodels.stats.power import TTestPower, TTestIndPower, FTestAnovaPower, GofChisquarePower, NormalIndPower
        except ImportError:
            logger.error("statsmodels package is required for power analysis")
            raise ImportError("statsmodels package is required for power analysis")
        
        if alpha is None:
            alpha = self.alpha
        
        results = {}
        
        if test_type == 't':
            # One-sample or two-sample t-test
            power_analysis = TTestIndPower()
            
            if sample_size is None:
                # Calculate required sample size
                sample_size = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    **kwargs
                )
                results['sample_size'] = sample_size
                logger.info(f"Required sample size for t-test: {sample_size:.2f}")
            else:
                # Calculate achieved power
                achieved_power = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    nobs=sample_size,
                    **kwargs
                )
                results['power'] = achieved_power
                logger.info(f"Achieved power for t-test with n={sample_size}: {achieved_power:.4f}")
        
        elif test_type == 'paired-t':
            # Paired t-test
            power_analysis = TTestPower()
            
            if sample_size is None:
                # Calculate required sample size
                sample_size = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    **kwargs
                )
                results['sample_size'] = sample_size
                logger.info(f"Required sample size for paired t-test: {sample_size:.2f}")
            else:
                # Calculate achieved power
                achieved_power = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    nobs=sample_size,
                    **kwargs
                )
                results['power'] = achieved_power
                logger.info(f"Achieved power for paired t-test with n={sample_size}: {achieved_power:.4f}")
        
        elif test_type == 'z':
            # Z-test (normal distribution)
            power_analysis = NormalIndPower()
            
            if sample_size is None:
                # Calculate required sample size
                sample_size = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    **kwargs
                )
                results['sample_size'] = sample_size
                logger.info(f"Required sample size for z-test: {sample_size:.2f}")
            else:
                # Calculate achieved power
                achieved_power = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    nobs=sample_size,
                    **kwargs
                )
                results['power'] = achieved_power
                logger.info(f"Achieved power for z-test with n={sample_size}: {achieved_power:.4f}")
        
        elif test_type == 'f':
            # ANOVA F-test
            power_analysis = FTestAnovaPower()
            
            k_groups = kwargs.get('k_groups', 3)  # Default to 3 groups
            
            if sample_size is None:
                # Calculate required sample size per group
                sample_size = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    k_groups=k_groups,
                    **kwargs
                )
                results['sample_size_per_group'] = sample_size
                results['total_sample_size'] = sample_size * k_groups
                logger.info(f"Required sample size for ANOVA with {k_groups} groups: {sample_size:.2f} per group, {sample_size * k_groups:.2f} total")
            else:
                # Calculate achieved power
                achieved_power = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    nobs=sample_size,
                    k_groups=k_groups,
                    **kwargs
                )
                results['power'] = achieved_power
                logger.info(f"Achieved power for ANOVA with n={sample_size} per group: {achieved_power:.4f}")
        
        elif test_type == 'chi2':
            # Chi-square test
            power_analysis = GofChisquarePower()
            
            if nobs is None and sample_size is not None:
                nobs = sample_size
            
            if nobs is None:
                # Calculate required sample size
                nobs = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    **kwargs
                )
                results['sample_size'] = nobs
                logger.info(f"Required sample size for chi-square test: {nobs:.2f}")
            else:
                # Calculate achieved power
                achieved_power = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    nobs=nobs,
                    **kwargs
                )
                results['power'] = achieved_power
                logger.info(f"Achieved power for chi-square test with n={nobs}: {achieved_power:.4f}")
        
        else:
            raise ValueError(f"Unknown test type: {test_type}. Use 't', 'paired-t', 'z', 'f', or 'chi2'.")
        
        # Add input parameters to results
        results['test_type'] = test_type
        results['effect_size'] = effect_size
        results['alpha'] = alpha
        
        return results
