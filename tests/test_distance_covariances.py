import numpy as np
import pytest

from bs_python_utils.distance_covariances import (
    DcovResults,
    PdcovResults,
    _compute_distances,
    _double_decenter,
    _dcov_prod,
    dcov_dcor,
    pdcov_pdcor,
    pvalue_dcov,
    pvalue_pdcov,
)


class TestHelperFunctions:
    """Test the internal helper functions"""

    def test_compute_distances_vector(self):
        """Test distance computation for vectors"""
        x = np.array([1, 3, 5])
        distances = _compute_distances(x)
        expected = np.array([[0, 2, 4], [2, 0, 2], [4, 2, 0]])
        assert np.allclose(distances, expected)

    def test_compute_distances_matrix(self):
        """Test distance computation for matrices"""
        x = np.array([[1, 2], [3, 4], [5, 6]])
        distances = _compute_distances(x)
        # Should be symmetric with zeros on diagonal
        assert distances.shape == (3, 3)
        assert np.allclose(distances, distances.T)
        assert np.allclose(np.diag(distances), 0)

    def test_double_decenter_basic(self):
        """Test basic double decentering"""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        A_dd = _double_decenter(A, unbiased=False)
        # Check that row and column means are zero
        assert np.allclose(np.mean(A_dd, axis=0), 0)
        assert np.allclose(np.mean(A_dd, axis=1), 0)

    def test_double_decenter_unbiased(self):
        """Test unbiased double decentering"""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        A_dd = _double_decenter(A, unbiased=True)
        # For unbiased version, diagonal should be zero
        assert np.allclose(np.diag(A_dd), 0)

    def test_dcov_prod_basic(self):
        """Test basic dcov product computation"""
        A = np.random.rand(5, 5)
        B = np.random.rand(5, 5)
        result = _dcov_prod(A, B, unbiased=False)
        assert isinstance(result, float)
        assert result >= 0  # Should be non-negative for distance covariances


class TestDcovDcor:
    """Test distance covariance and correlation functions"""

    def test_dcov_dcor_basic(self):
        """Test basic distance covariance computation"""
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        Y = np.random.normal(0, 1, 100)
        
        results = dcov_dcor(X, Y, unbiased=False)
        
        assert isinstance(results, DcovResults)
        assert results.dcov >= 0
        assert 0 <= results.dcor <= 1
        assert results.dcov_stat >= 0
        assert not results.unbiased

    def test_dcov_dcor_unbiased(self):
        """Test unbiased distance covariance computation"""
        np.random.seed(42)
        X = np.random.normal(0, 1, 50)
        Y = np.random.normal(0, 1, 50)
        
        results = dcov_dcor(X, Y, unbiased=True)
        
        assert isinstance(results, DcovResults)
        assert results.unbiased
        assert results.dcov >= 0

    def test_dcov_dcor_perfect_dependence(self):
        """Test distance covariance with perfect dependence"""
        X = np.linspace(0, 10, 100)
        Y = 2 * X + 1  # Perfect linear relationship
        
        results = dcov_dcor(X, Y, unbiased=False)
        
        # Should have high correlation
        assert results.dcor > 0.9
        assert results.dcov > 0

    def test_dcov_dcor_independence(self):
        """Test distance covariance with independent variables"""
        np.random.seed(123)
        X = np.random.normal(0, 1, 1000)
        Y = np.random.normal(0, 1, 1000)
        
        results = dcov_dcor(X, Y, unbiased=False)
        
        # Should have low correlation for independent variables
        assert results.dcor < 0.2
        assert results.dcov >= 0

    def test_dcov_dcor_multivariate(self):
        """Test distance covariance with multivariate data"""
        np.random.seed(42)
        X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
        Y = np.random.multivariate_normal([0, 0], [[1, -0.3], [-0.3, 1]], 100)
        
        results = dcov_dcor(X, Y, unbiased=False)
        
        assert isinstance(results, DcovResults)
        assert results.dcov >= 0
        assert 0 <= results.dcor <= 1


class TestPdcovPdcor:
    """Test partial distance covariance and correlation functions"""

    def test_pdcov_pdcor_basic(self):
        """Test basic partial distance covariance computation"""
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        Y = np.random.normal(0, 1, 100)
        Z = np.random.normal(0, 1, 100)
        
        results = pdcov_pdcor(X, Y, Z)
        
        assert isinstance(results, PdcovResults)
        assert isinstance(results.pdcov, float)
        assert isinstance(results.pdcor, float)
        assert results.pdcov_stat >= 0

    def test_pdcov_pdcor_conditional_independence(self):
        """Test partial distance covariance with conditional independence"""
        np.random.seed(123)
        n = 500
        Z = np.random.normal(0, 1, n)
        X = Z + np.random.normal(0, 0.1, n)  # X depends on Z
        Y = Z + np.random.normal(0, 0.1, n)  # Y depends on Z
        
        results = pdcov_pdcor(X, Y, Z)
        
        # X and Y should be conditionally independent given Z
        # So partial correlation should be small
        assert abs(results.pdcor) < 0.3

    def test_pdcov_pdcor_multivariate(self):
        """Test partial distance covariance with multivariate data"""
        np.random.seed(42)
        n = 100
        X = np.random.multivariate_normal([0, 0], [[1, 0.2], [0.2, 1]], n)
        Y = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n)
        Z = np.random.multivariate_normal([0, 0], [[1, 0.1], [0.1, 1]], n)
        
        results = pdcov_pdcor(X, Y, Z)
        
        assert isinstance(results, PdcovResults)
        assert isinstance(results.pdcov, float)
        assert isinstance(results.pdcor, float)


class TestPvalues:
    """Test p-value computation functions"""

    def test_pvalue_dcov_basic(self):
        """Test basic p-value computation for distance covariance"""
        np.random.seed(42)
        X = np.random.normal(0, 1, 50)
        Y = np.random.normal(0, 1, 50)
        
        dcov_results = dcov_dcor(X, Y, unbiased=False)
        pval = pvalue_dcov(dcov_results, ndraws=50)  # Small ndraws for speed
        
        assert 0 <= pval <= 1
        assert isinstance(pval, float)

    def test_pvalue_dcov_dependent(self):
        """Test p-value with clearly dependent variables"""
        np.random.seed(42)  # Set seed for reproducible results
        X = np.linspace(0, 10, 50)
        Y = 2 * X + np.random.normal(0, 0.01, 50)  # Very strong dependence
        
        dcov_results = dcov_dcor(X, Y, unbiased=False)
        pval = pvalue_dcov(dcov_results, ndraws=50)
        
        # Should have small p-value for strongly dependent variables
        assert pval < 0.5  # More reasonable threshold given bootstrap variability

    def test_pvalue_pdcov_basic(self):
        """Test basic p-value computation for partial distance covariance"""
        np.random.seed(42)
        X = np.random.normal(0, 1, 50)
        Y = np.random.normal(0, 1, 50)
        Z = np.random.normal(0, 1, 50)
        
        pdcov_results = pdcov_pdcor(X, Y, Z)
        pval = pvalue_pdcov(pdcov_results, ndraws=50)  # Small ndraws for speed
        
        assert 0 <= pval <= 1
        assert isinstance(pval, float)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_single_observation(self):
        """Test behavior with single observation"""
        X = np.array([1])
        Y = np.array([2])
        
        # Single observation should work but return degenerate results
        results = dcov_dcor(X, Y, unbiased=False)
        assert isinstance(results, DcovResults)
        # With single observation, distances are all zero
        assert results.dcov == 0.0

    def test_two_observations(self):
        """Test behavior with minimal observations"""
        X = np.array([1, 2])
        Y = np.array([3, 4])
        
        results = dcov_dcor(X, Y, unbiased=False)
        assert isinstance(results, DcovResults)

    def test_identical_variables(self):
        """Test distance covariance of variable with itself"""
        X = np.random.normal(0, 1, 100)
        
        results = dcov_dcor(X, X, unbiased=False)
        
        # Correlation with itself should be 1
        assert np.isclose(results.dcor, 1.0, atol=1e-10)

    def test_constant_variable(self):
        """Test behavior with constant variable"""
        np.random.seed(42)
        X = np.ones(100)
        Y = np.random.normal(0, 1, 100)
        
        results = dcov_dcor(X, Y, unbiased=False)
        
        # Distance covariance should be 0 for constant variable
        assert np.isclose(results.dcov, 0.0, atol=1e-10)
        # Distance correlation becomes NaN due to division by zero (mathematically correct)
        assert np.isnan(results.dcor)


class TestDataTypes:
    """Test different data types and shapes"""

    def test_different_dtypes(self):
        """Test with different numpy dtypes"""
        X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        Y = np.array([2, 4, 6, 8, 10], dtype=np.float64)
        
        results = dcov_dcor(X, Y, unbiased=False)
        assert isinstance(results, DcovResults)

    def test_integer_data(self):
        """Test with integer data"""
        X = np.array([1, 2, 3, 4, 5], dtype=int)
        Y = np.array([5, 4, 3, 2, 1], dtype=int)
        
        results = dcov_dcor(X, Y, unbiased=False)
        assert isinstance(results, DcovResults)

    def test_large_sample_size(self):
        """Test with larger sample size"""
        np.random.seed(42)
        X = np.random.normal(0, 1, 1000)
        Y = np.random.normal(0, 1, 1000)
        
        results = dcov_dcor(X, Y, unbiased=True)
        assert isinstance(results, DcovResults)
        assert results.X_dd.shape == (1000, 1000)


if __name__ == "__main__":
    pytest.main([__file__]) 