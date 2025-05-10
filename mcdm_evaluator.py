import numpy as np
from pymcdm import methods, normalizations
from scipy.stats import spearmanr

class MCDMEvaluator:
    """Combines multiple MCDM methods for robust recommendations"""
    
    def __init__(self):
        # Initialize MCDM methods
        self.topsis = methods.TOPSIS()
        self.cocoso = methods.COCOSO()
        self.edas = methods.EDAS()
        self.mabac = methods.MABAC()
        self.mairca = methods.MAIRCA()
        
    def normalize_matrix(self, matrix):
        """Normalize decision matrix using min-max normalization"""
        return normalizations.minmax_normalization(matrix)
    
    def critic_weights(self, matrix):
        """Calculate criteria weights using CRITIC method"""
        # Standard deviation for each criterion
        std_dev = np.std(matrix, axis=0)
        
        # Correlation between criteria
        correlation_matrix = np.abs(np.corrcoef(matrix.T))
        
        # Calculate conflict measure
        conflict = 1 - correlation_matrix
        conflict_sum = np.sum(conflict, axis=1)
        
        # Calculate weights
        weights = std_dev * conflict_sum
        return weights / np.sum(weights)
    
    def evaluate_alternatives(self, decision_matrix, weights=None):
        """Evaluate alternatives using multiple MCDM methods"""
        if weights is None:
            weights = self.critic_weights(decision_matrix)
        
        # Normalize matrix
        normalized_matrix = self.normalize_matrix(decision_matrix)
        
        # Get rankings from different methods
        rankings = {
            'topsis': self.topsis(normalized_matrix, weights),
            'cocoso': self.cocoso(normalized_matrix, weights),
            'edas': self.edas(normalized_matrix, weights),
            'mabac': self.mabac(normalized_matrix, weights),
            'mairca': self.mairca(normalized_matrix, weights)
        }
        
        # Apply Copeland method for consensus ranking
        n_alternatives = len(decision_matrix)
        copeland_scores = np.zeros(n_alternatives)
        
        for i in range(n_alternatives):
            for j in range(i + 1, n_alternatives):
                # Count how many methods prefer i over j
                preference_count = sum(1 for method in rankings.values() 
                                    if rankings[method][i] > rankings[method][j])
                
                if preference_count > len(rankings) / 2:
                    copeland_scores[i] += 1
                    copeland_scores[j] -= 1
        
        return copeland_scores