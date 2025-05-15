import numpy as np
from pymcdm import methods, normalizations

class TOPSIS_COMET:
    def __init__(self, weights=None, n_levels=3):
        self.weights = weights
        self.n_levels = n_levels
        self.characteristic_objects = None

    def fit(self, decision_matrix):
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(decision_matrix)
        
        if self.weights is None:
            self.weights = np.ones(normalized.shape[1]) / normalized.shape[1]
        else:
            self.weights = np.array(self.weights)
        
        weighted = normalized * self.weights
        ideal = np.max(weighted, axis=0)
        anti_ideal = np.min(weighted, axis=0)
        
        dist_ideal = np.linalg.norm(weighted - ideal, axis=1)
        dist_anti_ideal = np.linalg.norm(weighted - anti_ideal, axis=1)
        
        self.topsis_scores = dist_anti_ideal / (dist_ideal + dist_anti_ideal)
        
        quantiles = np.percentile(self.topsis_scores, 
                                 np.linspace(0, 100, self.n_levels + 1))
        
        self.characteristic_objects = []
        for i in range(self.n_levels):
            low, high = quantiles[i], quantiles[i+1]
            utility = (i + 1) / self.n_levels
            self.characteristic_objects.append({'range': (low, high), 'utility': utility})
        return self

    def predict(self, decision_matrix):
        def assign_utility(score):
            for co in self.characteristic_objects:
                low, high = co['range']
                if low <= score < high or (score == high and co == self.characteristic_objects[-1]):
                    return co['utility']
            return 0
        
        utilities = np.array([assign_utility(score) for score in self.topsis_scores])
        ranking = np.argsort(-utilities)
        return utilities, ranking


class MCDMEvaluator:
    """Combines multiple MCDM methods for robust recommendations"""
    
    def __init__(self):
        # Initialize MCDM methods
        self.topsis_comet = TOPSIS_COMET()
        self.cocoso = methods.COCOSO()
        self.edas = methods.EDAS()
        self.mabac = methods.MABAC()
        self.mairca = methods.MAIRCA()
        
    def normalize_matrix(self, matrix):
        """Normalize decision matrix using min-max normalization"""
        return normalizations.minmax_normalization(matrix)
    
    def critic_weights(self, matrix):
        """Calculate criteria weights using CRITIC method"""
        std_dev = np.std(matrix, axis=0)
        correlation_matrix = np.abs(np.corrcoef(matrix.T))
        conflict = 1 - correlation_matrix
        conflict_sum = np.sum(conflict, axis=1)
        weights = std_dev * conflict_sum
        return weights / np.sum(weights)
    
    def evaluate_alternatives(self, decision_matrix, weights=None):
        if weights is None:
            weights = self.critic_weights(decision_matrix)
        
        normalized_matrix = self.normalize_matrix(decision_matrix)
        
        # Use TOPSIS-COMET hybrid
        self.topsis_comet.weights = weights
        self.topsis_comet.fit(decision_matrix)
        comet_utilities, comet_ranking = self.topsis_comet.predict(decision_matrix)
        
        # Evaluate others normally
        rankings = {
            'topsis-comet': comet_utilities,
            'cocoso': self.cocoso(normalized_matrix, weights),
            'edas': self.edas(normalized_matrix, weights),
            'mabac': self.mabac(normalized_matrix, weights),
            'mairca': self.mairca(normalized_matrix, weights)
        }
        
        n_alt = len(decision_matrix)
        copeland_scores = np.zeros(n_alt)
        methods_list = list(rankings.keys())
        
        # For Copeland: use pairwise majority wins counting based on utilities or scores
        for i in range(n_alt):
            for j in range(i + 1, n_alt):
                preference_count = 0
                for method in methods_list:
                    # Higher score means preferred
                    if rankings[method][i] > rankings[method][j]:
                        preference_count += 1
                    elif rankings[method][i] < rankings[method][j]:
                        preference_count -= 1
                if preference_count > 0:
                    copeland_scores[i] += 1
                    copeland_scores[j] -= 1
                elif preference_count < 0:
                    copeland_scores[i] -= 1
                    copeland_scores[j] += 1
                # tie means no change
                
        return copeland_scores
