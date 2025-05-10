import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pymcdm
from pymcdm import methods as mcdm_methods
from pymcdm import normalizations as norm
import gc  # Garbage collection
import os
import json
import scipy.sparse
import re  # Add this import for regex operations
from mcdm_evaluator import MCDMEvaluator

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, x):
        return x + self.block(x)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        attn_output, _ = self.attention(x, x, x)
        return self.norm(attn_output.squeeze(1))

class LightweightProductEmbeddingModel(nn.Module):
    def __init__(self, input_size, embedding_size=512, hidden_size=1024):
        super().__init__()
        
        # Improved encoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            
            ResidualBlock(hidden_size),
            SelfAttention(hidden_size),
            
            nn.Linear(hidden_size, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.Tanh()
        )
        
        # Symmetric decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            
            ResidualBlock(hidden_size),
            SelfAttention(hidden_size),
            
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size * 2, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def get_embeddings(self, x):
        return self.encoder(x)

class MemoryEfficientProductRecommender:
    """
    Memory-efficient version of the product recommender
    """
    
    def __init__(self, embedding_size=32, hidden_size=64, batch_size=32, 
                 max_features=2000, epochs=10, learning_rate=0.001, 
                 use_cuda=True, chunk_size=1000, config=None, **kwargs):
        # Store config for later use
        self.config = config if config is not None else {}

        # Model parameters (from config or arguments)
        self.embedding_size = self.config.get('model_params', {}).get('embedding_size', embedding_size)
        self.hidden_size = self.config.get('model_params', {}).get('hidden_size', hidden_size)
        self.batch_size = self.config.get('model_params', {}).get('batch_size', batch_size)
        self.max_features = self.config.get('model_params', {}).get('max_features', max_features)
        self.epochs = self.config.get('model_params', {}).get('epochs', epochs)
        self.learning_rate = self.config.get('model_params', {}).get('learning_rate', learning_rate)
        self.use_cuda = self.config.get('model_params', {}).get('use_cuda', use_cuda) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.chunk_size = self.config.get('model_params', {}).get('chunk_size', chunk_size)

        # Feature selection parameters
        fs = self.config.get('feature_selection', {})
        self.min_df = fs.get('min_df', 5)
        self.max_df = fs.get('max_df', 0.8)
        self.ngram_range = tuple(fs.get('ngram_range', [1, 2]))

        # Initialize components
        self.products_df = None
        self.vectorizer = None
        self.model = None
        self.similarity_matrix = None
        self.trained = False
        self.mcdm_evaluator = MCDMEvaluator()

        print(f"Using device: {self.device}")
        print(f"Model parameters: embedding_size={self.embedding_size}, hidden_size={self.hidden_size}, "
              f"batch_size={self.batch_size}, max_features={self.max_features}")
    
    def preprocess_data(self, product_data_list=None, product_data_path=None, sample_size=None):
        """
        Preprocess product data with options for sampling and chunking
        
        Args:
            product_data_list: List of product dictionaries (optional)
            product_data_path: Path to JSONL file (optional)
            sample_size: Number of products to sample
        """
        if product_data_path and not product_data_list:
            # Stream from file instead of loading everything to memory
            print(f"Loading and preprocessing data from {product_data_path}")
            # Check file size for warning
            file_size_mb = os.path.getsize(product_data_path) / (1024 * 1024)
            print(f"File size: {file_size_mb:.2f} MB")
            
            product_chunks = []
            products_processed = 0
            
            with open(product_data_path, 'r', encoding='utf-8') as f:
                chunk = []
                for line in f:
                    if line.strip():
                        # Skip if we've reached sample size
                        if sample_size and products_processed >= sample_size:
                            break
                            
                        try:
                            product = json.loads(line)
                            chunk.append(product)
                            products_processed += 1
                            
                            # Process in chunks to avoid memory issues
                            if len(chunk) >= self.chunk_size:
                                # Create DataFrame for this chunk
                                chunk_df = pd.DataFrame(chunk)
                                # Add combined features
                                chunk_df['combined_features'] = chunk_df.apply(self._create_combined_features, axis=1)
                                product_chunks.append(chunk_df)
                                chunk = []
                                
                                print(f"Processed {products_processed} products", end="\r")
                                
                                # Force garbage collection
                                gc.collect()
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid JSON line")
                        except Exception as e:
                            print(f"Error: {e}")
                
                # Process any remaining products
                if chunk:
                    chunk_df = pd.DataFrame(chunk)
                    chunk_df['combined_features'] = chunk_df.apply(self._create_combined_features, axis=1)
                    product_chunks.append(chunk_df)
            
            # Combine all chunks
            self.products_df = pd.concat(product_chunks, ignore_index=True)
            print(f"\nTotal products processed: {len(self.products_df)}")
            
        elif product_data_list:
            # Use provided list
            if sample_size and len(product_data_list) > sample_size:
                # Sample randomly
                import random
                product_data_list = random.sample(product_data_list, sample_size)
                
            print(f"Processing {len(product_data_list)} products from provided list")
            self.products_df = pd.DataFrame(product_data_list)
            self.products_df['combined_features'] = self.products_df.apply(self._create_combined_features, axis=1)
        else:
            raise ValueError("Either product_data_list or product_data_path must be provided")
            
        # Create a unique index if not present
        if 'product_id' not in self.products_df.columns:
            self.products_df['product_id'] = range(len(self.products_df))
        
        # Add rating info if available, otherwise use default
        if 'average_rating' not in self.products_df.columns:
            self.products_df['average_rating'] = 3.0
        
        if 'rating_number' not in self.products_df.columns:
            self.products_df['rating_number'] = 1
            
        # Force garbage collection
        gc.collect()
            
        return self.products_df
    
    def _create_combined_features(self, row):
        """Enhanced feature extraction for subscription boxes"""
        features = []
        
        # Subscription-specific features with high weight
        if 'title' in row and pd.notna(row['title']):
            title = str(row['title']).lower()
            features.extend([title] * 3)  # Triple weight for title
            
            # Extract subscription keywords
            sub_keywords = ['monthly', 'subscription', 'box', 'club', 'kit']
            for keyword in sub_keywords:
                if keyword in title:
                    features.extend([keyword] * 2)
        
        # Category features
        if 'main_category' in row and pd.notna(row['main_category']):
            cat = str(row['main_category'])
            features.extend([cat] * 2)
            
            # Extract subscription type
            if 'subscription' in cat.lower():
                features.extend(['subscription_product'] * 3)
        
        # Product features with emphasis on subscription aspects
        if 'features' in row and isinstance(row['features'], list):
            for feat in row['features']:
                feat_str = str(feat).lower()
                features.append(feat_str)
                
                # Emphasized subscription-related features
                if any(keyword in feat_str for keyword in 
                      ['monthly', 'subscription', 'deliver', 'curated', 'exclusive']):
                    features.extend([feat_str] * 2)
        
        # Price and value indicators
        if 'price' in row and pd.notna(row['price']):
            price = float(row['price'])
            features.append(f'price_range_{int(price/10)*10}')
        
        # Rating features with weight
        if 'average_rating' in row and pd.notna(row['average_rating']):
            rating = float(row['average_rating'])
            features.extend([f'rating_{int(rating)}'] * 2)
        
        # Clean and normalize
        cleaned_features = []
        for feature in features:
            cleaned = re.sub(r'[^\w\s-]', ' ', str(feature))
            cleaned = re.sub(r'\s+', '_', cleaned.strip())
            if cleaned and len(cleaned) > 2:
                cleaned_features.append(cleaned.lower())
        
        return ' '.join(set(cleaned_features))
    
    class LightweightProductTextDataset(Dataset):
        """Memory-efficient dataset for sparse feature vectors"""
        
        def __init__(self, feature_vectors):
            self.feature_vectors = feature_vectors
            
        def __len__(self):
            return self.feature_vectors.shape[0]  # Fixed length calculation
            
        def __getitem__(self, idx):
            # Handle sparse matrix properly
            if scipy.sparse.issparse(self.feature_vectors):
                features = self.feature_vectors[idx].toarray().squeeze()
            else:
                features = self.feature_vectors[idx]
            return torch.FloatTensor(features)
    
    def _create_feature_vectors(self):
        """Create optimized feature vectors with proper error handling"""
        print("Creating feature vectors...")
        
        if not hasattr(self, 'products_df') or self.products_df is None:
            raise ValueError("Products DataFrame not initialized")
            
        if 'combined_features' not in self.products_df.columns:
            raise ValueError("Combined features not found in DataFrame")
        
        try:
            # Get parameters from config with fallbacks
            max_features = self.config.get('model_params', {}).get('max_features', 8000)
            min_df = self.config.get('feature_selection', {}).get('min_df', 3)
            max_df = self.config.get('feature_selection', {}).get('max_df', 0.9)
            ngram_range = tuple(self.config.get('feature_selection', {}).get('ngram_range', [1, 3]))
            
            # Initialize TF-IDF with parameters
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                ngram_range=ngram_range,
                stop_words='english'
            )
            
            # Create feature matrix
            feature_matrix = vectorizer.fit_transform(self.products_df['combined_features'])
            print(f"Feature vectors shape: {feature_matrix.shape}")
            
            # Store vectorizer for later use
            self.vectorizer = vectorizer
            
            # Convert to dense for small datasets
            if feature_matrix.shape[0] < 50000:
                return feature_matrix.toarray()
            return feature_matrix
                
        except Exception as e:
            print(f"Error creating feature vectors: {str(e)}")
            raise
    
    def train_model(self):
        feature_vectors = self._create_feature_vectors()
        input_size = feature_vectors.shape[1]
        
        print(f"Feature vectors shape: {feature_vectors.shape}")
        
        # Initialize model using the outer class
        self.model = LightweightProductEmbeddingModel(
            input_size=input_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size
        ).to(self.device)
        
        # Create dataset and dataloader
        dataset = self.LightweightProductTextDataset(feature_vectors)
        dataloader = DataLoader(dataset, 
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=0)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, features in enumerate(dataloader):
                features = features.to(self.device)
                
                # Forward pass
                encoded, decoded = self.model(features)
                loss = criterion(decoded, features)
                
                # Backward pass with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        self.trained = True
        print("Training completed!")
        
        # Generate and store embeddings after training
        print("Generating product embeddings...")
        self.all_embeddings = self._generate_embeddings(feature_vectors)
    
    def _generate_embeddings(self, feature_vectors=None):
        """Generate embeddings for all products"""
        if not self.trained:
            raise ValueError("Model not trained. Call train_model first.")
        
        if feature_vectors is None:
            feature_vectors = self._create_feature_vectors()
        
        # Process in batches to avoid memory issues
        batch_size = min(self.batch_size * 2, 64)  # Can use larger batch for inference
        
        # Create a simple dataset without labels
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(feature_vectors.toarray() if hasattr(feature_vectors, 'toarray') else feature_vectors, 
                         dtype=torch.float32)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        all_embeddings = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch[0].to(self.device)
                embeddings = self.model.get_embeddings(features)
                all_embeddings.append(embeddings.cpu().numpy())
                
                # Force garbage collection every few batches
                if len(all_embeddings) % 20 == 0:
                    gc.collect()
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        print(f"Generated embeddings shape: {all_embeddings.shape}")
        
        # Calculate similarity matrix using embeddings
        # If the dataset is too large, we'll calculate similarities on-demand instead
        if len(all_embeddings) > 10000:
            print("Large dataset detected. Similarity will be calculated on-demand.")
            self.all_embeddings = all_embeddings
            self.similarity_matrix = None
        else:
            print("Calculating similarity matrix...")
            self.similarity_matrix = cosine_similarity(all_embeddings)
        
        return all_embeddings
    
    def get_similarity(self, idx1, idx2):
        """Get similarity between two products (memory-efficient)"""
        if self.similarity_matrix is not None:
            return self.similarity_matrix[idx1, idx2]
        else:
            # Calculate on-demand
            emb1 = self.all_embeddings[idx1].reshape(1, -1)
            emb2 = self.all_embeddings[idx2].reshape(1, -1)
            return cosine_similarity(emb1, emb2)[0, 0]
    
    def get_recommendations(self, product_idx, top_n=5):
        """Enhanced recommendations combining MCDM methods with existing approach"""
        if not self.trained:
            raise ValueError("Model not trained yet")

        query_product = self.products_df.iloc[product_idx]
        query_embedding = self.all_embeddings[product_idx].reshape(1, -1)
        
        # Get candidates excluding query product
        candidates = []
        decision_matrices = []
        
        for idx in range(len(self.products_df)):
            if idx == product_idx:
                continue
                
            candidate = self.products_df.iloc[idx]
            
            # Calculate base similarities
            embedding_sim = cosine_similarity(query_embedding, 
                                           self.all_embeddings[idx].reshape(1, -1))[0][0]
            title_sim = self._text_similarity(query_product['title'], candidate['title'])
            features_sim = self._text_similarity(
                ' '.join(query_product['features']), 
                ' '.join(candidate['features'])
            )
            rating_score = (candidate['average_rating'] * 
                           np.log1p(candidate['rating_number']))
            
            # Create decision matrix for MCDM methods
            decision_matrix = np.array([
                [embedding_sim],
                [title_sim],
                [features_sim],
                [rating_score]
            ])
            
            decision_matrices.append(decision_matrix)
            candidates.append(idx)
        
        # Stack all decision matrices
        full_decision_matrix = np.hstack(decision_matrices).T
        
        # Get criteria weights from config
        criteria_weights = np.array(list(
            self.config['recommendation_params']['criteria_weights'].values()
        ))
        
        # Apply MCDM methods
        mcdm_scores = self.mcdm_evaluator.evaluate_alternatives(
            full_decision_matrix, 
            weights=criteria_weights
        )
        
        # Combine with original scoring for stability
        final_scores = []
        mcdm_weight = 0.3  # Controls influence of MCDM methods
        
        for idx, (candidate_idx, mcdm_score) in enumerate(zip(candidates, mcdm_scores)):
            # Original score calculation
            orig_score = (
                criteria_weights[0] * full_decision_matrix[idx, 0] +  # embedding_sim
                criteria_weights[1] * full_decision_matrix[idx, 1] +  # title_sim
                criteria_weights[2] * full_decision_matrix[idx, 2] +  # features_sim
                criteria_weights[3] * full_decision_matrix[idx, 3]    # rating_score
            )
            
            # Combine scores
            final_score = (1 - mcdm_weight) * orig_score + mcdm_weight * mcdm_score
            final_scores.append((final_score, candidate_idx))
        
        # Sort and get top-n recommendations
        final_scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for score, idx in final_scores[:top_n]]
        
        return self.products_df.iloc[top_indices].copy()
    
    def get_recommendations_by_title(self, title, top_n=5):
        """Get recommendations for a product based on its title"""
        # Find the product by title (case insensitive partial match)
        matches = self.products_df[self.products_df['title'].str.contains(title, case=False)]
        
        if len(matches) == 0:
            raise ValueError(f"No product found with title containing '{title}'")
        
        # Get recommendations for the first matching product
        return self.get_recommendations(matches.index[0], top_n)

    def save_model(self, path='recommender_model.pt'):
        """Save the trained model and parameters"""
        if not self.trained:
            raise ValueError("Model not trained. Cannot save.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'input_size': self.model.encoder[0].in_features
        }, path)

    def evaluate_metrics(self, test_data, top_ks=[5, 10, 20]):
        """Enhanced evaluation metrics with better similarity handling"""
        metrics = {}
        total = len(test_data)
        
        for k in top_ks:
            hits = 0
            precision_sum = 0
            recall_sum = 0
            ndcg_sum = 0
            
            for idx, test_product in enumerate(test_data):
                # Find similar products based on multiple criteria
                similar_products = []
                for _, train_product in self.products_df.iterrows():
                    similarity_score = 0
                    
                    # Category matching (30% weight)
                    if test_product['main_category'] == train_product['main_category']:
                        similarity_score += 0.3
                    
                    # Title similarity (40% weight)
                    title_sim = self._text_similarity(test_product['title'], train_product['title'])
                    similarity_score += 0.4 * title_sim
                    
                    # Features similarity (30% weight)
                    test_features = ' '.join(test_product.get('features', []))
                    train_features = ' '.join(train_product.get('features', []))
                    features_sim = self._text_similarity(test_features, train_features)
                    similarity_score += 0.3 * features_sim
                    
                    similar_products.append((similarity_score, train_product))
                
                # Sort by similarity score
                similar_products.sort(key=lambda x: x[0], reverse=True)
                rec_products = [p[1] for p in similar_products[:k]]
                rec_titles = [p['title'] for p in rec_products]
                
                # Calculate relevance based on multiple criteria
                relevant_items = set()
                for train_product in self.products_df.itertuples():
                    if (test_product['main_category'] == train_product.main_category and
                        self._text_similarity(test_product['title'], train_product.title) > 0.3):
                        relevant_items.add(train_product.title)
                
                # Metrics calculation
                hits += int(bool(set(rec_titles) & relevant_items))
                precision = len(set(rec_titles) & relevant_items) / k
                recall = len(set(rec_titles) & relevant_items) / len(relevant_items) if relevant_items else 0
                
                # NDCG calculation
                dcg = sum(1 / np.log2(i + 2) for i, title in enumerate(rec_titles) if title in relevant_items)
                idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
                ndcg = dcg / idcg if idcg > 0 else 0
                
                precision_sum += precision
                recall_sum += recall
                ndcg_sum += ndcg
            
            # Store metrics
            metrics[f'hit_ratio@{k}'] = hits / total
            metrics[f'precision@{k}'] = precision_sum / total
            metrics[f'recall@{k}'] = recall_sum / total
            metrics[f'ndcg@{k}'] = ndcg_sum / total
        
        return metrics

    def _text_similarity(self, text1, text2):
        """Calculate text similarity using TF-IDF and cosine similarity"""
        if not text1 or not text2:
            return 0
        
        # Use stored vectorizer or create a new one for single comparison
        if not hasattr(self, 'text_vectorizer'):
            self.text_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=1000
            )
        
        try:
            vectors = self.text_vectorizer.fit_transform([str(text1), str(text2)])
            return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        except:
            return 0

    def _calculate_criteria_weights(self, decision_matrix):
        """Calculate criteria weights using CRITIC method"""
        # Calculate CRITIC weights
        critic = pymcdm.weights.critic_weights(decision_matrix)
        return critic

