import json
import argparse
import pandas as pd
import random
import numpy as np
from Recommender import MemoryEfficientProductRecommender

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_products(product_path, sample_size=None, random_seed=42):
    """
    Load product data from JSON or JSONL file with option to sample
    
    Args:
        product_path: Path to the JSON or JSONL file
        sample_size: Number of products to sample (None = all products)
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    
    if product_path.endswith('.jsonl'):
        # For JSONL files, we can read line by line without loading the entire file
        products = []
        with open(product_path, 'r', encoding='utf-8') as f:
            if sample_size is None:
                # Load all products (not recommended for large files)
                products = [json.loads(line) for line in f if line.strip()]
            else:
                # Reservoir sampling to get a random sample without loading all data
                for i, line in enumerate(f):
                    if line.strip():
                        if len(products) < sample_size:
                            products.append(json.loads(line))
                        else:
                            # Replace elements with decreasing probability
                            r = random.randint(0, i)
                            if r < sample_size:
                                products[r] = json.loads(line)
    else:
        # For regular JSON files
        with open(product_path, 'r', encoding='utf-8') as f:
            all_products = json.load(f)
            if sample_size is None or sample_size >= len(all_products):
                products = all_products
            else:
                products = random.sample(all_products, sample_size)
    
    return products

def preprocess_and_split_data(raw_products, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    import numpy as np
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)

    def preprocess_product(prod):
        desc_list = prod.get("description", [""])
        desc = desc_list[0] if isinstance(desc_list, list) and len(desc_list) > 0 else ""
        return {
            "main_category": prod.get("main_category", ""),
            "title": prod.get("title", ""),
            "average_rating": prod.get("average_rating", 3.0),
            "rating_number": prod.get("rating_number", 1),
            "features": prod.get("features", [])[:3],
            "description": desc[:200],
            "categories": prod.get("categories", [])[:3],
            "material": prod.get("details", {}).get("Material", ""),
            "compatible_models": prod.get("details", {}).get("Compatible Phone Models", ""),
            "product_dimensions": prod.get("details", {}).get("Product Dimensions", ""),
            "brand": prod.get("details", {}).get("Brand", ""),
        }

    processed = [preprocess_product(p) for p in raw_products]
    indices = np.arange(len(processed))
    np.random.shuffle(indices)
    n_total = len(processed)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    train_list = [processed[i] for i in train_idx]
    val_list = [processed[i] for i in val_idx]
    test_list = [processed[i] for i in test_idx]

    print(f"Split: {len(train_list)} train, {len(val_list)} val, {len(test_list)} test")
    return train_list, val_list, test_list

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Enhanced Product Recommender')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--products', type=str, help='Path to product data JSON file')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--query-product', type=int, default=0, help='Index of product to get recommendations for')
    parser.add_argument('--features', type=str, nargs='+', help='Features to get recommendations for')
    parser.add_argument('--sample-size', type=int, default=10000, help='Number of products to sample (default=10000, 0=all)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override CUDA setting if specified
    if args.no_cuda:
        config['model_params']['use_cuda'] = False
    
    print("Starting Enhanced Product Recommender (Optimized)")
    print("------------------------------------------------")
    
    # Set sample size (0 means use all data, not recommended for large files)
    sample_size = None if args.sample_size == 0 else args.sample_size
    
    # Load product data
    if args.products:
        print(f"Loading products from {args.products}")
        if sample_size:
            print(f"Sampling {sample_size} products from the dataset")
        raw_products = load_products(args.products, sample_size=sample_size)
    else:
        print("Provide dataset path to load products")
    
    # Preprocess and split
    train_data, val_data, test_data = preprocess_and_split_data(raw_products)

    # Instantiate the recommender with config parameters
    recommender = MemoryEfficientProductRecommender(config=config)

    # Use only train_data for fitting the recommender
    print("\nProcessing product data (train set only)...")
    processed_df = recommender.preprocess_data(train_data)
    print(f"Processed {len(processed_df)} products (train set)")

    # Train the model on train set
    print("\nTraining the model...")
    try:
        recommender.train_model()
        print("Model training completed successfully!")

        # Evaluate on test set
        if test_data:
            print("\nEvaluation Metrics:")
            print("-" * 40)
            metrics = recommender.evaluate_metrics(test_data, top_ks=[5, 10, 20])
            
            # Print metrics in a clean format
            for k in [5, 10, 20]:
                print(f"\nTop-{k} Metrics:")
                print(f"Hit Ratio:  {metrics[f'hit_ratio@{k}']:.4f}")
                print(f"Precision:  {metrics[f'precision@{k}']:.4f}")
                print(f"Recall:     {metrics[f'recall@{k}']:.4f}")
                print(f"NDCG:       {metrics[f'ndcg@{k}']:.4f}")
            
            print("-" * 40)

        # Get recommendation parameters with defaults
        rec_params = config.get('recommendation_params', {'top_n': 5})
        top_n = rec_params.get('top_n', 5)
        
        # Only show recommendations if specifically requested
        if args.query_product is not None and args.query_product < len(raw_products):
            print(f"\nTop {top_n} Recommendations:")
            recommendations = recommender.get_recommendations(
                args.query_product, 
                top_n=top_n
            )
            for _, row in recommendations.iterrows():
                print(f"- {row['title']}")
        
        # Get recommendations by features if specified
        if args.features:
            print(f"\nRecommendations for features {args.features}:")
            recommendations = recommender.get_recommendations_by_features(args.features, top_n=config['recommendation_params']['top_n'])
            for idx, row in recommendations.iterrows():
                print(f"- {row['title']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nRecommendation process completed!")

if __name__ == "__main__":
    main()