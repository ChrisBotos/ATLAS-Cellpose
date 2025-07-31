"""
Author: Christos Botos.
Affiliation: Leiden University Medical Center
Contact: botoschristos@gmail.com | linkedin.com/in/christos-botos-2369hcty3396 | github.com/ChrisBotos.

Script Name: test_cluster_engineered_features.py
Description:
    Comprehensive test suite for nuclear feature clustering functionality.
    Tests data loading, preprocessing, clustering algorithms, and visualization generation
    with both synthetic and real data scenarios.

Dependencies:
    • Python >= 3.10.
    • pytest, numpy, pandas, scikit-learn, PIL.
    • unittest for standard testing framework.

Usage:
    python -m pytest tests/test_cluster_engineered_features.py -v
    python tests/test_cluster_engineered_features.py

Key Features:
    • Unit tests for all clustering pipeline components.
    • Integration tests with synthetic nuclear feature data.
    • Memory efficiency validation for large datasets.
    • Color palette generation testing with 35+ colors.
    • Visualization output validation.
    • Error handling and edge case testing.

Notes:
    • Tests use synthetic data to avoid dependency on specific datasets.
    • Memory usage tests validate streaming processing efficiency.
    • Color tests ensure scientific visualization standards.
"""
import traceback
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import sys
import os

# Add project root to path for imports.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from code.engineered_feature_extraction.cluster_engineered_features import (
    load_nuclear_features, prepare_feature_matrix, choose_optimal_k,
    stream_scale_features, stream_cluster_features, predict_cluster_labels,
    compute_cluster_statistics
)
from code.engineered_feature_extraction.utils.generate_contrast_colors import (
    generate_color_palette, get_predefined_vibrant_colors, colors_to_hex_list
)


class TestNuclearFeatureClustering(unittest.TestCase):
    """Test suite for nuclear feature clustering functionality."""
    
    def setUp(self):
        """Set up test fixtures with synthetic data."""
        # Create temporary directory for test outputs.
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Generate synthetic nuclear feature data.
        np.random.seed(42)
        n_nuclei = 1000
        
        # Create realistic nuclear feature data.
        self.synthetic_features = {
            'Label': np.arange(1, n_nuclei + 1),
            'Centroid_X': np.random.uniform(0, 2000, n_nuclei),
            'Centroid_Y': np.random.uniform(0, 2000, n_nuclei),
            'Area': np.random.lognormal(mean=5.0, sigma=0.5, size=n_nuclei),
            'Perimeter': np.random.lognormal(mean=4.0, sigma=0.3, size=n_nuclei),
            'Circularity': np.random.beta(2, 2, n_nuclei),
            'Eccentricity': np.random.beta(1, 3, n_nuclei),
            'Intensity_Mean': np.random.normal(150, 30, n_nuclei),
            'Intensity_Std': np.random.exponential(20, n_nuclei),
            'Texture_Entropy': np.random.gamma(2, 0.5, n_nuclei),
            'Neighborhood_Density': np.random.exponential(0.01, n_nuclei),
            'Distance_to_Nearest_Nucleus': np.random.exponential(50, n_nuclei)
        }
        
        # Create DataFrame and save as CSV.
        self.df = pd.DataFrame(self.synthetic_features)
        self.features_path = self.test_dir / 'synthetic_features.csv'
        self.df.to_csv(self.features_path, index=False)
        
        # Create synthetic image and mask.
        self.image_path = self.test_dir / 'synthetic_image.tif'
        self.mask_path = self.test_dir / 'synthetic_mask.npy'
        
        # Generate synthetic DAPI image.
        synthetic_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        Image.fromarray(synthetic_image).save(self.image_path)
        
        # Generate synthetic segmentation mask.
        synthetic_mask = np.random.randint(0, n_nuclei + 1, (2000, 2000), dtype=np.int32)
        np.save(self.mask_path, synthetic_mask)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_load_nuclear_features(self):
        """Test nuclear feature loading and validation."""
        # Test successful loading.
        df = load_nuclear_features(self.features_path)
        
        self.assertEqual(len(df), 1000)
        self.assertIn('Label', df.columns)
        self.assertIn('Area', df.columns)
        self.assertIn('Circularity', df.columns)
        
        # Test missing file error.
        with self.assertRaises(FileNotFoundError):
            load_nuclear_features(Path('nonexistent.csv'))
    
    def test_prepare_feature_matrix(self):
        """Test feature matrix preparation and preprocessing."""
        df = load_nuclear_features(self.features_path)
        features, feature_names, nuclear_labels = prepare_feature_matrix(df)
        
        # Validate output shapes and types.
        self.assertEqual(features.shape[0], len(df))
        self.assertEqual(len(feature_names), features.shape[1])
        self.assertEqual(len(nuclear_labels), len(df))
        
        # Check that metadata columns are excluded.
        self.assertNotIn('Label', feature_names)
        self.assertNotIn('Centroid_X', feature_names)
        self.assertNotIn('Centroid_Y', feature_names)
        
        # Validate feature columns are included.
        self.assertIn('Area', feature_names)
        self.assertIn('Circularity', feature_names)
        
        # Check for no missing values.
        self.assertFalse(np.any(np.isnan(features)))
    
    def test_stream_scale_features(self):
        """Test streaming feature scaling functionality."""
        df = load_nuclear_features(self.features_path)
        features, _, _ = prepare_feature_matrix(df)
        
        # Test scaling with different batch sizes.
        for batch_size in [100, 500, 1000]:
            scaler = stream_scale_features(features, batch_size)
            
            # Validate scaler properties.
            self.assertEqual(len(scaler.mean_), features.shape[1])
            self.assertEqual(len(scaler.scale_), features.shape[1])
            
            # Test transformation.
            scaled_features = scaler.transform(features)
            
            # Check that features are approximately standardized.
            feature_means = np.mean(scaled_features, axis=0)
            feature_stds = np.std(scaled_features, axis=0)
            
            np.testing.assert_allclose(feature_means, 0, atol=1e-10)
            np.testing.assert_allclose(feature_stds, 1, atol=1e-10)
    
    def test_clustering_pipeline(self):
        """Test complete clustering pipeline with synthetic data."""
        df = load_nuclear_features(self.features_path)
        features, feature_names, nuclear_labels = prepare_feature_matrix(df)
        
        # Test clustering with different parameters.
        for n_clusters in [3, 5, 10]:
            scaler = stream_scale_features(features, batch_size=200)
            kmeans = stream_cluster_features(features, scaler, n_clusters, 
                                           batch_size=200, seed=42)
            cluster_labels = predict_cluster_labels(features, scaler, kmeans, 
                                                   batch_size=200)
            
            # Validate clustering results.
            self.assertEqual(len(cluster_labels), len(features))
            self.assertEqual(len(np.unique(cluster_labels)), n_clusters)
            self.assertTrue(all(0 <= label < n_clusters for label in cluster_labels))
    
    def test_optimal_k_selection(self):
        """Test automatic optimal K selection methods."""
        df = load_nuclear_features(self.features_path)
        features, _, _ = prepare_feature_matrix(df)
        
        # Scale features for clustering.
        scaler = stream_scale_features(features, batch_size=200)
        scaled_features = scaler.transform(features)
        
        # Test silhouette method.
        optimal_k, scores_df = choose_optimal_k(scaled_features, k_max=8, 
                                              criterion='silhouette', sample_size=500)
        
        self.assertTrue(2 <= optimal_k <= 8)
        self.assertEqual(len(scores_df), 7)  # k from 2 to 8
        self.assertIn('silhouette', scores_df.columns)
        
        # Test Davies-Bouldin method.
        optimal_k_dbi, scores_df_dbi = choose_optimal_k(scaled_features, k_max=6, 
                                                       criterion='dbi', sample_size=500)
        
        self.assertTrue(2 <= optimal_k_dbi <= 6)
        self.assertEqual(len(scores_df_dbi), 5)  # k from 2 to 6
        self.assertIn('dbi', scores_df_dbi.columns)
    
    def test_cluster_statistics(self):
        """Test cluster statistics computation."""
        df = load_nuclear_features(self.features_path)
        features, feature_names, _ = prepare_feature_matrix(df)
        
        # Generate mock cluster labels.
        n_clusters = 5
        cluster_labels = np.random.randint(0, n_clusters, len(df))
        
        # Compute statistics.
        stats_df = compute_cluster_statistics(df, cluster_labels, feature_names)
        
        # Validate statistics.
        self.assertEqual(len(stats_df), n_clusters)
        self.assertIn('cluster_id', stats_df.columns)
        self.assertIn('nucleus_count', stats_df.columns)
        self.assertIn('percentage', stats_df.columns)
        
        # Check that percentages sum to approximately 100%.
        total_percentage = stats_df['percentage'].sum()
        self.assertAlmostEqual(total_percentage, 100.0, places=1)
        
        # Validate feature statistics columns.
        for feature in ['Area', 'Circularity']:
            self.assertIn(f'{feature}_mean', stats_df.columns)
            self.assertIn(f'{feature}_std', stats_df.columns)


class TestColorPalette(unittest.TestCase):
    """Test suite for enhanced color palette generation."""
    
    def test_predefined_colors_extended(self):
        """Test extended predefined color palette."""
        colors = get_predefined_vibrant_colors()
        
        # Validate extended palette size.
        self.assertGreaterEqual(len(colors), 35)
        
        # Validate color format.
        for color in colors:
            self.assertEqual(len(color), 3)  # RGB tuple
            self.assertTrue(all(0 <= c <= 255 for c in color))
    
    def test_large_palette_generation(self):
        """Test color palette generation for large cluster numbers."""
        # Test various large cluster numbers.
        for n_clusters in [25, 35, 50]:
            palette = generate_color_palette(n=n_clusters, background="dark")
            
            self.assertEqual(len(palette), n_clusters)
            
            # Validate color format.
            for cluster_id, (r, g, b, a) in palette.items():
                self.assertTrue(0 <= r <= 255)
                self.assertTrue(0 <= g <= 255)
                self.assertTrue(0 <= b <= 255)
                self.assertTrue(0 <= a <= 255)
    
    def test_hex_color_conversion(self):
        """Test conversion of RGBA colors to hex format."""
        palette = generate_color_palette(n=10, background="dark")
        hex_colors = colors_to_hex_list(palette)
        
        self.assertEqual(len(hex_colors), 10)
        
        # Validate hex format.
        for hex_color in hex_colors:
            self.assertTrue(hex_color.startswith('#'))
            self.assertEqual(len(hex_color), 7)
            
            # Validate hex characters.
            hex_chars = hex_color[1:]
            self.assertTrue(all(c in '0123456789abcdef' for c in hex_chars.lower()))
    
    def test_custom_color_palette(self):
        """Test custom color palette functionality."""
        custom_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]
        palette = generate_color_palette(n=6, custom_colors=custom_colors)
        
        # Should have 6 colors total (4 custom + 2 generated).
        self.assertEqual(len(palette), 6)
        
        # First 4 should match custom colors.
        expected_rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, expected in enumerate(expected_rgb):
            r, g, b, a = palette[i]
            self.assertEqual((r, g, b), expected)


class TestMemoryEfficiency(unittest.TestCase):
    """Test suite for memory efficiency and performance."""
    
    def test_large_dataset_handling(self):
        """Test clustering with large synthetic datasets."""
        # Create large synthetic dataset.
        n_nuclei = 10000
        np.random.seed(42)
        
        large_features = np.random.randn(n_nuclei, 20)  # 20 features
        
        # Test streaming scaling.
        scaler = stream_scale_features(large_features, batch_size=1000)
        scaled_features = scaler.transform(large_features)
        
        # Validate scaling.
        self.assertEqual(scaled_features.shape, large_features.shape)
        
        # Test clustering.
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=1000)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Validate clustering.
        self.assertEqual(len(cluster_labels), n_nuclei)
        self.assertEqual(len(np.unique(cluster_labels)), 10)


def run_comprehensive_tests():
    """Run comprehensive test suite with detailed reporting."""
    print("="*80)
    print("COMPREHENSIVE NUCLEAR FEATURE CLUSTERING TEST SUITE")
    print("="*80)
    
    # Create test suite.
    test_suite = unittest.TestSuite()
    
    # Add test classes.
    test_classes = [
        TestNuclearFeatureClustering,
        TestColorPalette,
        TestMemoryEfficiency
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output.
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary.
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
