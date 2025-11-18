"""
Main script for clustering and churn prediction analysis.
Orchestrates data loading, clustering, migration analysis, and ML preparation.
"""

import logging
import logging.config
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import (
    LOGGING_CONFIG, DATA_DIR, SEGMENT_TYPES, CLUSTER_NAMES, ML_CONFIG
)
from src.database import DatabaseConnector, DataLoader
from src.preprocessing import DataPreprocessor, SegmentFilter
from src.clustering import FMClustering, MigrationAnalyzer
from src.visualization import ClusterVisualizer
from src.ml_preparation import ChurnTargetCreator, FeatureEngineer, MLDatasetBuilder

warnings.filterwarnings('ignore')

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def print_banner():
    """Print application banner."""
    banner = """
╔════════════════════════════════════════════════════════════════╗
║   ЕДИНАЯ СИСТЕМА КЛАСТЕРИЗАЦИИ С АНАЛИЗОМ МИГРАЦИЙ            ║
║        Все сегменты + ML подготовка + Прогноз Churn            ║
╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info("Starting clustering and churn prediction analysis")


def analyze_segment(
    df: pd.DataFrame,
    segment_name: str,
    currency: str = 'KZT',
    enable_ml: bool = True
) -> Dict:
    """
    Perform complete analysis for a single segment.

    Args:
        df: Preprocessed DataFrame for segment
        segment_name: Name of the segment
        currency: Currency filter
        enable_ml: Whether to perform ML preparation

    Returns:
        Dictionary with analysis results
    """
    logger.info("="*80)
    logger.info(f"ANALYZING SEGMENT: {segment_name}")
    logger.info("="*80)

    # Filter by currency
    segment_filter = SegmentFilter()
    df_currency = segment_filter.filter_by_currency(df, currency)

    # Validate data quality
    if not segment_filter.validate_data_quality(df_currency):
        logger.warning(f"Insufficient data quality for {segment_name}")
        return None

    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_outliers(df_currency)
    df_clean = preprocessor.add_time_features(df_clean)

    # Clustering
    fm_clustering = FMClustering(use_log=True)

    # Annual clusters
    logger.info("Creating annual clusters...")
    fm_year = fm_clustering.calculate_fm_metrics(df_clean)
    fm_year, cluster_boundaries = fm_clustering.create_clusters(fm_year)

    # Monthly clusters
    logger.info("Creating monthly clusters...")
    monthly_clusters = fm_clustering.apply_clusters_to_monthly(df_clean, cluster_boundaries)

    # Migration analysis
    logger.info("Analyzing migrations...")
    migration_analyzer = MigrationAnalyzer()
    transitions = migration_analyzer.analyze_transitions(monthly_clusters)
    transition_matrix = migration_analyzer.calculate_transition_matrix(transitions)
    retention_rates = migration_analyzer.calculate_retention_rates(transition_matrix)

    # Visualizations
    logger.info("Creating visualizations...")
    visualizer = ClusterVisualizer()
    cluster_pivot = visualizer.plot_cluster_dynamics(monthly_clusters, segment_name)
    visualizer.plot_migration_analysis(transitions, segment_name)

    # Save cluster results
    fm_year.to_csv(DATA_DIR / f'clusters_{segment_name}.csv', index=False, encoding='utf-8-sig')
    monthly_clusters.to_csv(DATA_DIR / f'monthly_clusters_{segment_name}.csv', index=False, encoding='utf-8-sig')

    results = {
        'fm_year': fm_year,
        'monthly_clusters': monthly_clusters,
        'transitions': transitions,
        'transition_matrix': transition_matrix,
        'retention_rates': retention_rates,
        'cluster_pivot': cluster_pivot,
        'df_clean': df_clean
    }

    # ML Preparation
    if enable_ml and not transitions.empty:
        logger.info("="*80)
        logger.info(f"ML PREPARATION: {segment_name}")
        logger.info("="*80)

        # Create churn target
        churn_creator = ChurnTargetCreator(churn_definition=ML_CONFIG['churn_definition'])
        churn_target = churn_creator.create_churn_target(monthly_clusters, prediction_horizon=1)

        # Feature engineering
        feature_engineer = FeatureEngineer(window_months=ML_CONFIG['feature_window_months'])
        transaction_features = feature_engineer.create_transaction_features(df_clean, monthly_clusters)
        trend_features = feature_engineer.create_trend_features(monthly_clusters, lookback_months=3)

        # Build ML dataset
        ml_builder = MLDatasetBuilder()
        ml_dataset = ml_builder.build_dataset(churn_target, transaction_features, trend_features)

        # Split train/test
        train_df, test_df = ml_builder.split_train_test(
            ml_dataset,
            test_size=ML_CONFIG['test_size'],
            random_state=ML_CONFIG['random_state'],
            temporal_split=True
        )

        # Save ML datasets
        ml_builder.save_dataset(train_df, DATA_DIR / f'train_{segment_name}.csv', include_meta=False)
        ml_builder.save_dataset(test_df, DATA_DIR / f'test_{segment_name}.csv', include_meta=False)
        ml_dataset.to_csv(DATA_DIR / f'ml_dataset_{segment_name}.csv', index=False, encoding='utf-8-sig')

        # Visualize churn distribution
        visualizer.plot_churn_distribution(churn_target, segment_name)

        results.update({
            'churn_target': churn_target,
            'ml_dataset': ml_dataset,
            'train_df': train_df,
            'test_df': test_df,
            'feature_columns': ml_builder.feature_columns
        })

        logger.info(f"ML preparation completed for {segment_name}")
        logger.info(f"  Features: {len(ml_builder.feature_columns)}")
        logger.info(f"  Train samples: {len(train_df):,}")
        logger.info(f"  Test samples: {len(test_df):,}")

    return results


def main(
    currency: str = 'KZT',
    segments_to_analyze: Optional[List[str]] = None,
    enable_ml: bool = True
):
    """
    Main execution function.

    Args:
        currency: Currency to analyze
        segments_to_analyze: List of segments (None = all segments)
        enable_ml: Whether to perform ML preparation
    """
    print_banner()

    # Determine segments to analyze
    if segments_to_analyze is None:
        segments_to_analyze = SEGMENT_TYPES

    logger.info(f"Segments to analyze: {', '.join(segments_to_analyze)}")
    logger.info(f"Currency: {currency}")
    logger.info(f"ML preparation: {enable_ml}")

    # Load data from database
    logger.info("="*80)
    logger.info("LOADING DATA FROM DATABASE")
    logger.info("="*80)

    with DatabaseConnector() as db_connector:
        data_loader = DataLoader(db_connector)
        data = data_loader.load_all_data()

    # Preprocess and analyze each segment
    preprocessor = DataPreprocessor()
    all_results = {}

    for segment_name in segments_to_analyze:
        try:
            logger.info(f"\nProcessing segment: {segment_name}")

            # Preprocess segment
            if segment_name in ['MB_IP', 'MB_TOO', 'SB', 'NEW_CLIENTS']:
                df_segment = preprocessor.preprocess_segment(
                    data['main'],
                    data['business_type'],
                    segment_name
                )
            elif segment_name == 'SOC':
                df_segment = preprocessor.preprocess_segment(
                    data['soc'],
                    None,
                    segment_name
                )
            elif segment_name == 'CASH':
                df_segment = preprocessor.preprocess_segment(
                    data['cash'],
                    data['business_type'],
                    segment_name
                )
            else:
                logger.warning(f"Unknown segment: {segment_name}")
                continue

            if len(df_segment) == 0:
                logger.warning(f"No data for segment: {segment_name}")
                continue

            # Analyze segment
            results = analyze_segment(df_segment, segment_name, currency, enable_ml)

            if results:
                all_results[segment_name] = results

        except Exception as e:
            logger.error(f"Error analyzing {segment_name}: {e}", exc_info=True)
            continue

    # Generate summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    summary_data = []
    for segment_name, results in all_results.items():
        fm_data = results['fm_year']
        cluster_counts = fm_data['Cluster'].value_counts().sort_index()

        row = {
            'Сегмент': segment_name,
            'Всего': len(fm_data),
            'VIP': cluster_counts.get(0, 0),
            'Лояльные': cluster_counts.get(1, 0),
            'Обычные': cluster_counts.get(2, 0),
            'Спящие': cluster_counts.get(3, 0)
        }

        if enable_ml and 'churn_target' in results:
            churn_rate = results['churn_target']['Churn'].mean() * 100
            row['Churn Rate (%)'] = f"{churn_rate:.2f}"

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(DATA_DIR / 'summary_all_segments.csv', index=False, encoding='utf-8-sig')
    logger.info(f"\nSummary saved to {DATA_DIR / 'summary_all_segments.csv'}")

    # Final message
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"\nGenerated files in {DATA_DIR}:")
    logger.info("  • summary_all_segments.csv - Overall summary")
    logger.info("  • clusters_[SEGMENT].csv - Annual cluster assignments")
    logger.info("  • monthly_clusters_[SEGMENT].csv - Monthly cluster data")
    if enable_ml:
        logger.info("  • ml_dataset_[SEGMENT].csv - Complete ML dataset")
        logger.info("  • train_[SEGMENT].csv - Training set")
        logger.info("  • test_[SEGMENT].csv - Test set")
    logger.info("\nVisualizations:")
    logger.info("  • dynamics_[SEGMENT].png - Cluster dynamics charts")
    logger.info("  • migration_[SEGMENT].png - Migration analysis charts")
    if enable_ml:
        logger.info("  • churn_[SEGMENT].png - Churn distribution charts")

    return all_results, summary_df


if __name__ == "__main__":
    # Example 1: Analyze all segments with ML preparation
    all_results, summary = main(currency='KZT', enable_ml=True)

    # Example 2: Analyze only specific segments
    # all_results, summary = main(
    #     currency='KZT',
    #     segments_to_analyze=['MB_IP', 'NEW_CLIENTS'],
    #     enable_ml=True
    # )

    # Example 3: Only clustering without ML
    # all_results, summary = main(currency='KZT', enable_ml=False)

    # Access results:
    # - all_results['MB_IP']['ml_dataset'] - ML dataset for MB_IP
    # - all_results['MB_IP']['train_df'] - Training data
    # - all_results['MB_IP']['feature_columns'] - List of feature names
