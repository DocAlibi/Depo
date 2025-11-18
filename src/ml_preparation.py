"""
ML Preparation module for churn prediction.
Handles feature engineering and target variable creation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from config.settings import ML_CONFIG, CLUSTER_NAMES

logger = logging.getLogger(__name__)


class ChurnTargetCreator:
    """Creates churn target variable from cluster transitions."""

    def __init__(self, churn_definition: str = 'downgrade'):
        """
        Initialize churn target creator.

        Args:
            churn_definition: 'downgrade' or 'exit'
        """
        self.churn_definition = churn_definition

    def create_churn_target(
        self,
        monthly_clusters: pd.DataFrame,
        prediction_horizon: int = 1
    ) -> pd.DataFrame:
        """
        Create churn target variable.

        Args:
            monthly_clusters: DataFrame with monthly cluster assignments
            prediction_horizon: Number of months ahead to predict (default: 1)

        Returns:
            DataFrame with BINIIN, Month, Cluster, and Churn target
        """
        logger.info(f"Creating churn target (definition: {self.churn_definition}, "
                   f"horizon: {prediction_horizon} month(s))...")

        churn_data = []
        months = sorted(monthly_clusters['Month'].unique())

        for i in range(len(months) - prediction_horizon):
            current_month = months[i]
            future_month = months[i + prediction_horizon]

            # Get current and future cluster assignments
            current = monthly_clusters[monthly_clusters['Month'] == current_month][
                ['BINIIN', 'Cluster', 'Frequency', 'Monetary']
            ].copy()
            future = monthly_clusters[monthly_clusters['Month'] == future_month][
                ['BINIIN', 'Cluster']
            ]

            # Merge
            merged = current.merge(
                future,
                on='BINIIN',
                suffixes=('_current', '_future'),
                how='left'
            )

            # Define churn
            if self.churn_definition == 'downgrade':
                # Churn = moved to worse cluster (higher number = worse)
                merged['Churn'] = (merged['Cluster_future'] > merged['Cluster_current']).astype(int)

            elif self.churn_definition == 'exit':
                # Churn = no longer in dataset next month
                merged['Churn'] = merged['Cluster_future'].isna().astype(int)

            # Add metadata
            merged['Month'] = current_month
            merged['Prediction_Horizon'] = prediction_horizon
            merged.rename(columns={'Cluster_current': 'Cluster'}, inplace=True)

            churn_data.append(merged[['BINIIN', 'Month', 'Cluster', 'Frequency', 'Monetary', 'Churn']])

        result = pd.concat(churn_data, ignore_index=True)

        churn_rate = result['Churn'].mean() * 100
        logger.info(f"Churn target created: {len(result):,} samples, "
                   f"churn rate: {churn_rate:.2f}%")

        return result


class FeatureEngineer:
    """Engineers features from transaction data for ML models."""

    def __init__(self, window_months: int = 3):
        """
        Initialize feature engineer.

        Args:
            window_months: Number of months for rolling features
        """
        self.window_months = window_months

    def create_transaction_features(
        self,
        transactions: pd.DataFrame,
        monthly_clusters: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create comprehensive features from transaction data.

        Args:
            transactions: Raw transaction DataFrame
            monthly_clusters: Monthly cluster assignments

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features from transaction data...")

        # Ensure time columns
        transactions['DATE_BY_MONTH'] = pd.to_datetime(transactions['DATE_BY_MONTH'])
        transactions['MONTH'] = transactions['DATE_BY_MONTH'].dt.to_period('M')

        features_list = []
        months = sorted(transactions['MONTH'].unique())

        for month in months:
            # Get data for rolling window
            window_end = month.to_timestamp()
            window_start = (month - self.window_months).to_timestamp()

            window_data = transactions[
                (transactions['DATE_BY_MONTH'] >= window_start) &
                (transactions['DATE_BY_MONTH'] < window_end)
            ]

            if len(window_data) == 0:
                continue

            # Aggregate features by client
            month_features = self._calculate_features(window_data)
            month_features['Month'] = month

            features_list.append(month_features)

        if not features_list:
            logger.warning("No features created")
            return pd.DataFrame()

        all_features = pd.concat(features_list, ignore_index=True)

        # Merge with cluster information
        result = all_features.merge(
            monthly_clusters[['BINIIN', 'Month', 'Cluster']],
            on=['BINIIN', 'Month'],
            how='inner'
        )

        logger.info(f"Created {len(result.columns)} features for {len(result):,} samples")

        return result

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregated features for each client."""

        # Basic transaction features
        basic_features = df.groupby('BINIIN').agg({
            'AMOUNT_DT': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'DATE_BY_MONTH': ['min', 'max']
        })

        # Flatten columns
        basic_features.columns = ['_'.join(col).strip() for col in basic_features.columns]
        basic_features.reset_index(inplace=True)

        # Rename for clarity
        basic_features.rename(columns={
            'AMOUNT_DT_sum': 'total_amount',
            'AMOUNT_DT_mean': 'avg_amount',
            'AMOUNT_DT_std': 'std_amount',
            'AMOUNT_DT_min': 'min_amount',
            'AMOUNT_DT_max': 'max_amount',
            'AMOUNT_DT_count': 'transaction_count',
            'DATE_BY_MONTH_min': 'first_transaction_date',
            'DATE_BY_MONTH_max': 'last_transaction_date'
        }, inplace=True)

        # Derived features
        basic_features['amount_range'] = (
            basic_features['max_amount'] - basic_features['min_amount']
        )
        basic_features['cv_amount'] = (
            basic_features['std_amount'] / (basic_features['avg_amount'] + 1)
        )

        # Time-based features
        basic_features['days_active'] = (
            (basic_features['last_transaction_date'] - basic_features['first_transaction_date']).dt.days
        )
        basic_features['avg_days_between_transactions'] = (
            basic_features['days_active'] / (basic_features['transaction_count'] + 1)
        )

        # Operation type features (if available)
        if 'OPER_GROUP' in df.columns:
            oper_features = pd.crosstab(df['BINIIN'], df['OPER_GROUP'])
            oper_features = oper_features.add_prefix('oper_count_')
            basic_features = basic_features.merge(
                oper_features.reset_index(),
                on='BINIIN',
                how='left'
            ).fillna(0)

        # Counterparty diversity
        if 'PAYER_BINIIN' in df.columns:
            payer_diversity = df.groupby('BINIIN')['PAYER_BINIIN'].nunique().reset_index()
            payer_diversity.columns = ['BINIIN', 'unique_payers']
            basic_features = basic_features.merge(payer_diversity, on='BINIIN', how='left')

        if 'RECIPIENT_BINIIN' in df.columns:
            recipient_diversity = df.groupby('BINIIN')['RECIPIENT_BINIIN'].nunique().reset_index()
            recipient_diversity.columns = ['BINIIN', 'unique_recipients']
            basic_features = basic_features.merge(recipient_diversity, on='BINIIN', how='left')

        # Currency diversity
        if 'CURRENCY' in df.columns:
            currency_diversity = df.groupby('BINIIN')['CURRENCY'].nunique().reset_index()
            currency_diversity.columns = ['BINIIN', 'currency_count']
            basic_features = basic_features.merge(currency_diversity, on='BINIIN', how='left')

        # Fill NaN with 0
        basic_features.fillna(0, inplace=True)

        return basic_features

    def create_trend_features(
        self,
        monthly_clusters: pd.DataFrame,
        lookback_months: int = 3
    ) -> pd.DataFrame:
        """
        Create trend features from monthly metrics.

        Args:
            monthly_clusters: Monthly cluster data
            lookback_months: Number of months to look back

        Returns:
            DataFrame with trend features
        """
        logger.info(f"Creating trend features (lookback: {lookback_months} months)...")

        trend_features = []
        months = sorted(monthly_clusters['Month'].unique())

        for i in range(lookback_months, len(months)):
            current_month = months[i]
            lookback_data = monthly_clusters[
                monthly_clusters['Month'].isin(months[i-lookback_months:i])
            ]

            # Calculate trends per client
            client_trends = lookback_data.groupby('BINIIN').agg({
                'Frequency': ['mean', 'std', 'min', 'max'],
                'Monetary': ['mean', 'std', 'min', 'max']
            })

            client_trends.columns = ['_'.join(col).strip() for col in client_trends.columns]
            client_trends.reset_index(inplace=True)

            # Add growth rates
            first_period = lookback_data[lookback_data['Month'] == months[i-lookback_months]]
            last_period = lookback_data[lookback_data['Month'] == months[i-1]]

            growth = first_period[['BINIIN', 'Frequency', 'Monetary']].merge(
                last_period[['BINIIN', 'Frequency', 'Monetary']],
                on='BINIIN',
                suffixes=('_first', '_last'),
                how='inner'
            )

            growth['frequency_growth'] = (
                (growth['Frequency_last'] - growth['Frequency_first']) /
                (growth['Frequency_first'] + 1)
            )
            growth['monetary_growth'] = (
                (growth['Monetary_last'] - growth['Monetary_first']) /
                (growth['Monetary_first'] + 1)
            )

            # Merge all trend features
            trends = client_trends.merge(
                growth[['BINIIN', 'frequency_growth', 'monetary_growth']],
                on='BINIIN',
                how='left'
            )
            trends['Month'] = current_month

            trend_features.append(trends)

        if trend_features:
            result = pd.concat(trend_features, ignore_index=True)
            logger.info(f"Created trend features for {len(result):,} samples")
            return result
        else:
            return pd.DataFrame()


class MLDatasetBuilder:
    """Builds final ML-ready dataset."""

    def __init__(self):
        """Initialize dataset builder."""
        self.feature_columns = None
        self.target_column = 'Churn'

    def build_dataset(
        self,
        churn_target: pd.DataFrame,
        transaction_features: pd.DataFrame,
        trend_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build complete ML dataset.

        Args:
            churn_target: DataFrame with churn targets
            transaction_features: DataFrame with transaction features
            trend_features: Optional DataFrame with trend features

        Returns:
            Complete ML-ready DataFrame
        """
        logger.info("Building ML dataset...")

        # Merge target with transaction features
        dataset = churn_target.merge(
            transaction_features,
            on=['BINIIN', 'Month'],
            how='inner'
        )

        # Merge trend features if available
        if trend_features is not None and not trend_features.empty:
            dataset = dataset.merge(
                trend_features,
                on=['BINIIN', 'Month'],
                how='left'
            )

        # Identify feature columns (exclude meta columns)
        meta_cols = ['BINIIN', 'Month', 'Churn', 'Cluster', 'first_transaction_date', 'last_transaction_date']
        self.feature_columns = [col for col in dataset.columns if col not in meta_cols]

        logger.info(f"Final dataset: {len(dataset):,} samples, {len(self.feature_columns)} features")
        logger.info(f"Churn rate: {dataset['Churn'].mean()*100:.2f}%")

        return dataset

    def split_train_test(
        self,
        dataset: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        temporal_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train and test sets.

        Args:
            dataset: Complete dataset
            test_size: Proportion of test set
            random_state: Random seed
            temporal_split: If True, use temporal split instead of random

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Splitting dataset (temporal_split={temporal_split})...")

        if temporal_split:
            # Use later months for testing
            months = sorted(dataset['Month'].unique())
            split_idx = int(len(months) * (1 - test_size))
            train_months = months[:split_idx]
            test_months = months[split_idx:]

            train_df = dataset[dataset['Month'].isin(train_months)].copy()
            test_df = dataset[dataset['Month'].isin(test_months)].copy()

        else:
            # Random split
            train_df, test_df = train_test_split(
                dataset,
                test_size=test_size,
                random_state=random_state,
                stratify=dataset['Churn']
            )

        logger.info(f"Train set: {len(train_df):,} samples (churn rate: {train_df['Churn'].mean()*100:.2f}%)")
        logger.info(f"Test set: {len(test_df):,} samples (churn rate: {test_df['Churn'].mean()*100:.2f}%)")

        return train_df, test_df

    def save_dataset(
        self,
        dataset: pd.DataFrame,
        output_path: str,
        include_meta: bool = False
    ):
        """
        Save dataset to file.

        Args:
            dataset: Dataset to save
            output_path: Output file path
            include_meta: Whether to include metadata columns
        """
        if not include_meta:
            # Save only features and target
            cols_to_save = self.feature_columns + [self.target_column]
            dataset[cols_to_save].to_csv(output_path, index=False)
        else:
            dataset.to_csv(output_path, index=False)

        logger.info(f"Dataset saved to {output_path}")
