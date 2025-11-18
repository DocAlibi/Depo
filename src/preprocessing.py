"""
Data preprocessing module for clustering analysis.
Handles data cleaning, filtering, and outlier removal.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
from config.settings import ANALYSIS_CONFIG

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for different client segments."""

    def __init__(self):
        """Initialize preprocessor with configuration."""
        self.outlier_lower = ANALYSIS_CONFIG['outlier_percentile_lower']
        self.outlier_upper = ANALYSIS_CONFIG['outlier_percentile_upper']
        self.new_client_days = ANALYSIS_CONFIG['new_client_days']

    def preprocess_segment(
        self,
        df: pd.DataFrame,
        business_type: Optional[pd.DataFrame],
        segment_type: str
    ) -> pd.DataFrame:
        """
        Preprocess data for a specific segment.

        Args:
            df: Transaction DataFrame
            business_type: Business metadata DataFrame (optional for SOC/CASH)
            segment_type: Type of segment (MB_IP, MB_TOO, SB, NEW_CLIENTS, SOC, CASH)

        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing segment: {segment_type}")

        if segment_type in ['MB_IP', 'MB_TOO', 'SB', 'NEW_CLIENTS']:
            result = self._preprocess_business_segment(df, business_type, segment_type)
        elif segment_type in ['SOC', 'CASH']:
            result = self._preprocess_special_segment(df)
        else:
            logger.warning(f"Unknown segment type: {segment_type}")
            return pd.DataFrame()

        logger.info(f"  Transactions: {len(result):,}")
        logger.info(f"  Unique clients: {result['BINIIN'].nunique():,}")

        return result

    def _preprocess_business_segment(
        self,
        df: pd.DataFrame,
        business_type: pd.DataFrame,
        segment_type: str
    ) -> pd.DataFrame:
        """Preprocess business segments (MB_IP, MB_TOO, SB, NEW_CLIENTS)."""
        # Filter metadata
        business_filtered = business_type[
            business_type.LEGAL_FORM_ABBR.isin(['ИП', 'ТОО']) &
            business_type.BUSINESS_TYPE.isin(['МБ', 'СБ'])
        ].copy()

        business_filtered['REG_DATE'] = pd.to_datetime(
            business_filtered['REG_DATE'],
            errors='coerce'
        )

        # Merge with transactions
        result = df.merge(business_filtered, on='BINIIN', how='left')
        result = result[result.AMOUNT_DT > 0].copy()

        # Apply segment-specific filters
        if segment_type == 'MB_IP':
            result = result[
                (result.BUSINESS_TYPE == 'МБ') &
                (result.INDUSTRY != 'Сельское хозяйство') &
                (result.LEGAL_FORM_ABBR == 'ИП') &
                (result.OPER_GROUP != 'soc')
            ].copy()

        elif segment_type == 'MB_TOO':
            result = result[
                (result.BUSINESS_TYPE == 'МБ') &
                (result.INDUSTRY != 'Сельское хозяйство') &
                (result.LEGAL_FORM_ABBR == 'ТОО') &
                (result.OPER_GROUP != 'soc')
            ].copy()

        elif segment_type == 'SB':
            result = result[
                (result.BUSINESS_TYPE == 'СБ') &
                (result.INDUSTRY != 'Сельское хозяйство') &
                (result.OPER_GROUP != 'soc')
            ].copy()

        elif segment_type == 'NEW_CLIENTS':
            threshold_date = datetime.now() - timedelta(days=self.new_client_days)
            result = result[
                (result.REG_DATE >= threshold_date) &
                (result.OPER_GROUP != 'soc')
            ].copy()

        return result

    def _preprocess_special_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess special segments (SOC, CASH)."""
        return df[df.AMOUNT_DT > 0].copy()

    def clean_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using percentile method.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        logger.info("  Cleaning outliers...")
        initial_count = len(df)

        # Remove outliers by transaction amount
        amount_lower = df['AMOUNT_DT'].quantile(self.outlier_lower)
        amount_upper = df['AMOUNT_DT'].quantile(self.outlier_upper)

        df_clean = df[
            (df['AMOUNT_DT'] >= amount_lower) &
            (df['AMOUNT_DT'] <= amount_upper)
        ].copy()

        # Remove outliers by transaction count per client
        transaction_counts = df_clean.groupby('BINIIN').size().reset_index(name='trans_count')
        count_lower = transaction_counts['trans_count'].quantile(self.outlier_lower)
        count_upper = transaction_counts['trans_count'].quantile(self.outlier_upper)

        clients_to_keep = transaction_counts[
            (transaction_counts['trans_count'] >= count_lower) &
            (transaction_counts['trans_count'] <= count_upper)
        ]['BINIIN']

        df_clean = df_clean[df_clean['BINIIN'].isin(clients_to_keep)].copy()

        retention_pct = len(df_clean) / initial_count * 100
        logger.info(f"  Retained: {len(df_clean):,} ({retention_pct:.1f}%)")

        return df_clean

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to DataFrame.

        Args:
            df: DataFrame to enhance

        Returns:
            DataFrame with time features
        """
        df = df.copy()
        df['DATE_BY_MONTH'] = pd.to_datetime(df['DATE_BY_MONTH'])
        df['MONTH'] = df['DATE_BY_MONTH'].dt.to_period('M')
        df['YEAR'] = df['DATE_BY_MONTH'].dt.year
        df['QUARTER'] = df['DATE_BY_MONTH'].dt.quarter
        df['DAY_OF_WEEK'] = df['DATE_BY_MONTH'].dt.dayofweek

        return df


class SegmentFilter:
    """Filters data for specific segments."""

    @staticmethod
    def filter_by_currency(df: pd.DataFrame, currency: str = 'KZT') -> pd.DataFrame:
        """
        Filter transactions by currency.

        Args:
            df: DataFrame to filter
            currency: Currency code

        Returns:
            Filtered DataFrame
        """
        logger.info(f"Filtering by currency: {currency}")
        df_filtered = df[df['CURRENCY'] == currency].copy()
        logger.info(f"  Transactions ({currency}): {len(df_filtered):,}")
        logger.info(f"  Unique clients: {df_filtered['BINIIN'].nunique():,}")

        return df_filtered

    @staticmethod
    def validate_data_quality(df: pd.DataFrame, min_records: int = 100) -> bool:
        """
        Validate if dataset has sufficient quality for analysis.

        Args:
            df: DataFrame to validate
            min_records: Minimum number of records required

        Returns:
            True if data quality is sufficient
        """
        if len(df) < min_records:
            logger.warning(f"Insufficient data: {len(df)} records (minimum: {min_records})")
            return False

        # Check for required columns
        required_cols = ['BINIIN', 'DATE_BY_MONTH', 'AMOUNT_DT']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False

        # Check for null values in critical columns
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")

        return True
