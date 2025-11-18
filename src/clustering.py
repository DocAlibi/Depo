"""
Clustering module for FM (Frequency-Monetary) analysis.
Implements quantile-based clustering and migration tracking.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from config.settings import ANALYSIS_CONFIG, CLUSTER_NAMES

logger = logging.getLogger(__name__)


class FMClustering:
    """Frequency-Monetary clustering implementation."""

    def __init__(self, quantiles: Optional[List[float]] = None, use_log: bool = True):
        """
        Initialize FM clustering.

        Args:
            quantiles: Quantile thresholds for clustering
            use_log: Whether to use log transformation
        """
        self.quantiles = quantiles or ANALYSIS_CONFIG['cluster_quantiles']
        self.use_log = use_log
        self.cluster_boundaries = None

    def calculate_fm_metrics(
        self,
        df: pd.DataFrame,
        group_cols: List[str] = ['BINIIN']
    ) -> pd.DataFrame:
        """
        Calculate Frequency and Monetary metrics.

        Args:
            df: Transaction DataFrame
            group_cols: Columns to group by

        Returns:
            DataFrame with FM metrics
        """
        logger.debug(f"Calculating FM metrics grouped by {group_cols}")

        # Ensure time features exist
        if 'MONTH' not in df.columns and 'DATE_BY_MONTH' in df.columns:
            df['MONTH'] = pd.to_datetime(df['DATE_BY_MONTH']).dt.to_period('M')

        # Calculate aggregations
        fm_data = df.groupby(group_cols).agg({
            'AMOUNT_DT': ['sum', 'mean'],
            'DATE_BY_MONTH': 'count'
        }).reset_index()

        # Flatten column names
        fm_data.columns = ['_'.join(col).strip() if col[1] else col[0]
                          for col in fm_data.columns.values]

        # Rename to standard names
        fm_data.rename(columns={
            'AMOUNT_DT_sum': 'Monetary',
            'AMOUNT_DT_mean': 'Avg_Transaction',
            'DATE_BY_MONTH_count': 'Frequency'
        }, inplace=True)

        # Apply log transformation if enabled
        if self.use_log:
            fm_data['Log_Frequency'] = np.log1p(fm_data['Frequency'])
            fm_data['Log_Monetary'] = np.log1p(fm_data['Monetary'])

        logger.debug(f"FM metrics calculated for {len(fm_data):,} entities")
        return fm_data

    def create_clusters(self, fm_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Create clusters using quantile-based method.

        Args:
            fm_data: DataFrame with FM metrics

        Returns:
            Tuple of (clustered DataFrame, cluster boundaries)
        """
        logger.info("Creating clusters using quantile method...")

        # Determine which columns to use
        freq_col = 'Log_Frequency' if self.use_log and 'Log_Frequency' in fm_data.columns else 'Frequency'
        mon_col = 'Log_Monetary' if self.use_log and 'Log_Monetary' in fm_data.columns else 'Monetary'

        # Calculate quantiles
        freq_quantiles = fm_data[freq_col].quantile(self.quantiles)
        mon_quantiles = fm_data[mon_col].quantile(self.quantiles)

        # Define cluster conditions (vectorized)
        conditions = [
            # VIP: High frequency AND high monetary
            (fm_data[freq_col] >= freq_quantiles[0.75]) & (fm_data[mon_col] >= mon_quantiles[0.75]),
            # Loyal: Medium-high frequency AND monetary
            (fm_data[freq_col] >= freq_quantiles[0.5]) & (fm_data[mon_col] >= mon_quantiles[0.5]),
            # Sleeping: Low frequency OR low monetary
            (fm_data[freq_col] < freq_quantiles[0.25]) | (fm_data[mon_col] < mon_quantiles[0.25])
        ]

        choices = [0, 1, 3]  # VIP, Loyal, Sleeping
        fm_data['Cluster'] = np.select(conditions, choices, default=2)  # Default: Regular

        # Store boundaries
        self.cluster_boundaries = {
            'freq_quantiles': freq_quantiles.to_dict(),
            'mon_quantiles': mon_quantiles.to_dict(),
            'freq_col': freq_col,
            'mon_col': mon_col
        }

        # Log cluster distribution
        cluster_counts = fm_data['Cluster'].value_counts().sort_index()
        logger.info("Cluster distribution:")
        for cluster_id, count in cluster_counts.items():
            cluster_name = CLUSTER_NAMES.get(cluster_id, f'Cluster_{cluster_id}')
            pct = count / len(fm_data) * 100
            logger.info(f"  {cluster_name:12}: {count:6,} ({pct:5.1f}%)")

        return fm_data, self.cluster_boundaries

    def apply_clusters_to_monthly(
        self,
        df: pd.DataFrame,
        cluster_boundaries: Dict
    ) -> pd.DataFrame:
        """
        Apply cluster boundaries to monthly data.

        Args:
            df: Transaction DataFrame with MONTH column
            cluster_boundaries: Cluster boundaries from create_clusters

        Returns:
            DataFrame with monthly clusters
        """
        logger.info("Applying clusters to monthly data...")

        freq_col = cluster_boundaries['freq_col']
        mon_col = cluster_boundaries['mon_col']
        freq_q = cluster_boundaries['freq_quantiles']
        mon_q = cluster_boundaries['mon_quantiles']

        monthly_clusters = []
        months = sorted(df['MONTH'].unique())

        for month in months:
            month_data = df[df['MONTH'] == month]
            fm_month = self.calculate_fm_metrics(month_data)

            # Apply same clustering rules
            conditions = [
                (fm_month[freq_col] >= freq_q[0.75]) & (fm_month[mon_col] >= mon_q[0.75]),
                (fm_month[freq_col] >= freq_q[0.5]) & (fm_month[mon_col] >= mon_q[0.5]),
                (fm_month[freq_col] < freq_q[0.25]) | (fm_month[mon_col] < mon_q[0.25])
            ]
            choices = [0, 1, 3]
            fm_month['Cluster'] = np.select(conditions, choices, default=2)
            fm_month['Month'] = month

            monthly_clusters.append(fm_month)

        result = pd.concat(monthly_clusters, ignore_index=True)
        logger.info(f"Monthly clusters created for {len(months)} months")

        return result


class MigrationAnalyzer:
    """Analyzes customer migrations between clusters."""

    def __init__(self):
        """Initialize migration analyzer."""
        self.transitions = []

    def analyze_transitions(self, monthly_clusters: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze cluster transitions month-over-month.

        Args:
            monthly_clusters: DataFrame with monthly cluster assignments

        Returns:
            DataFrame with transition information
        """
        logger.info("Analyzing cluster transitions...")

        transitions = []
        months = sorted(monthly_clusters['Month'].unique())

        for i in range(len(months) - 1):
            current_month = months[i]
            next_month = months[i + 1]

            # Get cluster assignments for consecutive months
            current_data = monthly_clusters[monthly_clusters['Month'] == current_month][
                ['BINIIN', 'Cluster']
            ]
            next_data = monthly_clusters[monthly_clusters['Month'] == next_month][
                ['BINIIN', 'Cluster']
            ]

            # Merge to find transitions
            merged = current_data.merge(
                next_data,
                on='BINIIN',
                suffixes=('_from', '_to')
            )
            merged['Transition'] = (
                current_month.strftime('%Y-%m') + ' → ' + next_month.strftime('%Y-%m')
            )

            transitions.append(merged)

        if transitions:
            all_transitions = pd.concat(transitions, ignore_index=True)
            logger.info(f"Analyzed {len(all_transitions):,} transitions")
            self.transitions = all_transitions
            return all_transitions
        else:
            logger.warning("No transitions found")
            return pd.DataFrame()

    def calculate_transition_matrix(self, transitions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate transition probability matrix.

        Args:
            transitions: DataFrame with transition data

        Returns:
            Transition probability matrix
        """
        if transitions.empty:
            return pd.DataFrame()

        # Calculate transition probabilities
        transition_matrix = pd.crosstab(
            transitions['Cluster_from'],
            transitions['Cluster_to'],
            normalize='index'
        ) * 100

        # Rename indices and columns
        transition_matrix.index = [
            CLUSTER_NAMES.get(idx, f'K{idx}') for idx in transition_matrix.index
        ]
        transition_matrix.columns = [
            CLUSTER_NAMES.get(col, f'K{col}') for col in transition_matrix.columns
        ]

        return transition_matrix

    def calculate_retention_rates(self, transition_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate retention rates for each cluster.

        Args:
            transition_matrix: Transition probability matrix

        Returns:
            Dictionary of retention rates
        """
        retention = {}

        for cluster in transition_matrix.index:
            if cluster in transition_matrix.columns:
                retention[cluster] = transition_matrix.loc[cluster, cluster]

        logger.info("Retention rates:")
        for cluster, rate in retention.items():
            logger.info(f"  {cluster:12}: {rate:.1f}%")

        return retention

    def identify_churn_customers(
        self,
        transitions: pd.DataFrame,
        definition: str = 'downgrade'
    ) -> pd.DataFrame:
        """
        Identify customers who churned based on cluster transitions.

        Args:
            transitions: DataFrame with transition data
            definition: 'downgrade' (moved to lower cluster) or 'exit' (left entirely)

        Returns:
            DataFrame with churn customers
        """
        logger.info(f"Identifying churn customers (definition: {definition})...")

        if definition == 'downgrade':
            # Churn = moved to a lower cluster (higher cluster number)
            churn_df = transitions[transitions['Cluster_to'] > transitions['Cluster_from']].copy()
        elif definition == 'exit':
            # This would require additional data about customers who stopped transacting
            logger.warning("Exit definition requires additional implementation")
            churn_df = pd.DataFrame()
        else:
            raise ValueError(f"Unknown churn definition: {definition}")

        churn_df['Churn'] = 1
        logger.info(f"Identified {len(churn_df):,} churn cases")

        return churn_df
