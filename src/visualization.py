"""
Visualization module for clustering analysis.
Creates dynamic charts and migration flow diagrams.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from config.settings import VIZ_CONFIG, CLUSTER_NAMES, DATA_DIR

logger = logging.getLogger(__name__)

# Apply style settings
plt.style.use(VIZ_CONFIG['style'])
sns.set_palette(VIZ_CONFIG['palette'])


class ClusterVisualizer:
    """Visualizes cluster dynamics and migrations."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots (default: DATA_DIR)
        """
        self.output_dir = output_dir or DATA_DIR
        self.output_dir.mkdir(exist_ok=True)

    def plot_cluster_dynamics(
        self,
        monthly_clusters: pd.DataFrame,
        segment_name: str,
        save: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Create comprehensive cluster dynamics visualization.

        Args:
            monthly_clusters: DataFrame with monthly cluster data
            segment_name: Name of the segment
            save: Whether to save the plot

        Returns:
            Cluster pivot table
        """
        logger.info(f"Creating cluster dynamics visualization: {segment_name}")

        # Aggregate cluster counts by month
        cluster_counts = monthly_clusters.groupby(['Month', 'Cluster']).size().reset_index(name='Count')
        cluster_pivot = cluster_counts.pivot(index='Month', columns='Cluster', values='Count').fillna(0)

        # Rename columns
        cluster_pivot.columns = [
            CLUSTER_NAMES.get(col, f'Кластер {col}') for col in cluster_pivot.columns
        ]

        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=VIZ_CONFIG['figsize_dynamics'])
        fig.suptitle(f'Динамика кластеров: {segment_name}', fontsize=16, fontweight='bold')

        # 1. Absolute numbers trend
        ax1 = axes[0, 0]
        cluster_pivot.plot(kind='line', ax=ax1, marker='o', linewidth=2.5, markersize=8)
        ax1.set_title('Динамика численности кластеров', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Месяц')
        ax1.set_ylabel('Количество клиентов')
        ax1.grid(True, alpha=0.3)
        ax1.legend(title='Кластер', loc='best')

        # 2. Percentage distribution
        ax2 = axes[0, 1]
        cluster_pivot_pct = cluster_pivot.div(cluster_pivot.sum(axis=1), axis=0) * 100
        cluster_pivot_pct.plot(kind='area', stacked=True, ax=ax2, alpha=0.7)
        ax2.set_title('Доля кластеров по месяцам (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Месяц')
        ax2.set_ylabel('Доля (%)')
        ax2.legend(title='Кластер', loc='center left', bbox_to_anchor=(1, 0.5))

        # 3. Growth index
        ax3 = axes[1, 0]
        if len(cluster_pivot) > 0 and cluster_pivot.iloc[0].sum() > 0:
            base_index = (cluster_pivot / cluster_pivot.iloc[0]) * 100
            base_index.plot(kind='line', ax=ax3, marker='s', linewidth=2, markersize=6)
            ax3.axhline(y=100, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('Индекс роста (первый месяц = 100)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Месяц')
            ax3.set_ylabel('Индекс (%)')
            ax3.grid(True, alpha=0.3)
            ax3.legend(title='Кластер', loc='best')

        # 4. Month-over-month changes
        ax4 = axes[1, 1]
        monthly_change = cluster_pivot.diff()
        monthly_change.plot(kind='bar', ax=ax4, width=0.8)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title('Месячные изменения', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Месяц')
        ax4.set_ylabel('Изменение количества')
        ax4.legend(title='Кластер', loc='center left', bbox_to_anchor=(1, 0.5))
        ax4.tick_params(axis='x', rotation=45)

        # 5. Heatmap
        ax5 = axes[2, 0]
        heatmap_data = cluster_pivot.T
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Клиентов'})
        ax5.set_title('Тепловая карта активности', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Месяц')
        ax5.set_ylabel('Кластер')

        # 6. Sleeping customers rate
        ax6 = axes[2, 1]
        if 'Спящие' in cluster_pivot.columns:
            sleeping_rate = (cluster_pivot['Спящие'] / cluster_pivot.sum(axis=1)) * 100
            sleeping_rate.plot(kind='line', ax=ax6, marker='o', color='red', linewidth=2.5, markersize=8)
            ax6.fill_between(range(len(sleeping_rate)), sleeping_rate.values, alpha=0.3, color='red')
            ax6.set_title('Доля спящих клиентов', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Месяц')
            ax6.set_ylabel('Доля спящих (%)')
            ax6.grid(True, alpha=0.3)

            avg_sleeping = sleeping_rate.mean()
            ax6.axhline(y=avg_sleeping, color='darkred', linestyle='--', alpha=0.7,
                       label=f'Среднее: {avg_sleeping:.1f}%')
            ax6.legend()

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f'dynamics_{segment_name}.png'
            plt.savefig(output_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved dynamics plot to {output_path}")

        plt.show()

        return cluster_pivot

    def plot_migration_analysis(
        self,
        transitions: pd.DataFrame,
        segment_name: str,
        save: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Create migration analysis visualization.

        Args:
            transitions: DataFrame with transition data
            segment_name: Name of the segment
            save: Whether to save the plot

        Returns:
            Transition matrix
        """
        logger.info(f"Creating migration analysis visualization: {segment_name}")

        if transitions.empty:
            logger.warning("No transitions data to visualize")
            return None

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=VIZ_CONFIG['figsize_migration'])
        fig.suptitle(f'Анализ миграций: {segment_name}', fontsize=16, fontweight='bold')

        # 1. Transition probability matrix
        ax1 = axes[0, 0]
        transition_matrix = pd.crosstab(
            transitions['Cluster_from'],
            transitions['Cluster_to'],
            normalize='index'
        ) * 100

        transition_matrix.index = [CLUSTER_NAMES.get(idx, f'K{idx}') for idx in transition_matrix.index]
        transition_matrix.columns = [CLUSTER_NAMES.get(col, f'K{col}') for col in transition_matrix.columns]

        sns.heatmap(transition_matrix, annot=True, fmt='.1f', cmap='Blues', ax=ax1,
                   cbar_kws={'label': 'Вероятность перехода (%)'})
        ax1.set_title('Матрица вероятностей переходов', fontsize=12, fontweight='bold')
        ax1.set_xlabel('В кластер')
        ax1.set_ylabel('Из кластера')

        # 2. Top transitions
        ax2 = axes[0, 1]
        transition_counts = pd.crosstab(transitions['Cluster_from'], transitions['Cluster_to'])

        top_transitions = []
        for from_cluster in transition_counts.index:
            for to_cluster in transition_counts.columns:
                if from_cluster != to_cluster:
                    count = transition_counts.loc[from_cluster, to_cluster]
                    if count > 0:
                        from_name = CLUSTER_NAMES.get(from_cluster, f'K{from_cluster}')
                        to_name = CLUSTER_NAMES.get(to_cluster, f'K{to_cluster}')
                        top_transitions.append({'From': from_name, 'To': to_name, 'Count': count})

        if top_transitions:
            top_transitions_df = pd.DataFrame(top_transitions).sort_values('Count', ascending=False).head(10)
            y_pos = np.arange(len(top_transitions_df))
            ax2.barh(y_pos, top_transitions_df['Count'])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f"{row['From']} → {row['To']}" for _, row in top_transitions_df.iterrows()])
            ax2.set_title('Топ-10 переходов', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Количество клиентов')

        # 3. Retention rates
        ax3 = axes[1, 0]
        retention_by_cluster = {}
        for cluster in transition_matrix.index:
            if cluster in transition_matrix.columns:
                retention_by_cluster[cluster] = transition_matrix.loc[cluster, cluster]

        if retention_by_cluster:
            clusters = list(retention_by_cluster.keys())
            retention_values = list(retention_by_cluster.values())
            colors = ['green' if v > 70 else 'orange' if v > 50 else 'red' for v in retention_values]

            bars = ax3.bar(clusters, retention_values, color=colors)
            ax3.set_title('Уровень удержания (%)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Кластер')
            ax3.set_ylabel('Удержание (%)')
            ax3.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Хорошее (>70%)')
            ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Среднее (>50%)')
            ax3.legend()

            for bar, value in zip(bars, retention_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom')

        # 4. Reactivation from sleeping
        ax4 = axes[1, 1]
        if 'Спящие' in transition_matrix.index:
            reactivation = transition_matrix.loc['Спящие'].copy()
            reactivation = reactivation[reactivation.index != 'Спящие']

            if not reactivation.empty:
                reactivation.plot(kind='bar', ax=ax4, color='green')
                ax4.set_title('Реактивация спящих клиентов', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Целевой кластер')
                ax4.set_ylabel('Вероятность реактивации (%)')
                ax4.tick_params(axis='x', rotation=45)

                for i, (idx, val) in enumerate(reactivation.items()):
                    ax4.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom')

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f'migration_{segment_name}.png'
            plt.savefig(output_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved migration plot to {output_path}")

        plt.show()

        return transition_matrix

    def plot_churn_distribution(
        self,
        churn_data: pd.DataFrame,
        segment_name: str,
        save: bool = True
    ):
        """
        Visualize churn distribution.

        Args:
            churn_data: DataFrame with churn target
            segment_name: Name of the segment
            save: Whether to save the plot
        """
        logger.info(f"Creating churn distribution visualization: {segment_name}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Распределение Churn: {segment_name}', fontsize=16, fontweight='bold')

        # 1. Overall churn rate
        ax1 = axes[0, 0]
        churn_counts = churn_data['Churn'].value_counts()
        colors = ['green', 'red']
        ax1.pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax1.set_title('Общий уровень Churn', fontsize=12, fontweight='bold')

        # 2. Churn by cluster
        ax2 = axes[0, 1]
        churn_by_cluster = pd.crosstab(
            churn_data['Cluster'].map(CLUSTER_NAMES),
            churn_data['Churn'],
            normalize='index'
        ) * 100
        churn_by_cluster[1].plot(kind='bar', ax=ax2, color='red')
        ax2.set_title('Churn rate по кластерам', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Кластер')
        ax2.set_ylabel('Churn rate (%)')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Churn over time
        ax3 = axes[1, 0]
        churn_by_month = churn_data.groupby('Month')['Churn'].mean() * 100
        churn_by_month.plot(kind='line', ax=ax3, marker='o', color='red', linewidth=2)
        ax3.fill_between(range(len(churn_by_month)), churn_by_month.values, alpha=0.3, color='red')
        ax3.set_title('Churn rate по месяцам', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Месяц')
        ax3.set_ylabel('Churn rate (%)')
        ax3.grid(True, alpha=0.3)

        # 4. Distribution of key metrics by churn status
        ax4 = axes[1, 1]
        if 'Frequency' in churn_data.columns:
            churn_data.boxplot(column='Frequency', by='Churn', ax=ax4)
            ax4.set_title('Распределение Frequency по Churn статусу', fontsize=12)
            ax4.set_xlabel('Churn (0=No, 1=Yes)')
            ax4.set_ylabel('Frequency')
            plt.suptitle('')  # Remove default title

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f'churn_{segment_name}.png'
            plt.savefig(output_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved churn plot to {output_path}")

        plt.show()
