"""
Database connection and data loading module.
Handles Oracle database connections and SQL queries.
"""

import logging
import pandas as pd
import oracledb
from typing import Dict, Optional
from config.settings import DB_CONFIG, ANALYSIS_CONFIG

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Handles Oracle database connections and data loading."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize database connector.

        Args:
            config: Database configuration dictionary. If None, uses DB_CONFIG from settings.
        """
        self.config = config or DB_CONFIG
        self.connection = None

    def connect(self) -> oracledb.Connection:
        """
        Establish connection to Oracle database.

        Returns:
            Oracle database connection object.

        Raises:
            ConnectionError: If connection fails.
        """
        try:
            self.connection = oracledb.connect(
                user=self.config['user'],
                password=self.config['password'],
                host=self.config['host'],
                port=self.config['port'],
                service_name=self.config['service_name']
            )
            logger.info("Successfully connected to Oracle database")
            return self.connection
        except oracledb.Error as e:
            logger.error(f"Failed to connect to Oracle database: {e}")
            raise ConnectionError(f"Database connection failed: {e}")

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
            self.connection = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class DataLoader:
    """Loads data from database for clustering analysis."""

    def __init__(self, connector: DatabaseConnector):
        """
        Initialize data loader.

        Args:
            connector: DatabaseConnector instance.
        """
        self.connector = connector
        self.start_date = ANALYSIS_CONFIG['start_date']

    def load_main_transactions(self) -> pd.DataFrame:
        """
        Load main RKO transactions.

        Returns:
            DataFrame with main transactions.
        """
        logger.info("Loading main transactions...")

        query = f"""
            SELECT
                BINIIN, DATE_BY_MONTH, KNP, KNP_1, AMOUNT_DT, CURRENCY,
                PAYER_BINIIN, PAYER_BANK, RECIPIENT_BINIIN, RECIPIENT_BANK, OPER_GROUP
            FROM MSBMCKINSEY.t_dz_all_rko_operations
            WHERE DATE_BY_MONTH > TO_DATE('{self.start_date}', 'YYYY-MM-DD')
        """

        df = pd.read_sql(query, self.connector.connection)
        logger.info(f"Loaded {len(df):,} main transactions")
        return df

    def load_business_metadata(self) -> pd.DataFrame:
        """
        Load business type metadata for clients.

        Returns:
            DataFrame with client metadata.
        """
        logger.info("Loading business metadata...")

        query = """
            SELECT BINIIN, BUSINESS_TYPE, INDUSTRY, LEGAL_FORM_ABBR, REG_DATE
            FROM MSBMCKINSEY.msb_feature_store_our
        """

        df = pd.read_sql(query, self.connector.connection)
        logger.info(f"Loaded metadata for {len(df):,} clients")
        return df

    def load_social_transactions(self) -> pd.DataFrame:
        """
        Load social payment transactions.

        Returns:
            DataFrame with social transactions.
        """
        logger.info("Loading social transactions...")

        query = """
            SELECT BINIIN, DATE_BY_MONTH, KNP, KNP_1, AMOUNT_DT, CURRENCY,
                   PAYER_BINIIN, PAYER_BANK, RECIPIENT_BINIIN, RECIPIENT_BANK, OPER_GROUP
            FROM MSBMCKINSEY.t_dz_com_soc_perevody
            WHERE ACT_DATE >= TO_DATE('01.01.2025', 'DD.MM.YYYY')
                AND biniin NOT IN (
                    SELECT biniin FROM MSBMCKINSEY.t_dz_all_rko_operations
                    WHERE oper_group NOT IN ('prochie','comission','soc','soc?')
                        AND ACT_DATE >= TO_DATE('01.01.2025', 'DD.MM.YYYY')
                    GROUP BY biniin
                )
        """

        df = pd.read_sql(query, self.connector.connection)
        logger.info(f"Loaded {len(df):,} social transactions")
        return df

    def load_cash_transactions(self) -> pd.DataFrame:
        """
        Load cash withdrawal transactions.

        Returns:
            DataFrame with cash transactions.
        """
        logger.info("Loading cash transactions...")

        query = f"""
            SELECT BINIIN, DATE_BY_MONTH, KNP, KNP_1, AMOUNT_DT, CURRENCY,
                   PAYER_BINIIN, PAYER_BANK, RECIPIENT_BINIIN, RECIPIENT_BANK, OPER_GROUP
            FROM MSBMCKINSEY.t_dz_com_vydacha
            WHERE DATE_BY_MONTH > TO_DATE('{self.start_date}', 'YYYY-MM-DD')
                AND biniin NOT IN (
                    SELECT biniin FROM MSBMCKINSEY.t_dz_all_rko_operations
                    WHERE oper_group NOT IN ('prochie','comission','vydacha','vydacha ak')
                        AND act_date >= TO_DATE('01.01.2025', 'DD.MM.YYYY')
                    GROUP BY biniin
                )
        """

        df = pd.read_sql(query, self.connector.connection)
        logger.info(f"Loaded {len(df):,} cash transactions")
        return df

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data required for analysis.

        Returns:
            Dictionary with all loaded DataFrames.
        """
        logger.info("="*80)
        logger.info("LOADING DATA FROM DATABASE")
        logger.info("="*80)

        data = {
            'main': self.load_main_transactions(),
            'business_type': self.load_business_metadata(),
            'soc': self.load_social_transactions(),
            'cash': self.load_cash_transactions()
        }

        logger.info("All data loaded successfully")
        return data
