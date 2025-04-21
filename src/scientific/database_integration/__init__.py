"""
Scientific database integration initialization module for the Prometheus AI Automation Platform.

This module provides initialization and utility functions for the scientific database
integration components.
"""

import os
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import API and connectors
from .api import initialize_scientific_db_api, get_scientific_db_api
from .database_connectors import (
    DatabaseConnector,
    PubMedConnector,
    ArXivConnector,
    CrossRefConnector,
    DataCiteConnector,
    OrcidConnector,
    GenBankConnector,
    ChemicalDatabaseConnector,
    ScientificDatabaseManager
)

def setup_scientific_databases(cache_dir: Optional[str] = None, 
                              api_keys_file: Optional[str] = None) -> None:
    """
    Set up scientific database integration components.
    
    Args:
        cache_dir: Directory for caching database responses
        api_keys_file: Path to JSON file containing API keys
    """
    # Create cache directory if specified and doesn't exist
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Created cache directory: {cache_dir}")
    
    # Initialize the scientific database API
    initialize_scientific_db_api(cache_dir=cache_dir, api_keys_file=api_keys_file)
    logger.info("Scientific database integration initialized successfully")

def get_available_databases() -> Dict[str, str]:
    """
    Get a dictionary of available scientific databases.
    
    Returns:
        Dictionary mapping database names to descriptions
    """
    return {
        'pubmed': 'PubMed/NCBI biomedical literature database',
        'arxiv': 'arXiv preprint repository',
        'crossref': 'CrossRef publication metadata database',
        'datacite': 'DataCite research dataset repository',
        'orcid': 'ORCID researcher identification registry',
        'genbank': 'GenBank/NCBI genetic sequence database',
        'pubchem': 'PubChem chemical compound database'
    }

def get_database_capabilities() -> Dict[str, Dict[str, Any]]:
    """
    Get capabilities of each scientific database.
    
    Returns:
        Dictionary mapping database names to capability information
    """
    return {
        'pubmed': {
            'search': True,
            'fetch_by_id': True,
            'metadata': True,
            'full_text': False,
            'rate_limit': '3 requests per second with API key',
            'api_key_required': False,
            'api_key_recommended': True
        },
        'arxiv': {
            'search': True,
            'fetch_by_id': True,
            'metadata': True,
            'full_text': True,
            'rate_limit': '1 request per 3 seconds',
            'api_key_required': False,
            'api_key_recommended': False
        },
        'crossref': {
            'search': True,
            'fetch_by_id': True,
            'metadata': True,
            'full_text': False,
            'rate_limit': '50 requests per second with polite pool',
            'api_key_required': False,
            'api_key_recommended': False
        },
        'datacite': {
            'search': True,
            'fetch_by_id': True,
            'metadata': True,
            'full_text': False,
            'rate_limit': '1 request per second',
            'api_key_required': False,
            'api_key_recommended': False
        },
        'orcid': {
            'search': True,
            'fetch_by_id': True,
            'metadata': True,
            'full_text': False,
            'rate_limit': '1 request per second',
            'api_key_required': False,
            'api_key_recommended': True
        },
        'genbank': {
            'search': True,
            'fetch_by_id': True,
            'metadata': True,
            'full_text': True,
            'rate_limit': '3 requests per second with API key',
            'api_key_required': False,
            'api_key_recommended': True
        },
        'pubchem': {
            'search': True,
            'fetch_by_id': True,
            'metadata': True,
            'full_text': False,
            'rate_limit': '5 requests per second',
            'api_key_required': False,
            'api_key_recommended': False
        }
    }
