"""
Scientific database integration API for the Prometheus AI Automation Platform.

This module provides a high-level API for interacting with scientific databases
and retrieving research data for scientific discovery and analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import time
from datetime import datetime

from .database_connectors import ScientificDatabaseManager

# Configure logging
logger = logging.getLogger(__name__)

class ScientificDatabaseAPI:
    """
    High-level API for scientific database integration.
    
    This class provides simplified methods for retrieving scientific data
    from various databases and APIs, with automatic caching and error handling.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, api_keys_file: Optional[str] = None):
        """
        Initialize the ScientificDatabaseAPI.
        
        Args:
            cache_dir: Directory for caching responses
            api_keys_file: Path to JSON file containing API keys
        """
        # Load API keys if file is provided
        api_keys = {}
        if api_keys_file and os.path.exists(api_keys_file):
            try:
                with open(api_keys_file, 'r') as f:
                    api_keys = json.load(f)
                logger.info(f"Loaded API keys from {api_keys_file}")
            except Exception as e:
                logger.error(f"Error loading API keys from {api_keys_file}: {e}")
        
        # Initialize database manager
        self.db_manager = ScientificDatabaseManager(cache_dir=cache_dir, api_keys=api_keys)
        
        logger.info("Initialized ScientificDatabaseAPI")
    
    def search_literature(self, query: str, 
                         sources: List[str] = ['pubmed', 'arxiv', 'crossref'],
                         max_results: int = 50,
                         sort_by: str = 'relevance',
                         filter_by_year: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """
        Search for scientific literature across multiple sources.
        
        Args:
            query: Search query
            sources: List of sources to search ('pubmed', 'arxiv', 'crossref')
            max_results: Maximum total number of results to return
            sort_by: Sort order ('relevance', 'date')
            filter_by_year: Tuple of (start_year, end_year) to filter results
            
        Returns:
            DataFrame with combined search results
        """
        # Calculate max results per source
        max_per_source = max(5, max_results // len(sources))
        
        # Search across sources
        results_dict = self.db_manager.search_publications(
            query=query,
            sources=sources,
            max_results_per_source=max_per_source
        )
        
        # Combine results
        combined_results = []
        
        for source, df in results_dict.items():
            if df.empty:
                continue
            
            # Add source column
            df['source'] = source
            
            # Standardize column names
            if source == 'pubmed':
                df = df.rename(columns={
                    'pmid': 'id',
                    'publication_date': 'date'
                })
            elif source == 'arxiv':
                df = df.rename(columns={
                    'arxiv_id': 'id',
                    'publication_date': 'date'
                })
            elif source == 'crossref':
                df = df.rename(columns={
                    'doi': 'id',
                    'publication_date': 'date'
                })
            
            # Append to combined results
            combined_results.append(df)
        
        if not combined_results:
            logger.warning(f"No results found for query: {query}")
            return pd.DataFrame()
        
        # Concatenate results
        result_df = pd.concat(combined_results, ignore_index=True)
        
        # Filter by year if specified
        if filter_by_year:
            start_year, end_year = filter_by_year
            
            # Extract year from date column
            def extract_year(date_str):
                if not date_str:
                    return None
                
                # Try different date formats
                for fmt in ('%Y-%m-%d', '%Y-%m', '%Y', '%Y-%m-%dT%H:%M:%SZ'):
                    try:
                        return datetime.strptime(date_str.split('T')[0], fmt).year
                    except (ValueError, AttributeError):
                        continue
                
                # Try to extract year directly
                try:
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                    if year_match:
                        return int(year_match.group(0))
                except:
                    pass
                
                return None
            
            # Apply year filter
            result_df['year'] = result_df['date'].apply(extract_year)
            result_df = result_df[(result_df['year'] >= start_year) & (result_df['year'] <= end_year)]
            
            if result_df.empty:
                logger.warning(f"No results found after year filter: {start_year}-{end_year}")
                return pd.DataFrame()
        
        # Sort results
        if sort_by == 'date':
            # Sort by date (descending)
            result_df = result_df.sort_values(by='date', ascending=False, na_position='last')
        
        # Limit to max_results
        if len(result_df) > max_results:
            result_df = result_df.iloc[:max_results]
        
        return result_df
    
    def get_publication_details(self, publication_id: str, source: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific publication.
        
        Args:
            publication_id: Publication identifier (DOI, PMID, arXiv ID)
            source: Source database ('pubmed', 'arxiv', 'crossref')
            
        Returns:
            Dictionary with publication details
        """
        if source == 'pubmed':
            try:
                # Fetch publication from PubMed
                results = self.db_manager.pubmed.fetch_publications([publication_id])
                if not results.empty:
                    return results.iloc[0].to_dict()
            except Exception as e:
                logger.error(f"Error fetching publication from PubMed: {e}")
        
        elif source == 'arxiv':
            try:
                # Construct arXiv query
                query = f"id:{publication_id}"
                results = self.db_manager.arxiv.search(query, max_results=1)
                if not results.empty:
                    return results.iloc[0].to_dict()
            except Exception as e:
                logger.error(f"Error fetching publication from arXiv: {e}")
        
        elif source == 'crossref':
            try:
                # Fetch publication from CrossRef by DOI
                return self.db_manager.get_publication_by_doi(publication_id)
            except Exception as e:
                logger.error(f"Error fetching publication from CrossRef: {e}")
        
        return {}
    
    def search_datasets(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search for scientific datasets.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with dataset information
        """
        return self.db_manager.search_datasets(query, max_results=max_results)
    
    def get_dataset_details(self, dataset_doi: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific dataset.
        
        Args:
            dataset_doi: Dataset DOI
            
        Returns:
            Dictionary with dataset details
        """
        return self.db_manager.get_dataset_by_doi(dataset_doi)
    
    def search_researchers(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search for researchers.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with researcher information
        """
        return self.db_manager.search_researchers(query, max_results=max_results)
    
    def get_researcher_publications(self, orcid_id: str) -> pd.DataFrame:
        """
        Get publications for a specific researcher.
        
        Args:
            orcid_id: ORCID identifier
            
        Returns:
            DataFrame with publication information
        """
        return self.db_manager.get_researcher_works(orcid_id)
    
    def search_genetic_sequences(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search for genetic sequences.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with sequence information
        """
        return self.db_manager.search_genetic_sequences(query, max_results=max_results)
    
    def get_genetic_sequence(self, sequence_id: str, format: str = 'fasta') -> Dict[str, str]:
        """
        Get a specific genetic sequence.
        
        Args:
            sequence_id: GenBank sequence ID
            format: Sequence format ('fasta', 'gb', 'gbc')
            
        Returns:
            Dictionary with sequence information
        """
        return self.db_manager.get_genetic_sequence(sequence_id, format=format)
    
    def search_chemical_compounds(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search for chemical compounds.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with compound information
        """
        return self.db_manager.search_chemical_compounds(query, max_results=max_results)
    
    def get_chemical_compound_properties(self, compound_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get properties for a specific chemical compound.
        
        Args:
            compound_id: PubChem Compound ID
            
        Returns:
            Dictionary with compound properties
        """
        properties = [
            'MolecularFormula', 'MolecularWeight', 'CanonicalSMILES', 
            'IUPACName', 'XLogP', 'HBondDonorCount', 'HBondAcceptorCount', 
            'RotatableBondCount', 'ExactMass', 'MonoisotopicMass'
        ]
        
        try:
            return self.db_manager.pubchem.get_compound_properties(compound_id, properties)
        except Exception as e:
            logger.error(f"Error fetching chemical compound properties: {e}")
            return {}
    
    def get_citation_network(self, doi: str, depth: int = 1) -> Dict[str, Any]:
        """
        Get citation network for a publication.
        
        Args:
            doi: Digital Object Identifier
            depth: Depth of citation network (1 = direct citations only)
            
        Returns:
            Dictionary with citation network information
        """
        # This is a placeholder for a more complex implementation
        # that would retrieve citation data from CrossRef or other sources
        
        logger.warning("Citation network retrieval is not fully implemented")
        
        return {
            'doi': doi,
            'depth': depth,
            'citations': [],
            'references': []
        }
    
    def get_research_trend(self, query: str, years: int = 5) -> Dict[str, Any]:
        """
        Get research trend data for a topic.
        
        Args:
            query: Search query
            years: Number of years to analyze
            
        Returns:
            Dictionary with trend data
        """
        # This is a placeholder for a more complex implementation
        # that would analyze publication trends over time
        
        logger.warning("Research trend analysis is not fully implemented")
        
        current_year = datetime.now().year
        years_range = list(range(current_year - years + 1, current_year + 1))
        
        return {
            'query': query,
            'years': years_range,
            'publication_counts': [0] * years,
            'trending_keywords': []
        }


# Create a singleton instance for easy import
scientific_db_api = None

def initialize_scientific_db_api(cache_dir: Optional[str] = None, api_keys_file: Optional[str] = None) -> ScientificDatabaseAPI:
    """
    Initialize the scientific database API singleton.
    
    Args:
        cache_dir: Directory for caching responses
        api_keys_file: Path to JSON file containing API keys
        
    Returns:
        ScientificDatabaseAPI instance
    """
    global scientific_db_api
    
    if scientific_db_api is None:
        scientific_db_api = ScientificDatabaseAPI(cache_dir=cache_dir, api_keys_file=api_keys_file)
    
    return scientific_db_api

def get_scientific_db_api() -> ScientificDatabaseAPI:
    """
    Get the scientific database API singleton instance.
    
    Returns:
        ScientificDatabaseAPI instance
    """
    global scientific_db_api
    
    if scientific_db_api is None:
        # Initialize with default settings
        scientific_db_api = ScientificDatabaseAPI()
    
    return scientific_db_api
