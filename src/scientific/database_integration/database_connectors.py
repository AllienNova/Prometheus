"""
Scientific database integration module for the Prometheus AI Automation Platform.

This module provides classes and functions for connecting to and retrieving data from
various scientific databases and APIs, including academic publication databases,
genomic databases, chemical databases, and more.
"""

import requests
import json
import pandas as pd
import numpy as np
import os
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
from io import StringIO
import re
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Base class for connecting to scientific databases and APIs.
    
    This class provides common functionality for authentication, rate limiting,
    error handling, and data retrieval from scientific databases and APIs.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                base_url: str = "",
                rate_limit: float = 1.0,
                timeout: float = 30.0,
                cache_dir: Optional[str] = None):
        """
        Initialize the DatabaseConnector.
        
        Args:
            api_key: API key for authentication (if required)
            base_url: Base URL for API requests
            rate_limit: Minimum time between requests in seconds
            timeout: Request timeout in seconds
            cache_dir: Directory for caching responses (if None, caching is disabled)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.last_request_time = 0
        
        # Create cache directory if specified and doesn't exist
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Cache directory set to {cache_dir}")
        
        logger.info(f"Initialized DatabaseConnector for {base_url}")
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_path(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Get the cache file path for a request.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Cache file path
        """
        if self.cache_dir is None:
            return ""
        
        # Create a unique filename based on endpoint and parameters
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hash(param_str) % 10000
        filename = f"{endpoint.replace('/', '_')}_{param_hash}.json"
        
        return os.path.join(self.cache_dir, filename)
    
    def _check_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """
        Check if a cached response exists and is valid.
        
        Args:
            cache_path: Cache file path
            
        Returns:
            Cached response data if valid, None otherwise
        """
        if not cache_path or not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired (default: 24 hours)
            cache_time = cached_data.get('_cache_time', 0)
            if time.time() - cache_time > 86400:  # 24 hours in seconds
                logger.debug(f"Cache expired for {cache_path}")
                return None
            
            logger.debug(f"Using cached response from {cache_path}")
            return cached_data
        except Exception as e:
            logger.warning(f"Error reading cache file {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_path: str, data: Dict[str, Any]) -> None:
        """
        Save response data to cache.
        
        Args:
            cache_path: Cache file path
            data: Response data to cache
        """
        if not cache_path:
            return
        
        try:
            # Add cache timestamp
            data_to_cache = data.copy()
            data_to_cache['_cache_time'] = time.time()
            
            with open(cache_path, 'w') as f:
                json.dump(data_to_cache, f)
            
            logger.debug(f"Saved response to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving to cache file {cache_path}: {e}")
    
    def make_request(self, endpoint: str, 
                    method: str = "GET", 
                    params: Optional[Dict[str, Any]] = None,
                    data: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None,
                    use_cache: bool = True) -> Dict[str, Any]:
        """
        Make a request to the API.
        
        Args:
            endpoint: API endpoint (will be appended to base_url)
            method: HTTP method (GET, POST, PUT, DELETE)
            params: Query parameters
            data: Request body for POST/PUT requests
            headers: Additional HTTP headers
            use_cache: Whether to use cached responses
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.RequestException: If the request fails
        """
        # Initialize parameters and headers if None
        params = params or {}
        headers = headers or {}
        
        # Add API key to parameters if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Construct full URL
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Check cache if enabled and method is GET
        cache_path = ""
        if use_cache and method.upper() == "GET" and self.cache_dir is not None:
            cache_path = self._get_cache_path(endpoint, params)
            cached_data = self._check_cache(cache_path)
            if cached_data is not None:
                return cached_data
        
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        # Make the request
        try:
            logger.debug(f"Making {method} request to {url}")
            
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=self.timeout
            )
            
            # Raise exception for HTTP errors
            response.raise_for_status()
            
            # Parse response based on content type
            if 'application/json' in response.headers.get('Content-Type', ''):
                result = response.json()
            elif 'text/xml' in response.headers.get('Content-Type', '') or 'application/xml' in response.headers.get('Content-Type', ''):
                # Return XML as string for now, specific parsers can handle it
                result = {'xml_content': response.text}
            else:
                # Return text content
                result = {'text_content': response.text}
            
            # Save to cache if enabled and method is GET
            if use_cache and method.upper() == "GET" and cache_path:
                self._save_to_cache(cache_path, result)
            
            return result
        
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional arguments for make_request
            
        Returns:
            Response data as dictionary
        """
        return self.make_request(endpoint, method="GET", params=params, **kwargs)
    
    def post(self, endpoint: str, data: Dict[str, Any], params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            **kwargs: Additional arguments for make_request
            
        Returns:
            Response data as dictionary
        """
        return self.make_request(endpoint, method="POST", params=params, data=data, **kwargs)
    
    def put(self, endpoint: str, data: Dict[str, Any], params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Make a PUT request to the API.
        
        Args:
            endpoint: API endpoint
            data: Request body
            params: Query parameters
            **kwargs: Additional arguments for make_request
            
        Returns:
            Response data as dictionary
        """
        return self.make_request(endpoint, method="PUT", params=params, data=data, **kwargs)
    
    def delete(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Make a DELETE request to the API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional arguments for make_request
            
        Returns:
            Response data as dictionary
        """
        return self.make_request(endpoint, method="DELETE", params=params, **kwargs)


class PubMedConnector(DatabaseConnector):
    """
    Connector for the PubMed/NCBI API.
    
    This class provides methods for searching and retrieving scientific publications
    from the PubMed database using the NCBI E-utilities API.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                email: str = "prometheus@example.com",
                tool: str = "PrometheusAI",
                rate_limit: float = 0.34,  # NCBI allows 3 requests per second with API key
                cache_dir: Optional[str] = None):
        """
        Initialize the PubMedConnector.
        
        Args:
            api_key: NCBI API key (optional but recommended)
            email: Email address for NCBI API usage tracking
            tool: Tool name for NCBI API usage tracking
            rate_limit: Minimum time between requests in seconds
            cache_dir: Directory for caching responses
        """
        super().__init__(
            api_key=api_key,
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate_limit=rate_limit,
            cache_dir=cache_dir
        )
        
        self.email = email
        self.tool = tool
        
        logger.info(f"Initialized PubMedConnector with tool={tool}, email={email}")
    
    def search(self, query: str, 
              max_results: int = 20,
              sort: str = "relevance",
              min_date: Optional[str] = None,
              max_date: Optional[str] = None) -> pd.DataFrame:
        """
        Search for publications in PubMed.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            sort: Sort order ('relevance', 'pub_date')
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)
            
        Returns:
            DataFrame with search results
        """
        # Prepare parameters
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance' if sort == 'relevance' else 'pub_date',
            'email': self.email,
            'tool': self.tool
        }
        
        # Add date range if specified
        if min_date:
            params['mindate'] = min_date.replace('/', '')
        if max_date:
            params['maxdate'] = max_date.replace('/', '')
        
        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Search for publication IDs
        logger.info(f"Searching PubMed for: {query}")
        search_results = self.get('esearch.fcgi', params=params)
        
        # Extract PMIDs
        pmids = search_results.get('esearchresult', {}).get('idlist', [])
        
        if not pmids:
            logger.warning(f"No results found for query: {query}")
            return pd.DataFrame()
        
        # Fetch publication details
        return self.fetch_publications(pmids)
    
    def fetch_publications(self, pmids: List[str]) -> pd.DataFrame:
        """
        Fetch publication details for a list of PubMed IDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            DataFrame with publication details
        """
        # Prepare parameters
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'email': self.email,
            'tool': self.tool
        }
        
        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Fetch publication details
        logger.info(f"Fetching details for {len(pmids)} publications")
        fetch_results = self.get('efetch.fcgi', params=params)
        
        # Parse XML content
        xml_content = fetch_results.get('xml_content', '')
        
        if not xml_content:
            logger.warning("No XML content returned from PubMed")
            return pd.DataFrame()
        
        # Parse XML and extract publication details
        return self._parse_pubmed_xml(xml_content)
    
    def _parse_pubmed_xml(self, xml_content: str) -> pd.DataFrame:
        """
        Parse PubMed XML content and extract publication details.
        
        Args:
            xml_content: PubMed XML content
            
        Returns:
            DataFrame with publication details
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Initialize lists for publication data
            pmids = []
            titles = []
            abstracts = []
            authors = []
            journals = []
            pub_dates = []
            dois = []
            
            # Extract publication details
            for article in root.findall('.//PubmedArticle'):
                # Extract PMID
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else ""
                pmids.append(pmid)
                
                # Extract title
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else ""
                titles.append(title)
                
                # Extract abstract
                abstract_elems = article.findall('.//AbstractText')
                abstract = ' '.join([elem.text for elem in abstract_elems if elem.text]) if abstract_elems else ""
                abstracts.append(abstract)
                
                # Extract authors
                author_elems = article.findall('.//Author')
                author_list = []
                for author in author_elems:
                    last_name = author.find('LastName')
                    fore_name = author.find('ForeName')
                    if last_name is not None and fore_name is not None:
                        author_list.append(f"{last_name.text}, {fore_name.text}")
                    elif last_name is not None:
                        author_list.append(last_name.text)
                authors.append('; '.join(author_list))
                
                # Extract journal
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else ""
                journals.append(journal)
                
                # Extract publication date
                pub_date_elem = article.find('.//PubDate')
                pub_date = ""
                if pub_date_elem is not None:
                    year = pub_date_elem.find('Year')
                    month = pub_date_elem.find('Month')
                    day = pub_date_elem.find('Day')
                    
                    year_text = year.text if year is not None else ""
                    month_text = month.text if month is not None else ""
                    day_text = day.text if day is not None else ""
                    
                    if year_text:
                        pub_date = year_text
                        if month_text:
                            pub_date += f"-{month_text}"
                            if day_text:
                                pub_date += f"-{day_text}"
                
                pub_dates.append(pub_date)
                
                # Extract DOI
                doi_elem = article.find('.//ArticleId[@IdType="doi"]')
                doi = doi_elem.text if doi_elem is not None else ""
                dois.append(doi)
            
            # Create DataFrame
            df = pd.DataFrame({
                'pmid': pmids,
                'title': titles,
                'abstract': abstracts,
                'authors': authors,
                'journal': journals,
                'publication_date': pub_dates,
                'doi': dois
            })
            
            return df
        
        except Exception as e:
            logger.error(f"Error parsing PubMed XML: {e}")
            return pd.DataFrame()


class ArXivConnector(DatabaseConnector):
    """
    Connector for the arXiv API.
    
    This class provides methods for searching and retrieving scientific preprints
    from the arXiv repository.
    """
    
    def __init__(self, rate_limit: float = 3.0,  # arXiv allows 1 request per 3 seconds
                cache_dir: Optional[str] = None):
        """
        Initialize the ArXivConnector.
        
        Args:
            rate_limit: Minimum time between requests in seconds
            cache_dir: Directory for caching responses
        """
        super().__init__(
            base_url="http://export.arxiv.org/api/query",
            rate_limit=rate_limit,
            cache_dir=cache_dir
        )
        
        logger.info("Initialized ArXivConnector")
    
    def search(self, query: str,
              max_results: int = 20,
              sort_by: str = "relevance",
              sort_order: str = "descending",
              categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Search for preprints in arXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            sort_by: Sort field ('relevance', 'lastUpdatedDate', 'submittedDate')
            sort_order: Sort order ('ascending', 'descending')
            categories: List of arXiv categories to search in (e.g., ['cs.AI', 'physics.comp-ph'])
            
        Returns:
            DataFrame with search results
        """
        # Prepare search query
        search_query = query
        
        # Add category filtering if specified
        if categories:
            category_filter = ' OR '.join([f'cat:{cat}' for cat in categories])
            search_query = f"({search_query}) AND ({category_filter})"
        
        # Prepare parameters
        params = {
            'search_query': search_query,
            'max_results': max_results,
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        # Make request
        logger.info(f"Searching arXiv for: {search_query}")
        response = self.get('', params=params)
        
        # Parse XML content
        xml_content = response.get('xml_content', '')
        
        if not xml_content:
            logger.warning("No XML content returned from arXiv")
            return pd.DataFrame()
        
        # Parse XML and extract publication details
        return self._parse_arxiv_xml(xml_content)
    
    def _parse_arxiv_xml(self, xml_content: str) -> pd.DataFrame:
        """
        Parse arXiv XML content and extract publication details.
        
        Args:
            xml_content: arXiv XML content
            
        Returns:
            DataFrame with publication details
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Define namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                 'arxiv': 'http://arxiv.org/schemas/atom'}
            
            # Initialize lists for publication data
            ids = []
            titles = []
            abstracts = []
            authors = []
            categories = []
            pub_dates = []
            update_dates = []
            pdf_urls = []
            
            # Extract publication details
            for entry in root.findall('.//atom:entry', ns):
                # Extract ID
                id_elem = entry.find('atom:id', ns)
                id_text = id_elem.text if id_elem is not None else ""
                # Extract arXiv ID from the URL
                arxiv_id = id_text.split('/')[-1] if id_text else ""
                ids.append(arxiv_id)
                
                # Extract title
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text if title_elem is not None else ""
                titles.append(title)
                
                # Extract abstract
                abstract_elem = entry.find('atom:summary', ns)
                abstract = abstract_elem.text if abstract_elem is not None else ""
                abstracts.append(abstract)
                
                # Extract authors
                author_elems = entry.findall('atom:author/atom:name', ns)
                author_list = [author.text for author in author_elems if author.text]
                authors.append('; '.join(author_list))
                
                # Extract categories
                category_elems = entry.findall('atom:category', ns)
                category_list = [cat.get('term') for cat in category_elems if cat.get('term')]
                categories.append('; '.join(category_list))
                
                # Extract publication date
                pub_date_elem = entry.find('atom:published', ns)
                pub_date = pub_date_elem.text if pub_date_elem is not None else ""
                pub_dates.append(pub_date)
                
                # Extract update date
                update_date_elem = entry.find('atom:updated', ns)
                update_date = update_date_elem.text if update_date_elem is not None else ""
                update_dates.append(update_date)
                
                # Extract PDF URL
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
                pdf_urls.append(pdf_url)
            
            # Create DataFrame
            df = pd.DataFrame({
                'arxiv_id': ids,
                'title': titles,
                'abstract': abstracts,
                'authors': authors,
                'categories': categories,
                'publication_date': pub_dates,
                'update_date': update_dates,
                'pdf_url': pdf_urls
            })
            
            return df
        
        except Exception as e:
            logger.error(f"Error parsing arXiv XML: {e}")
            return pd.DataFrame()


class CrossRefConnector(DatabaseConnector):
    """
    Connector for the CrossRef API.
    
    This class provides methods for searching and retrieving metadata for academic
    publications using the CrossRef API.
    """
    
    def __init__(self, mailto: str = "prometheus@example.com",
                rate_limit: float = 0.5,
                cache_dir: Optional[str] = None):
        """
        Initialize the CrossRefConnector.
        
        Args:
            mailto: Email address for CrossRef API usage tracking
            rate_limit: Minimum time between requests in seconds
            cache_dir: Directory for caching responses
        """
        super().__init__(
            base_url="https://api.crossref.org",
            rate_limit=rate_limit,
            cache_dir=cache_dir
        )
        
        self.mailto = mailto
        
        logger.info(f"Initialized CrossRefConnector with mailto={mailto}")
    
    def search(self, query: str,
              max_results: int = 20,
              filter_type: Optional[str] = None,
              from_date: Optional[str] = None,
              until_date: Optional[str] = None) -> pd.DataFrame:
        """
        Search for publications in CrossRef.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            filter_type: Filter by type (e.g., 'journal-article', 'book', 'proceedings-article')
            from_date: Start date in format YYYY-MM-DD
            until_date: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame with search results
        """
        # Prepare parameters
        params = {
            'query': query,
            'rows': max_results,
            'mailto': self.mailto
        }
        
        # Add filters if specified
        filters = []
        if filter_type:
            filters.append(f"type:{filter_type}")
        if from_date:
            filters.append(f"from-pub-date:{from_date}")
        if until_date:
            filters.append(f"until-pub-date:{until_date}")
        
        if filters:
            params['filter'] = ','.join(filters)
        
        # Make request
        logger.info(f"Searching CrossRef for: {query}")
        response = self.get('works', params=params)
        
        # Extract items
        items = response.get('message', {}).get('items', [])
        
        if not items:
            logger.warning(f"No results found for query: {query}")
            return pd.DataFrame()
        
        # Parse items and create DataFrame
        return self._parse_crossref_items(items)
    
    def fetch_by_doi(self, doi: str) -> Dict[str, Any]:
        """
        Fetch publication metadata by DOI.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Dictionary with publication metadata
        """
        # Prepare parameters
        params = {
            'mailto': self.mailto
        }
        
        # Make request
        logger.info(f"Fetching CrossRef metadata for DOI: {doi}")
        response = self.get(f"works/{doi}", params=params)
        
        return response.get('message', {})
    
    def _parse_crossref_items(self, items: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Parse CrossRef items and extract publication details.
        
        Args:
            items: List of CrossRef items
            
        Returns:
            DataFrame with publication details
        """
        # Initialize lists for publication data
        dois = []
        titles = []
        abstracts = []
        authors = []
        journals = []
        pub_dates = []
        types = []
        urls = []
        
        # Extract publication details
        for item in items:
            # Extract DOI
            doi = item.get('DOI', '')
            dois.append(doi)
            
            # Extract title
            title_list = item.get('title', [])
            title = title_list[0] if title_list else ""
            titles.append(title)
            
            # Extract abstract
            abstract = item.get('abstract', '')
            abstracts.append(abstract)
            
            # Extract authors
            author_list = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if given and family:
                    author_list.append(f"{family}, {given}")
                elif family:
                    author_list.append(family)
            authors.append('; '.join(author_list))
            
            # Extract journal
            container_title_list = item.get('container-title', [])
            journal = container_title_list[0] if container_title_list else ""
            journals.append(journal)
            
            # Extract publication date
            pub_date = ""
            published = item.get('published', {})
            date_parts = published.get('date-parts', [[]])
            if date_parts and date_parts[0]:
                date_part = date_parts[0]
                if len(date_part) >= 1:
                    pub_date = str(date_part[0])  # Year
                    if len(date_part) >= 2:
                        pub_date += f"-{date_part[1]:02d}"  # Month
                        if len(date_part) >= 3:
                            pub_date += f"-{date_part[2]:02d}"  # Day
            pub_dates.append(pub_date)
            
            # Extract type
            type_value = item.get('type', '')
            types.append(type_value)
            
            # Extract URL
            url = item.get('URL', '')
            urls.append(url)
        
        # Create DataFrame
        df = pd.DataFrame({
            'doi': dois,
            'title': titles,
            'abstract': abstracts,
            'authors': authors,
            'journal': journals,
            'publication_date': pub_dates,
            'type': types,
            'url': urls
        })
        
        return df


class DataCiteConnector(DatabaseConnector):
    """
    Connector for the DataCite API.
    
    This class provides methods for searching and retrieving metadata for research
    datasets using the DataCite API.
    """
    
    def __init__(self, rate_limit: float = 1.0,
                cache_dir: Optional[str] = None):
        """
        Initialize the DataCiteConnector.
        
        Args:
            rate_limit: Minimum time between requests in seconds
            cache_dir: Directory for caching responses
        """
        super().__init__(
            base_url="https://api.datacite.org",
            rate_limit=rate_limit,
            cache_dir=cache_dir
        )
        
        logger.info("Initialized DataCiteConnector")
    
    def search(self, query: str,
              max_results: int = 20,
              resource_type: Optional[str] = None,
              from_date: Optional[str] = None,
              until_date: Optional[str] = None) -> pd.DataFrame:
        """
        Search for datasets in DataCite.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            resource_type: Filter by resource type (e.g., 'dataset', 'software', 'text')
            from_date: Start date in format YYYY-MM-DD
            until_date: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame with search results
        """
        # Prepare parameters
        params = {
            'query': query,
            'page[size]': max_results
        }
        
        # Add filters if specified
        if resource_type:
            params['resource-type-id'] = resource_type
        if from_date:
            params['created'] = f"[{from_date}T00:00:00Z TO *]"
        if until_date:
            if 'created' in params:
                params['created'] = params['created'].replace(" TO *]", f" TO {until_date}T23:59:59Z]")
            else:
                params['created'] = f"[* TO {until_date}T23:59:59Z]"
        
        # Make request
        logger.info(f"Searching DataCite for: {query}")
        response = self.get('dois', params=params)
        
        # Extract items
        items = response.get('data', [])
        
        if not items:
            logger.warning(f"No results found for query: {query}")
            return pd.DataFrame()
        
        # Parse items and create DataFrame
        return self._parse_datacite_items(items)
    
    def fetch_by_doi(self, doi: str) -> Dict[str, Any]:
        """
        Fetch dataset metadata by DOI.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Dictionary with dataset metadata
        """
        # Make request
        logger.info(f"Fetching DataCite metadata for DOI: {doi}")
        response = self.get(f"dois/{doi}")
        
        return response.get('data', {})
    
    def _parse_datacite_items(self, items: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Parse DataCite items and extract dataset details.
        
        Args:
            items: List of DataCite items
            
        Returns:
            DataFrame with dataset details
        """
        # Initialize lists for dataset data
        dois = []
        titles = []
        descriptions = []
        creators = []
        publishers = []
        pub_years = []
        resource_types = []
        urls = []
        
        # Extract dataset details
        for item in items:
            # Extract DOI
            doi = item.get('id', '')
            dois.append(doi)
            
            # Extract attributes
            attributes = item.get('attributes', {})
            
            # Extract title
            title_list = attributes.get('titles', [])
            title = title_list[0].get('title', '') if title_list else ""
            titles.append(title)
            
            # Extract description
            description_list = attributes.get('descriptions', [])
            description = description_list[0].get('description', '') if description_list else ""
            descriptions.append(description)
            
            # Extract creators
            creator_list = []
            for creator in attributes.get('creators', []):
                name = creator.get('name', '')
                if name:
                    creator_list.append(name)
            creators.append('; '.join(creator_list))
            
            # Extract publisher
            publisher = attributes.get('publisher', '')
            publishers.append(publisher)
            
            # Extract publication year
            pub_year = attributes.get('publicationYear', '')
            pub_years.append(str(pub_year) if pub_year else "")
            
            # Extract resource type
            resource_type = attributes.get('resourceType', {}).get('resourceTypeGeneral', '')
            resource_types.append(resource_type)
            
            # Extract URL
            url = attributes.get('url', '')
            urls.append(url)
        
        # Create DataFrame
        df = pd.DataFrame({
            'doi': dois,
            'title': titles,
            'description': descriptions,
            'creators': creators,
            'publisher': publishers,
            'publication_year': pub_years,
            'resource_type': resource_types,
            'url': urls
        })
        
        return df


class OrcidConnector(DatabaseConnector):
    """
    Connector for the ORCID API.
    
    This class provides methods for retrieving researcher information and
    publication records from the ORCID registry.
    """
    
    def __init__(self, client_id: Optional[str] = None,
                client_secret: Optional[str] = None,
                rate_limit: float = 1.0,
                cache_dir: Optional[str] = None):
        """
        Initialize the OrcidConnector.
        
        Args:
            client_id: ORCID API client ID for authentication
            client_secret: ORCID API client secret for authentication
            rate_limit: Minimum time between requests in seconds
            cache_dir: Directory for caching responses
        """
        super().__init__(
            base_url="https://pub.orcid.org/v3.0",
            rate_limit=rate_limit,
            cache_dir=cache_dir
        )
        
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        
        logger.info("Initialized OrcidConnector")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers for ORCID API requests.
        
        Returns:
            Dictionary with HTTP headers
        """
        headers = {
            'Accept': 'application/json'
        }
        
        # Add authorization header if access token is available
        if self.access_token:
            headers['Authorization'] = f"Bearer {self.access_token}"
        
        return headers
    
    def search_researchers(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search for researchers in ORCID.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with researcher information
        """
        # Prepare parameters
        params = {
            'q': query,
            'rows': max_results
        }
        
        # Make request
        logger.info(f"Searching ORCID for researchers: {query}")
        response = self.get('search', params=params, headers=self._get_headers())
        
        # Extract results
        result = response.get('result', [])
        
        if not result:
            logger.warning(f"No researchers found for query: {query}")
            return pd.DataFrame()
        
        # Parse results and create DataFrame
        return self._parse_orcid_search_results(result)
    
    def get_researcher(self, orcid_id: str) -> Dict[str, Any]:
        """
        Get researcher information by ORCID ID.
        
        Args:
            orcid_id: ORCID identifier
            
        Returns:
            Dictionary with researcher information
        """
        # Make request
        logger.info(f"Fetching ORCID record for: {orcid_id}")
        response = self.get(f"{orcid_id}/person", headers=self._get_headers())
        
        return response
    
    def get_works(self, orcid_id: str) -> pd.DataFrame:
        """
        Get works (publications) for a researcher by ORCID ID.
        
        Args:
            orcid_id: ORCID identifier
            
        Returns:
            DataFrame with works information
        """
        # Make request
        logger.info(f"Fetching works for ORCID: {orcid_id}")
        response = self.get(f"{orcid_id}/works", headers=self._get_headers())
        
        # Extract works
        group = response.get('group', [])
        
        if not group:
            logger.warning(f"No works found for ORCID: {orcid_id}")
            return pd.DataFrame()
        
        # Parse works and create DataFrame
        return self._parse_orcid_works(group)
    
    def _parse_orcid_search_results(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Parse ORCID search results and extract researcher information.
        
        Args:
            results: List of ORCID search results
            
        Returns:
            DataFrame with researcher information
        """
        # Initialize lists for researcher data
        orcid_ids = []
        given_names = []
        family_names = []
        credit_names = []
        institutions = []
        
        # Extract researcher details
        for result in results:
            # Extract ORCID ID
            orcid_id = result.get('orcid-identifier', {}).get('path', '')
            orcid_ids.append(orcid_id)
            
            # Extract name
            given_name = ""
            family_name = ""
            credit_name = ""
            
            person = result.get('person', {})
            name = person.get('name', {})
            
            if name:
                given_name = name.get('given-names', {}).get('value', '')
                family_name = name.get('family-name', {}).get('value', '')
                credit_name_obj = name.get('credit-name', {})
                if credit_name_obj:
                    credit_name = credit_name_obj.get('value', '')
            
            given_names.append(given_name)
            family_names.append(family_name)
            credit_names.append(credit_name)
            
            # Extract institution
            institution = ""
            affiliations = person.get('affiliations', {}).get('affiliation', [])
            
            if affiliations:
                org = affiliations[0].get('organization', {})
                if org:
                    institution = org.get('name', '')
            
            institutions.append(institution)
        
        # Create DataFrame
        df = pd.DataFrame({
            'orcid_id': orcid_ids,
            'given_name': given_names,
            'family_name': family_names,
            'credit_name': credit_names,
            'institution': institutions
        })
        
        return df
    
    def _parse_orcid_works(self, works_group: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Parse ORCID works and extract publication details.
        
        Args:
            works_group: List of ORCID works groups
            
        Returns:
            DataFrame with publication details
        """
        # Initialize lists for publication data
        titles = []
        types = []
        pub_years = []
        journals = []
        dois = []
        urls = []
        
        # Extract publication details
        for group in works_group:
            work_summary = group.get('work-summary', [])
            
            if not work_summary:
                continue
            
            # Use the first work summary (preferred version)
            summary = work_summary[0]
            
            # Extract title
            title = summary.get('title', {}).get('title', {}).get('value', '')
            titles.append(title)
            
            # Extract type
            type_value = summary.get('type', '')
            types.append(type_value)
            
            # Extract publication year
            pub_year = ""
            publication_date = summary.get('publication-date', {})
            if publication_date:
                year = publication_date.get('year', {}).get('value', '')
                pub_year = year
            pub_years.append(pub_year)
            
            # Extract journal
            journal = summary.get('journal-title', {}).get('value', '')
            journals.append(journal)
            
            # Extract DOI
            doi = ""
            external_ids = summary.get('external-ids', {}).get('external-id', [])
            for ext_id in external_ids:
                if ext_id.get('external-id-type', '') == 'doi':
                    doi = ext_id.get('external-id-value', '')
                    break
            dois.append(doi)
            
            # Extract URL
            url = summary.get('url', {}).get('value', '')
            urls.append(url)
        
        # Create DataFrame
        df = pd.DataFrame({
            'title': titles,
            'type': types,
            'publication_year': pub_years,
            'journal': journals,
            'doi': dois,
            'url': urls
        })
        
        return df


class GenBankConnector(DatabaseConnector):
    """
    Connector for the GenBank/NCBI Nucleotide database.
    
    This class provides methods for retrieving genetic sequence data from
    the GenBank database using the NCBI E-utilities API.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                email: str = "prometheus@example.com",
                tool: str = "PrometheusAI",
                rate_limit: float = 0.34,  # NCBI allows 3 requests per second with API key
                cache_dir: Optional[str] = None):
        """
        Initialize the GenBankConnector.
        
        Args:
            api_key: NCBI API key (optional but recommended)
            email: Email address for NCBI API usage tracking
            tool: Tool name for NCBI API usage tracking
            rate_limit: Minimum time between requests in seconds
            cache_dir: Directory for caching responses
        """
        super().__init__(
            api_key=api_key,
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            rate_limit=rate_limit,
            cache_dir=cache_dir
        )
        
        self.email = email
        self.tool = tool
        
        logger.info(f"Initialized GenBankConnector with tool={tool}, email={email}")
    
    def search(self, query: str, max_results: int = 20) -> List[str]:
        """
        Search for genetic sequences in GenBank.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of GenBank accession numbers
        """
        # Prepare parameters
        params = {
            'db': 'nucleotide',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'email': self.email,
            'tool': self.tool
        }
        
        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Search for sequence IDs
        logger.info(f"Searching GenBank for: {query}")
        search_results = self.get('esearch.fcgi', params=params)
        
        # Extract IDs
        ids = search_results.get('esearchresult', {}).get('idlist', [])
        
        if not ids:
            logger.warning(f"No results found for query: {query}")
            return []
        
        return ids
    
    def fetch_sequences(self, ids: List[str], format: str = 'fasta') -> Dict[str, str]:
        """
        Fetch genetic sequences by GenBank IDs.
        
        Args:
            ids: List of GenBank IDs
            format: Sequence format ('fasta', 'gb', 'gbc')
            
        Returns:
            Dictionary mapping IDs to sequences
        """
        # Prepare parameters
        params = {
            'db': 'nucleotide',
            'id': ','.join(ids),
            'rettype': format,
            'retmode': 'text',
            'email': self.email,
            'tool': self.tool
        }
        
        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Fetch sequences
        logger.info(f"Fetching {len(ids)} sequences from GenBank")
        fetch_results = self.get('efetch.fcgi', params=params)
        
        # Extract text content
        text_content = fetch_results.get('text_content', '')
        
        if not text_content:
            logger.warning("No sequence content returned from GenBank")
            return {}
        
        # Parse sequences based on format
        if format == 'fasta':
            return self._parse_fasta(text_content)
        else:
            # For other formats, return the raw content
            return {'raw_content': text_content}
    
    def fetch_sequence_info(self, ids: List[str]) -> pd.DataFrame:
        """
        Fetch metadata for genetic sequences by GenBank IDs.
        
        Args:
            ids: List of GenBank IDs
            
        Returns:
            DataFrame with sequence metadata
        """
        # Prepare parameters
        params = {
            'db': 'nucleotide',
            'id': ','.join(ids),
            'retmode': 'xml',
            'email': self.email,
            'tool': self.tool
        }
        
        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key
        
        # Fetch sequence metadata
        logger.info(f"Fetching metadata for {len(ids)} sequences from GenBank")
        fetch_results = self.get('esummary.fcgi', params=params)
        
        # Extract XML content
        xml_content = fetch_results.get('xml_content', '')
        
        if not xml_content:
            logger.warning("No metadata content returned from GenBank")
            return pd.DataFrame()
        
        # Parse XML and extract sequence metadata
        return self._parse_genbank_xml(xml_content)
    
    def _parse_fasta(self, fasta_content: str) -> Dict[str, str]:
        """
        Parse FASTA format content and extract sequences.
        
        Args:
            fasta_content: FASTA format content
            
        Returns:
            Dictionary mapping sequence IDs to sequences
        """
        sequences = {}
        current_id = None
        current_seq = []
        
        for line in fasta_content.splitlines():
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                
                # Start new sequence
                header = line[1:].strip()
                current_id = header.split()[0]  # Use first word as ID
                current_seq = []
            else:
                # Append to current sequence
                if current_id is not None:
                    current_seq.append(line.strip())
        
        # Save last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    def _parse_genbank_xml(self, xml_content: str) -> pd.DataFrame:
        """
        Parse GenBank XML content and extract sequence metadata.
        
        Args:
            xml_content: GenBank XML content
            
        Returns:
            DataFrame with sequence metadata
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Initialize lists for sequence data
            ids = []
            accessions = []
            titles = []
            lengths = []
            organisms = []
            taxonomies = []
            update_dates = []
            
            # Extract sequence details
            for doc_sum in root.findall('.//DocSum'):
                # Extract ID
                id_elem = doc_sum.find('./Id')
                id_value = id_elem.text if id_elem is not None else ""
                ids.append(id_value)
                
                # Extract other metadata
                accession = ""
                title = ""
                length = ""
                organism = ""
                taxonomy = ""
                update_date = ""
                
                for item in doc_sum.findall('./Item'):
                    name = item.get('Name', '')
                    
                    if name == 'Caption':
                        accession = item.text if item.text else ""
                    elif name == 'Title':
                        title = item.text if item.text else ""
                    elif name == 'Length':
                        length = item.text if item.text else ""
                    elif name == 'Organism':
                        organism = item.text if item.text else ""
                    elif name == 'TaxId':
                        taxonomy = item.text if item.text else ""
                    elif name == 'UpdateDate':
                        update_date = item.text if item.text else ""
                
                accessions.append(accession)
                titles.append(title)
                lengths.append(length)
                organisms.append(organism)
                taxonomies.append(taxonomy)
                update_dates.append(update_date)
            
            # Create DataFrame
            df = pd.DataFrame({
                'id': ids,
                'accession': accessions,
                'title': titles,
                'length': lengths,
                'organism': organisms,
                'taxonomy_id': taxonomies,
                'update_date': update_dates
            })
            
            return df
        
        except Exception as e:
            logger.error(f"Error parsing GenBank XML: {e}")
            return pd.DataFrame()


class ChemicalDatabaseConnector(DatabaseConnector):
    """
    Connector for the PubChem chemical database.
    
    This class provides methods for retrieving chemical compound information
    from the PubChem database using the PubChem PUG REST API.
    """
    
    def __init__(self, rate_limit: float = 0.2,  # PubChem allows 5 requests per second
                cache_dir: Optional[str] = None):
        """
        Initialize the ChemicalDatabaseConnector.
        
        Args:
            rate_limit: Minimum time between requests in seconds
            cache_dir: Directory for caching responses
        """
        super().__init__(
            base_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            rate_limit=rate_limit,
            cache_dir=cache_dir
        )
        
        logger.info("Initialized ChemicalDatabaseConnector for PubChem")
    
    def search_compounds(self, query: str, max_results: int = 20) -> List[int]:
        """
        Search for chemical compounds in PubChem.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of PubChem compound IDs (CIDs)
        """
        # Prepare parameters
        params = {
            'name': query,
            'maxrecords': max_results
        }
        
        # Make request
        logger.info(f"Searching PubChem for compounds: {query}")
        response = self.get('compound/name/cids/JSON', params=params)
        
        # Extract CIDs
        identifiers_list = response.get('IdentifierList', {})
        cids = identifiers_list.get('CID', [])
        
        if not cids:
            logger.warning(f"No compounds found for query: {query}")
            return []
        
        return cids
    
    def get_compound_info(self, cid: Union[int, str]) -> Dict[str, Any]:
        """
        Get chemical compound information by PubChem CID.
        
        Args:
            cid: PubChem Compound ID
            
        Returns:
            Dictionary with compound information
        """
        # Make request
        logger.info(f"Fetching PubChem compound info for CID: {cid}")
        response = self.get(f"compound/cid/{cid}/JSON")
        
        # Extract compound information
        pc_compounds = response.get('PC_Compounds', [])
        
        if not pc_compounds:
            logger.warning(f"No compound information found for CID: {cid}")
            return {}
        
        return pc_compounds[0]
    
    def get_compound_properties(self, cid: Union[int, str], properties: List[str]) -> Dict[str, Any]:
        """
        Get specific properties for a chemical compound by PubChem CID.
        
        Args:
            cid: PubChem Compound ID
            properties: List of property names to retrieve
                (e.g., ['MolecularFormula', 'MolecularWeight', 'CanonicalSMILES'])
            
        Returns:
            Dictionary with compound properties
        """
        # Prepare property string
        property_str = ','.join(properties)
        
        # Make request
        logger.info(f"Fetching PubChem compound properties for CID: {cid}")
        response = self.get(f"compound/cid/{cid}/property/{property_str}/JSON")
        
        # Extract property information
        property_table = response.get('PropertyTable', {})
        properties_list = property_table.get('Properties', [])
        
        if not properties_list:
            logger.warning(f"No property information found for CID: {cid}")
            return {}
        
        return properties_list[0]
    
    def get_compound_image(self, cid: Union[int, str], image_type: str = 'png', size: str = 'large') -> bytes:
        """
        Get chemical compound image by PubChem CID.
        
        Args:
            cid: PubChem Compound ID
            image_type: Image format ('png', 'svg')
            size: Image size ('small', 'medium', 'large')
            
        Returns:
            Image data as bytes
        """
        # Validate parameters
        if image_type not in ['png', 'svg']:
            raise ValueError("Image type must be 'png' or 'svg'")
        
        if size not in ['small', 'medium', 'large']:
            raise ValueError("Size must be 'small', 'medium', or 'large'")
        
        # Make request
        logger.info(f"Fetching PubChem compound image for CID: {cid}")
        url = f"{self.base_url}/compound/cid/{cid}/PNG?image_size={size}" if image_type == 'png' else f"{self.base_url}/compound/cid/{cid}/SVG"
        
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        # Make direct request to get binary data
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        
        return response.content
    
    def search_substances(self, query: str, max_results: int = 20) -> List[int]:
        """
        Search for chemical substances in PubChem.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of PubChem substance IDs (SIDs)
        """
        # Prepare parameters
        params = {
            'name': query,
            'maxrecords': max_results
        }
        
        # Make request
        logger.info(f"Searching PubChem for substances: {query}")
        response = self.get('substance/name/sids/JSON', params=params)
        
        # Extract SIDs
        identifiers_list = response.get('IdentifierList', {})
        sids = identifiers_list.get('SID', [])
        
        if not sids:
            logger.warning(f"No substances found for query: {query}")
            return []
        
        return sids
    
    def get_substance_info(self, sid: Union[int, str]) -> Dict[str, Any]:
        """
        Get chemical substance information by PubChem SID.
        
        Args:
            sid: PubChem Substance ID
            
        Returns:
            Dictionary with substance information
        """
        # Make request
        logger.info(f"Fetching PubChem substance info for SID: {sid}")
        response = self.get(f"substance/sid/{sid}/JSON")
        
        # Extract substance information
        pc_substances = response.get('PC_Substances', [])
        
        if not pc_substances:
            logger.warning(f"No substance information found for SID: {sid}")
            return {}
        
        return pc_substances[0]


class ScientificDatabaseManager:
    """
    Manager class for scientific database connectors.
    
    This class provides a unified interface for accessing various scientific
    databases and APIs, managing connections, and retrieving data.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the ScientificDatabaseManager.
        
        Args:
            cache_dir: Directory for caching responses
            api_keys: Dictionary mapping database names to API keys
        """
        self.cache_dir = cache_dir
        self.api_keys = api_keys or {}
        
        # Create cache directory if specified and doesn't exist
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Cache directory set to {cache_dir}")
        
        # Initialize connectors
        self._init_connectors()
        
        logger.info("Initialized ScientificDatabaseManager")
    
    def _init_connectors(self) -> None:
        """Initialize database connectors."""
        # Create database-specific cache directories
        pubmed_cache_dir = os.path.join(self.cache_dir, 'pubmed') if self.cache_dir else None
        arxiv_cache_dir = os.path.join(self.cache_dir, 'arxiv') if self.cache_dir else None
        crossref_cache_dir = os.path.join(self.cache_dir, 'crossref') if self.cache_dir else None
        datacite_cache_dir = os.path.join(self.cache_dir, 'datacite') if self.cache_dir else None
        orcid_cache_dir = os.path.join(self.cache_dir, 'orcid') if self.cache_dir else None
        genbank_cache_dir = os.path.join(self.cache_dir, 'genbank') if self.cache_dir else None
        pubchem_cache_dir = os.path.join(self.cache_dir, 'pubchem') if self.cache_dir else None
        
        # Create cache directories if needed
        for cache_dir in [pubmed_cache_dir, arxiv_cache_dir, crossref_cache_dir, 
                         datacite_cache_dir, orcid_cache_dir, genbank_cache_dir, 
                         pubchem_cache_dir]:
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize connectors
        self.pubmed = PubMedConnector(
            api_key=self.api_keys.get('pubmed'),
            cache_dir=pubmed_cache_dir
        )
        
        self.arxiv = ArXivConnector(
            cache_dir=arxiv_cache_dir
        )
        
        self.crossref = CrossRefConnector(
            cache_dir=crossref_cache_dir
        )
        
        self.datacite = DataCiteConnector(
            cache_dir=datacite_cache_dir
        )
        
        self.orcid = OrcidConnector(
            client_id=self.api_keys.get('orcid_client_id'),
            client_secret=self.api_keys.get('orcid_client_secret'),
            cache_dir=orcid_cache_dir
        )
        
        self.genbank = GenBankConnector(
            api_key=self.api_keys.get('genbank'),
            cache_dir=genbank_cache_dir
        )
        
        self.pubchem = ChemicalDatabaseConnector(
            cache_dir=pubchem_cache_dir
        )
    
    def search_publications(self, query: str, 
                          sources: List[str] = ['pubmed', 'arxiv', 'crossref'],
                          max_results_per_source: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Search for publications across multiple sources.
        
        Args:
            query: Search query
            sources: List of sources to search ('pubmed', 'arxiv', 'crossref')
            max_results_per_source: Maximum number of results per source
            
        Returns:
            Dictionary mapping source names to DataFrames with search results
        """
        results = {}
        
        # Search PubMed
        if 'pubmed' in sources:
            try:
                logger.info(f"Searching PubMed for: {query}")
                pubmed_results = self.pubmed.search(query, max_results=max_results_per_source)
                results['pubmed'] = pubmed_results
            except Exception as e:
                logger.error(f"Error searching PubMed: {e}")
        
        # Search arXiv
        if 'arxiv' in sources:
            try:
                logger.info(f"Searching arXiv for: {query}")
                arxiv_results = self.arxiv.search(query, max_results=max_results_per_source)
                results['arxiv'] = arxiv_results
            except Exception as e:
                logger.error(f"Error searching arXiv: {e}")
        
        # Search CrossRef
        if 'crossref' in sources:
            try:
                logger.info(f"Searching CrossRef for: {query}")
                crossref_results = self.crossref.search(query, max_results=max_results_per_source)
                results['crossref'] = crossref_results
            except Exception as e:
                logger.error(f"Error searching CrossRef: {e}")
        
        return results
    
    def search_datasets(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search for datasets in DataCite.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with dataset information
        """
        try:
            logger.info(f"Searching DataCite for datasets: {query}")
            return self.datacite.search(query, max_results=max_results, resource_type='dataset')
        except Exception as e:
            logger.error(f"Error searching DataCite: {e}")
            return pd.DataFrame()
    
    def search_researchers(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search for researchers in ORCID.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with researcher information
        """
        try:
            logger.info(f"Searching ORCID for researchers: {query}")
            return self.orcid.search_researchers(query, max_results=max_results)
        except Exception as e:
            logger.error(f"Error searching ORCID: {e}")
            return pd.DataFrame()
    
    def search_genetic_sequences(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search for genetic sequences in GenBank.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with sequence information
        """
        try:
            logger.info(f"Searching GenBank for sequences: {query}")
            ids = self.genbank.search(query, max_results=max_results)
            if not ids:
                return pd.DataFrame()
            
            return self.genbank.fetch_sequence_info(ids)
        except Exception as e:
            logger.error(f"Error searching GenBank: {e}")
            return pd.DataFrame()
    
    def search_chemical_compounds(self, query: str, max_results: int = 20) -> pd.DataFrame:
        """
        Search for chemical compounds in PubChem.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with compound information
        """
        try:
            logger.info(f"Searching PubChem for compounds: {query}")
            cids = self.pubchem.search_compounds(query, max_results=max_results)
            if not cids:
                return pd.DataFrame()
            
            # Get properties for each compound
            properties = ['MolecularFormula', 'MolecularWeight', 'CanonicalSMILES', 
                         'IUPACName', 'XLogP', 'HBondDonorCount', 'HBondAcceptorCount', 
                         'RotatableBondCount', 'ExactMass', 'MonoisotopicMass']
            
            # Initialize lists for compound data
            compound_data = []
            
            for cid in cids:
                try:
                    prop_data = self.pubchem.get_compound_properties(cid, properties)
                    if prop_data:
                        prop_data['CID'] = cid
                        compound_data.append(prop_data)
                except Exception as e:
                    logger.warning(f"Error getting properties for CID {cid}: {e}")
            
            if not compound_data:
                return pd.DataFrame()
            
            # Create DataFrame
            return pd.DataFrame(compound_data)
        
        except Exception as e:
            logger.error(f"Error searching PubChem: {e}")
            return pd.DataFrame()
    
    def get_publication_by_doi(self, doi: str) -> Dict[str, Any]:
        """
        Get publication information by DOI.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Dictionary with publication information
        """
        try:
            logger.info(f"Fetching publication information for DOI: {doi}")
            return self.crossref.fetch_by_doi(doi)
        except Exception as e:
            logger.error(f"Error fetching publication by DOI: {e}")
            return {}
    
    def get_dataset_by_doi(self, doi: str) -> Dict[str, Any]:
        """
        Get dataset information by DOI.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Dictionary with dataset information
        """
        try:
            logger.info(f"Fetching dataset information for DOI: {doi}")
            return self.datacite.fetch_by_doi(doi)
        except Exception as e:
            logger.error(f"Error fetching dataset by DOI: {e}")
            return {}
    
    def get_researcher_by_orcid(self, orcid_id: str) -> Dict[str, Any]:
        """
        Get researcher information by ORCID ID.
        
        Args:
            orcid_id: ORCID identifier
            
        Returns:
            Dictionary with researcher information
        """
        try:
            logger.info(f"Fetching researcher information for ORCID: {orcid_id}")
            return self.orcid.get_researcher(orcid_id)
        except Exception as e:
            logger.error(f"Error fetching researcher by ORCID: {e}")
            return {}
    
    def get_researcher_works(self, orcid_id: str) -> pd.DataFrame:
        """
        Get works (publications) for a researcher by ORCID ID.
        
        Args:
            orcid_id: ORCID identifier
            
        Returns:
            DataFrame with works information
        """
        try:
            logger.info(f"Fetching works for ORCID: {orcid_id}")
            return self.orcid.get_works(orcid_id)
        except Exception as e:
            logger.error(f"Error fetching works by ORCID: {e}")
            return pd.DataFrame()
    
    def get_genetic_sequence(self, sequence_id: str, format: str = 'fasta') -> Dict[str, str]:
        """
        Get genetic sequence by GenBank ID.
        
        Args:
            sequence_id: GenBank sequence ID
            format: Sequence format ('fasta', 'gb', 'gbc')
            
        Returns:
            Dictionary with sequence information
        """
        try:
            logger.info(f"Fetching genetic sequence for ID: {sequence_id}")
            return self.genbank.fetch_sequences([sequence_id], format=format)
        except Exception as e:
            logger.error(f"Error fetching genetic sequence: {e}")
            return {}
    
    def get_chemical_compound(self, cid: Union[int, str]) -> Dict[str, Any]:
        """
        Get chemical compound information by PubChem CID.
        
        Args:
            cid: PubChem Compound ID
            
        Returns:
            Dictionary with compound information
        """
        try:
            logger.info(f"Fetching chemical compound information for CID: {cid}")
            return self.pubchem.get_compound_info(cid)
        except Exception as e:
            logger.error(f"Error fetching chemical compound: {e}")
            return {}
