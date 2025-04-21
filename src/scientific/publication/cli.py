"""
Publication CLI module for the Prometheus AI Automation Platform.

This module provides a command-line interface for creating, managing, and exporting
scientific publications.
"""

import os
import sys
import argparse
import logging
import json
import datetime
from typing import List, Optional, Dict, Any

from .publication import (
    Publication, Author, Section, Reference, Figure, Table,
    PublicationFormat, CitationStyle, JournalTemplate,
    get_publication_manager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_publication(args: argparse.Namespace) -> None:
    """
    Create a new publication.
    
    Args:
        args: Command-line arguments
    """
    # Parse authors
    authors = []
    for author_str in args.authors:
        parts = author_str.split(',')
        name = parts[0].strip()
        
        author_kwargs = {'name': name}
        
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'email':
                    author_kwargs['email'] = value
                elif key == 'affiliation':
                    author_kwargs['affiliation'] = value
                elif key == 'orcid':
                    author_kwargs['orcid'] = value
                elif key == 'corresponding' and value.lower() in ('true', 'yes', '1'):
                    author_kwargs['is_corresponding'] = True
        
        authors.append(Author(**author_kwargs))
    
    # Parse sections
    sections = []
    if args.sections_file:
        with open(args.sections_file, 'r') as f:
            sections_data = json.load(f)
        
        for section_data in sections_data:
            sections.append(Section(**section_data))
    else:
        # Create a basic structure with empty sections
        sections = [
            Section(title="Introduction", content="", level=1, id="introduction"),
            Section(title="Methods", content="", level=1, id="methods"),
            Section(title="Results", content="", level=1, id="results"),
            Section(title="Discussion", content="", level=1, id="discussion"),
            Section(title="Conclusion", content="", level=1, id="conclusion")
        ]
    
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Create publication
    publication = manager.create_publication(
        title=args.title,
        authors=authors,
        abstract=args.abstract,
        keywords=args.keywords.split(',') if args.keywords else [],
        sections=sections
    )
    
    # Save publication
    filepath = manager.save_publication(publication.id)
    
    print(f"Created publication '{publication.title}' with ID {publication.id}")
    print(f"Saved to {filepath}")

def list_publications(args: argparse.Namespace) -> None:
    """
    List all publications.
    
    Args:
        args: Command-line arguments
    """
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Load all publications
    manager.load_all_publications()
    
    # List publications
    publications = manager.list_publications()
    
    if not publications:
        print("No publications found.")
        return
    
    print(f"Found {len(publications)} publications:")
    for publication in publications:
        print(f"- {publication.id}: {publication.title} ({len(publication.sections)} sections, {len(publication.references)} references)")

def show_publication(args: argparse.Namespace) -> None:
    """
    Show details of a publication.
    
    Args:
        args: Command-line arguments
    """
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Load all publications
    manager.load_all_publications()
    
    # Get publication
    publication = manager.get_publication(args.id)
    
    if publication is None:
        print(f"Publication with ID {args.id} not found.")
        return
    
    # Print publication details
    print(f"Publication ID: {publication.id}")
    print(f"Title: {publication.title}")
    print(f"Authors: {', '.join([author.name for author in publication.authors])}")
    print(f"Abstract: {publication.abstract}")
    print(f"Keywords: {', '.join(publication.keywords)}")
    print(f"Created: {publication.created_at}")
    print(f"Updated: {publication.updated_at}")
    
    print(f"\nSections ({len(publication.sections)}):")
    for section in publication.sections:
        print(f"- {section.title} (Level {section.level})")
    
    print(f"\nFigures ({len(publication.figures)}):")
    for figure in publication.figures:
        print(f"- {figure.id}: {figure.title}")
    
    print(f"\nTables ({len(publication.tables)}):")
    for table in publication.tables:
        print(f"- {table.id}: {table.title}")
    
    print(f"\nReferences ({len(publication.references)}):")
    for reference in publication.references:
        print(f"- {reference.id}: {reference.title} ({reference.year})")

def export_publication(args: argparse.Namespace) -> None:
    """
    Export a publication to the specified format.
    
    Args:
        args: Command-line arguments
    """
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Load all publications
    manager.load_all_publications()
    
    # Get publication
    publication = manager.get_publication(args.id)
    
    if publication is None:
        print(f"Publication with ID {args.id} not found.")
        return
    
    # Parse format
    try:
        format = PublicationFormat(args.format)
    except ValueError:
        print(f"Invalid format: {args.format}")
        print(f"Valid formats: {', '.join([f.value for f in PublicationFormat])}")
        return
    
    # Parse citation style
    try:
        citation_style = CitationStyle(args.citation_style)
    except ValueError:
        print(f"Invalid citation style: {args.citation_style}")
        print(f"Valid citation styles: {', '.join([s.value for s in CitationStyle])}")
        return
    
    # Parse journal template
    try:
        journal_template = JournalTemplate(args.journal_template)
    except ValueError:
        print(f"Invalid journal template: {args.journal_template}")
        print(f"Valid journal templates: {', '.join([t.value for t in JournalTemplate])}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export publication
    filepath = manager.export_publication(
        publication_id=args.id,
        format=format,
        output_dir=args.output_dir,
        citation_style=citation_style,
        journal_template=journal_template
    )
    
    if filepath:
        print(f"Exported publication '{publication.title}' to {filepath}")
    else:
        print(f"Failed to export publication '{publication.title}'")

def delete_publication(args: argparse.Namespace) -> None:
    """
    Delete a publication.
    
    Args:
        args: Command-line arguments
    """
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Load all publications
    manager.load_all_publications()
    
    # Delete publication
    success = manager.delete_publication(args.id)
    
    if success:
        print(f"Deleted publication with ID {args.id}")
    else:
        print(f"Failed to delete publication with ID {args.id}")

def add_section(args: argparse.Namespace) -> None:
    """
    Add a section to a publication.
    
    Args:
        args: Command-line arguments
    """
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Load all publications
    manager.load_all_publications()
    
    # Get publication
    publication = manager.get_publication(args.id)
    
    if publication is None:
        print(f"Publication with ID {args.id} not found.")
        return
    
    # Create section
    section = Section(
        title=args.title,
        content=args.content,
        level=args.level,
        id=args.section_id
    )
    
    # Add section to publication
    publication.add_section(section)
    
    # Save publication
    filepath = manager.save_publication(publication.id)
    
    print(f"Added section '{section.title}' to publication '{publication.title}'")
    print(f"Saved to {filepath}")

def add_reference(args: argparse.Namespace) -> None:
    """
    Add a reference to a publication.
    
    Args:
        args: Command-line arguments
    """
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Load all publications
    manager.load_all_publications()
    
    # Get publication
    publication = manager.get_publication(args.id)
    
    if publication is None:
        print(f"Publication with ID {args.id} not found.")
        return
    
    # Create reference
    reference = Reference(
        id=args.reference_id,
        type=args.type,
        title=args.title,
        authors=args.authors.split(','),
        year=args.year,
        source=args.source,
        doi=args.doi,
        url=args.url,
        volume=args.volume,
        issue=args.issue,
        pages=args.pages,
        publisher=args.publisher
    )
    
    # Add reference to publication
    publication.add_reference(reference)
    
    # Save publication
    filepath = manager.save_publication(publication.id)
    
    print(f"Added reference '{reference.id}' to publication '{publication.title}'")
    print(f"Saved to {filepath}")

def add_figure(args: argparse.Namespace) -> None:
    """
    Add a figure to a publication.
    
    Args:
        args: Command-line arguments
    """
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Load all publications
    manager.load_all_publications()
    
    # Get publication
    publication = manager.get_publication(args.id)
    
    if publication is None:
        print(f"Publication with ID {args.id} not found.")
        return
    
    # Check if figure path exists
    if not os.path.exists(args.path):
        print(f"Figure path '{args.path}' does not exist.")
        return
    
    # Create figure
    figure = Figure(
        id=args.figure_id,
        title=args.title,
        caption=args.caption,
        path=args.path,
        type=args.type,
        width=args.width,
        height=args.height,
        source=args.source,
        notes=args.notes
    )
    
    # Add figure to publication
    publication.add_figure(figure)
    
    # Save publication
    filepath = manager.save_publication(publication.id)
    
    print(f"Added figure '{figure.id}' to publication '{publication.title}'")
    print(f"Saved to {filepath}")

def add_table(args: argparse.Namespace) -> None:
    """
    Add a table to a publication.
    
    Args:
        args: Command-line arguments
    """
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Load all publications
    manager.load_all_publications()
    
    # Get publication
    publication = manager.get_publication(args.id)
    
    if publication is None:
        print(f"Publication with ID {args.id} not found.")
        return
    
    # Parse data
    if args.data_file:
        # Data is a path to a CSV file
        if not os.path.exists(args.data_file):
            print(f"Data file '{args.data_file}' does not exist.")
            return
        
        data = args.data_file
    else:
        # Data is provided as a JSON string
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError:
            print(f"Invalid JSON data: {args.data}")
            return
    
    # Parse header
    header = None
    if args.header:
        try:
            header = json.loads(args.header)
        except json.JSONDecodeError:
            print(f"Invalid JSON header: {args.header}")
            return
    
    # Create table
    table = Table(
        id=args.table_id,
        title=args.title,
        caption=args.caption,
        data=data,
        header=header,
        footer=args.footer,
        notes=args.notes
    )
    
    # Add table to publication
    publication.add_table(table)
    
    # Save publication
    filepath = manager.save_publication(publication.id)
    
    print(f"Added table '{table.id}' to publication '{publication.title}'")
    print(f"Saved to {filepath}")

def create_from_experiment(args: argparse.Namespace) -> None:
    """
    Create a publication from an experiment.
    
    Args:
        args: Command-line arguments
    """
    # Get publication manager
    manager = get_publication_manager(args.storage_dir)
    
    # Parse authors
    authors = []
    if args.authors:
        for author_str in args.authors:
            parts = author_str.split(',')
            name = parts[0].strip()
            
            author_kwargs = {'name': name}
            
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'email':
                        author_kwargs['email'] = value
                    elif key == 'affiliation':
                        author_kwargs['affiliation'] = value
                    elif key == 'orcid':
                        author_kwargs['orcid'] = value
                    elif key == 'corresponding' and value.lower() in ('true', 'yes', '1'):
                        author_kwargs['is_corresponding'] = True
            
            authors.append(Author(**author_kwargs))
    
    # Create publication from experiment
    publication = manager.create_publication_from_experiment(
        experiment_id=args.experiment_id,
        title=args.title,
        authors=authors if authors else None,
        abstract=args.abstract,
        keywords=args.keywords.split(',') if args.keywords else None
    )
    
    if publication:
        print(f"Created publication '{publication.title}' from experiment '{args.experiment_id}'")
        print(f"Publication ID: {publication.id}")
    else:
        print(f"Failed to create publication from experiment '{args.experiment_id}'")

def main() -> None:
    """
    Main entry point for the publication CLI.
    """
    parser = argparse.ArgumentParser(description='Prometheus Publication CLI')
    parser.add_argument('--storage-dir', default='publications', help='Directory for storing publication data')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create publication command
    create_parser = subparsers.add_parser('create', help='Create a new publication')
    create_parser.add_argument('--title', required=True, help='Publication title')
    create_parser.add_argument('--authors', required=True, nargs='+', help='List of authors (format: "Name, email=email@example.com, affiliation=University, orcid=0000-0000-0000-0000, corresponding=true")')
    create_parser.add_argument('--abstract', required=True, help='Publication abstract')
    create_parser.add_argument('--keywords', help='Comma-separated list of keywords')
    create_parser.add_argument('--sections-file', help='Path to a JSON file containing sections')
    create_parser.set_defaults(func=create_publication)
    
    # List publications command
    list_parser = subparsers.add_parser('list', help='List all publications')
    list_parser.set_defaults(func=list_publications)
    
    # Show publication command
    show_parser = subparsers.add_parser('show', help='Show details of a publication')
    show_parser.add_argument('id', help='Publication ID')
    show_parser.set_defaults(func=show_publication)
    
    # Export publication command
    export_parser = subparsers.add_parser('export', help='Export a publication to the specified format')
    export_parser.add_argument('id', help='Publication ID')
    export_parser.add_argument('--format', required=True, help=f'Output format ({", ".join([f.value for f in PublicationFormat])})')
    export_parser.add_argument('--output-dir', required=True, help='Directory to save the exported file')
    export_parser.add_argument('--citation-style', default='apa', help=f'Citation style ({", ".join([s.value for s in CitationStyle])})')
    export_parser.add_argument('--journal-template', default='generic', help=f'Journal template ({", ".join([t.value for t in JournalTemplate])})')
    export_parser.set_defaults(func=export_publication)
    
    # Delete publication command
    delete_parser = subparsers.add_parser('delete', help='Delete a publication')
    delete_parser.add_argument('id', help='Publication ID')
    delete_parser.set_defaults(func=delete_publication)
    
    # Add section command
    add_section_parser = subparsers.add_parser('add-section', help='Add a section to a publication')
    add_section_parser.add_argument('id', help='Publication ID')
    add_section_parser.add_argument('--title', required=True, help='Section title')
    add_section_parser.add_argument('--content', required=True, help='Section content')
    add_section_parser.add_argument('--level', type=int, default=1, help='Section level (1 = main section, 2 = subsection, etc.)')
    add_section_parser.add_argument('--section-id', help='Section ID (optional)')
    add_section_parser.set_defaults(func=add_section)
    
    # Add reference command
    add_reference_parser = subparsers.add_parser('add-reference', help='Add a reference to a publication')
    add_reference_parser.add_argument('id', help='Publication ID')
    add_reference_parser.add_argument('--reference-id', required=True, help='Reference ID')
    add_reference_parser.add_argument('--type', required=True, help='Reference type (article, book, conference, etc.)')
    add_reference_parser.add_argument('--title', required=True, help='Reference title')
    add_reference_parser.add_argument('--authors', required=True, help='Comma-separated list of authors')
    add_reference_parser.add_argument('--year', type=int, required=True, help='Publication year')
    add_reference_parser.add_argument('--source', required=True, help='Source (journal name, book publisher, etc.)')
    add_reference_parser.add_argument('--doi', help='DOI')
    add_reference_parser.add_argument('--url', help='URL')
    add_reference_parser.add_argument('--volume', help='Volume')
    add_reference_parser.add_argument('--issue', help='Issue')
    add_reference_parser.add_argument('--pages', help='Pages')
    add_reference_parser.add_argument('--publisher', help='Publisher')
    add_reference_parser.set_defaults(func=add_reference)
    
    # Add figure command
    add_figure_parser = subparsers.add_parser('add-figure', help='Add a figure to a publication')
    add_figure_parser.add_argument('id', help='Publication ID')
    add_figure_parser.add_argument('--figure-id', required=True, help='Figure ID')
    add_figure_parser.add_argument('--title', required=True, help='Figure title')
    add_figure_parser.add_argument('--caption', required=True, help='Figure caption')
    add_figure_parser.add_argument('--path', required=True, help='Path to the figure file')
    add_figure_parser.add_argument('--type', default='image', help='Figure type (image, chart, diagram, etc.)')
    add_figure_parser.add_argument('--width', help='Figure width')
    add_figure_parser.add_argument('--height', help='Figure height')
    add_figure_parser.add_argument('--source', help='Figure source')
    add_figure_parser.add_argument('--notes', help='Figure notes')
    add_figure_parser.set_defaults(func=add_figure)
    
    # Add table command
    add_table_parser = subparsers.add_parser('add-table', help='Add a table to a publication')
    add_table_parser.add_argument('id', help='Publication ID')
    add_table_parser.add_argument('--table-id', required=True, help='Table ID')
    add_table_parser.add_argument('--title', required=True, help='Table title')
    add_table_parser.add_argument('--caption', required=True, help='Table caption')
    add_table_parser.add_argument('--data', help='Table data as a JSON array of arrays')
    add_table_parser.add_argument('--data-file', help='Path to a CSV file containing table data')
    add_table_parser.add_argument('--header', help='Table header as a JSON array')
    add_table_parser.add_argument('--footer', help='Table footer')
    add_table_parser.add_argument('--notes', help='Table notes')
    add_table_parser.set_defaults(func=add_table)
    
    # Create from experiment command
    create_from_experiment_parser = subparsers.add_parser('create-from-experiment', help='Create a publication from an experiment')
    create_from_experiment_parser.add_argument('experiment_id', help='Experiment ID')
    create_from_experiment_parser.add_argument('--title', help='Publication title (defaults to experiment name)')
    create_from_experiment_parser.add_argument('--authors', nargs='+', help='List of authors (format: "Name, email=email@example.com, affiliation=University, orcid=0000-0000-0000-0000, corresponding=true")')
    create_from_experiment_parser.add_argument('--abstract', help='Publication abstract (defaults to experiment description)')
    create_from_experiment_parser.add_argument('--keywords', help='Comma-separated list of keywords (defaults to experiment tags)')
    create_from_experiment_parser.set_defaults(func=create_from_experiment)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)

if __name__ == '__main__':
    main()
