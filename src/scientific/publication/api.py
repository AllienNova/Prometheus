"""
Publication API module for the Prometheus AI Automation Platform.

This module provides a RESTful API for creating, managing, and exporting
scientific publications.
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .publication import (
    Publication, Author, Section, Reference, Figure, Table,
    PublicationFormat, CitationStyle, JournalTemplate,
    get_publication_manager
)

# Create FastAPI app
app = FastAPI(
    title="Prometheus Publication API",
    description="API for managing scientific publications",
    version="0.1.0"
)

# Models for API requests and responses
class AuthorModel(BaseModel):
    name: str
    email: Optional[str] = None
    affiliation: Optional[str] = None
    orcid: Optional[str] = None
    is_corresponding: bool = False

class SectionModel(BaseModel):
    title: str
    content: str
    level: int = 1
    id: Optional[str] = None

class ReferenceModel(BaseModel):
    id: str
    type: str
    title: str
    authors: List[str]
    year: int
    source: str
    doi: Optional[str] = None
    url: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = []

class FigureModel(BaseModel):
    id: str
    title: str
    caption: str
    path: str
    type: str = "image"
    width: Optional[str] = None
    height: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

class TableModel(BaseModel):
    id: str
    title: str
    caption: str
    data: Any  # Can be a 2D array or a path to a CSV file
    header: Optional[List[str]] = None
    footer: Optional[str] = None
    notes: Optional[str] = None

class PublicationCreateModel(BaseModel):
    title: str
    authors: List[AuthorModel]
    abstract: str
    keywords: List[str]
    sections: List[SectionModel]

class PublicationUpdateModel(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    acknowledgments: Optional[str] = None
    funding: Optional[str] = None
    conflicts_of_interest: Optional[str] = None

class PublicationExportModel(BaseModel):
    format: str
    citation_style: str = "apa"
    journal_template: str = "generic"

class PublicationFromExperimentModel(BaseModel):
    experiment_id: str
    title: Optional[str] = None
    authors: Optional[List[AuthorModel]] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None

class PublicationResponseModel(BaseModel):
    id: str
    title: str
    authors: List[AuthorModel]
    abstract: str
    keywords: List[str]
    created_at: datetime.datetime
    updated_at: datetime.datetime
    sections_count: int
    figures_count: int
    tables_count: int
    references_count: int

# Helper functions
def publication_to_response(publication: Publication) -> PublicationResponseModel:
    """
    Convert a Publication object to a PublicationResponseModel.
    
    Args:
        publication: Publication object
        
    Returns:
        PublicationResponseModel
    """
    return PublicationResponseModel(
        id=publication.id,
        title=publication.title,
        authors=[
            AuthorModel(
                name=author.name,
                email=author.email,
                affiliation=author.affiliation,
                orcid=author.orcid,
                is_corresponding=author.is_corresponding
            )
            for author in publication.authors
        ],
        abstract=publication.abstract,
        keywords=publication.keywords,
        created_at=publication.created_at,
        updated_at=publication.updated_at,
        sections_count=len(publication.sections),
        figures_count=len(publication.figures),
        tables_count=len(publication.tables),
        references_count=len(publication.references)
    )

# API routes
@app.get("/publications", response_model=List[PublicationResponseModel])
def list_publications():
    """
    List all publications.
    
    Returns:
        List of publications
    """
    manager = get_publication_manager()
    manager.load_all_publications()
    
    publications = manager.list_publications()
    return [publication_to_response(publication) for publication in publications]

@app.post("/publications", response_model=PublicationResponseModel)
def create_publication(publication_data: PublicationCreateModel):
    """
    Create a new publication.
    
    Args:
        publication_data: Publication data
        
    Returns:
        Created publication
    """
    manager = get_publication_manager()
    
    # Convert authors
    authors = [
        Author(
            name=author.name,
            email=author.email,
            affiliation=author.affiliation,
            orcid=author.orcid,
            is_corresponding=author.is_corresponding
        )
        for author in publication_data.authors
    ]
    
    # Convert sections
    sections = [
        Section(
            title=section.title,
            content=section.content,
            level=section.level,
            id=section.id
        )
        for section in publication_data.sections
    ]
    
    # Create publication
    publication = manager.create_publication(
        title=publication_data.title,
        authors=authors,
        abstract=publication_data.abstract,
        keywords=publication_data.keywords,
        sections=sections
    )
    
    # Save publication
    manager.save_publication(publication.id)
    
    return publication_to_response(publication)

@app.get("/publications/{publication_id}", response_model=PublicationResponseModel)
def get_publication(publication_id: str):
    """
    Get a publication by ID.
    
    Args:
        publication_id: Publication ID
        
    Returns:
        Publication
    """
    manager = get_publication_manager()
    manager.load_all_publications()
    
    publication = manager.get_publication(publication_id)
    
    if publication is None:
        raise HTTPException(status_code=404, detail=f"Publication with ID {publication_id} not found")
    
    return publication_to_response(publication)

@app.put("/publications/{publication_id}", response_model=PublicationResponseModel)
def update_publication(publication_id: str, publication_data: PublicationUpdateModel):
    """
    Update a publication.
    
    Args:
        publication_id: Publication ID
        publication_data: Publication data
        
    Returns:
        Updated publication
    """
    manager = get_publication_manager()
    manager.load_all_publications()
    
    publication = manager.get_publication(publication_id)
    
    if publication is None:
        raise HTTPException(status_code=404, detail=f"Publication with ID {publication_id} not found")
    
    # Update fields
    if publication_data.title is not None:
        publication.title = publication_data.title
    
    if publication_data.abstract is not None:
        publication.abstract = publication_data.abstract
    
    if publication_data.keywords is not None:
        publication.keywords = publication_data.keywords
    
    if publication_data.acknowledgments is not None:
        publication.acknowledgments = publication_data.acknowledgments
    
    if publication_data.funding is not None:
        publication.funding = publication_data.funding
    
    if publication_data.conflicts_of_interest is not None:
        publication.conflicts_of_interest = publication_data.conflicts_of_interest
    
    # Update timestamp
    publication.updated_at = datetime.datetime.now()
    
    # Save publication
    manager.save_publication(publication.id)
    
    return publication_to_response(publication)

@app.delete("/publications/{publication_id}")
def delete_publication(publication_id: str):
    """
    Delete a publication.
    
    Args:
        publication_id: Publication ID
        
    Returns:
        Success message
    """
    manager = get_publication_manager()
    manager.load_all_publications()
    
    success = manager.delete_publication(publication_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Publication with ID {publication_id} not found")
    
    return {"message": f"Publication with ID {publication_id} deleted"}

@app.post("/publications/{publication_id}/sections", response_model=PublicationResponseModel)
def add_section(publication_id: str, section_data: SectionModel):
    """
    Add a section to a publication.
    
    Args:
        publication_id: Publication ID
        section_data: Section data
        
    Returns:
        Updated publication
    """
    manager = get_publication_manager()
    manager.load_all_publications()
    
    publication = manager.get_publication(publication_id)
    
    if publication is None:
        raise HTTPException(status_code=404, detail=f"Publication with ID {publication_id} not found")
    
    # Create section
    section = Section(
        title=section_data.title,
        content=section_data.content,
        level=section_data.level,
        id=section_data.id
    )
    
    # Add section to publication
    publication.add_section(section)
    
    # Save publication
    manager.save_publication(publication.id)
    
    return publication_to_response(publication)

@app.post("/publications/{publication_id}/references", response_model=PublicationResponseModel)
def add_reference(publication_id: str, reference_data: ReferenceModel):
    """
    Add a reference to a publication.
    
    Args:
        publication_id: Publication ID
        reference_data: Reference data
        
    Returns:
        Updated publication
    """
    manager = get_publication_manager()
    manager.load_all_publications()
    
    publication = manager.get_publication(publication_id)
    
    if publication is None:
        raise HTTPException(status_code=404, detail=f"Publication with ID {publication_id} not found")
    
    # Create reference
    reference = Reference(
        id=reference_data.id,
        type=reference_data.type,
        title=reference_data.title,
        authors=reference_data.authors,
        year=reference_data.year,
        source=reference_data.source,
        doi=reference_data.doi,
        url=reference_data.url,
        volume=reference_data.volume,
        issue=reference_data.issue,
        pages=reference_data.pages,
        publisher=reference_data.publisher,
        abstract=reference_data.abstract,
        keywords=reference_data.keywords
    )
    
    # Add reference to publication
    publication.add_reference(reference)
    
    # Save publication
    manager.save_publication(publication.id)
    
    return publication_to_response(publication)

@app.post("/publications/{publication_id}/figures", response_model=PublicationResponseModel)
async def add_figure(
    publication_id: str,
    title: str,
    caption: str,
    figure_id: str,
    figure_type: str = "image",
    width: Optional[str] = None,
    height: Optional[str] = None,
    source: Optional[str] = None,
    notes: Optional[str] = None,
    file: UploadFile = File(...)
):
    """
    Add a figure to a publication.
    
    Args:
        publication_id: Publication ID
        title: Figure title
        caption: Figure caption
        figure_id: Figure ID
        figure_type: Figure type
        width: Figure width
        height: Figure height
        source: Figure source
        notes: Figure notes
        file: Figure file
        
    Returns:
        Updated publication
    """
    manager = get_publication_manager()
    manager.load_all_publications()
    
    publication = manager.get_publication(publication_id)
    
    if publication is None:
        raise HTTPException(status_code=404, detail=f"Publication with ID {publication_id} not found")
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(manager.storage_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save figure file
    figure_filename = f"{publication_id}_{figure_id}_{file.filename}"
    figure_path = os.path.join(figures_dir, figure_filename)
    
    with open(figure_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create figure
    figure = Figure(
        id=figure_id,
        title=title,
        caption=caption,
        path=figure_path,
        type=figure_type,
        width=width,
        height=height,
        source=source,
        notes=notes
    )
    
    # Add figure to publication
    publication.add_figure(figure)
    
    # Save publication
    manager.save_publication(publication.id)
    
    return publication_to_response(publication)

@app.post("/publications/{publication_id}/tables", response_model=PublicationResponseModel)
async def add_table(
    publication_id: str,
    table_data: TableModel
):
    """
    Add a table to a publication.
    
    Args:
        publication_id: Publication ID
        table_data: Table data
        
    Returns:
        Updated publication
    """
    manager = get_publication_manager()
    manager.load_all_publications()
    
    publication = manager.get_publication(publication_id)
    
    if publication is None:
        raise HTTPException(status_code=404, detail=f"Publication with ID {publication_id} not found")
    
    # Create table
    table = Table(
        id=table_data.id,
        title=table_data.title,
        caption=table_data.caption,
        data=table_data.data,
        header=table_data.header,
        footer=table_data.footer,
        notes=table_data.notes
    )
    
    # Add table to publication
    publication.add_table(table)
    
    # Save publication
    manager.save_publication(publication.id)
    
    return publication_to_response(publication)

@app.post("/publications/{publication_id}/export")
def export_publication(
    publication_id: str,
    export_data: PublicationExportModel,
    background_tasks: BackgroundTasks
):
    """
    Export a publication to the specified format.
    
    Args:
        publication_id: Publication ID
        export_data: Export data
        background_tasks: Background tasks
        
    Returns:
        Path to the exported file
    """
    manager = get_publication_manager()
    manager.load_all_publications()
    
    publication = manager.get_publication(publication_id)
    
    if publication is None:
        raise HTTPException(status_code=404, detail=f"Publication with ID {publication_id} not found")
    
    # Parse format
    try:
        format = PublicationFormat(export_data.format)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format: {export_data.format}. Valid formats: {', '.join([f.value for f in PublicationFormat])}"
        )
    
    # Parse citation style
    try:
        citation_style = CitationStyle(export_data.citation_style)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid citation style: {export_data.citation_style}. Valid citation styles: {', '.join([s.value for s in CitationStyle])}"
        )
    
    # Parse journal template
    try:
        journal_template = JournalTemplate(export_data.journal_template)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid journal template: {export_data.journal_template}. Valid journal templates: {', '.join([t.value for t in JournalTemplate])}"
        )
    
    # Create exports directory if it doesn't exist
    exports_dir = os.path.join(manager.storage_dir, "exports")
    os.makedirs(exports_dir, exist_ok=True)
    
    # Export publication
    filepath = manager.export_publication(
        publication_id=publication_id,
        format=format,
        output_dir=exports_dir,
        citation_style=citation_style,
        journal_template=journal_template
    )
    
    if not filepath:
        raise HTTPException(status_code=500, detail=f"Failed to export publication '{publication.title}'")
    
    # Clean up old exports in the background
    background_tasks.add_task(clean_old_exports, exports_dir)
    
    return FileResponse(filepath, filename=os.path.basename(filepath))

@app.post("/publications/from-experiment", response_model=PublicationResponseModel)
def create_publication_from_experiment(publication_data: PublicationFromExperimentModel):
    """
    Create a publication from an experiment.
    
    Args:
        publication_data: Publication data
        
    Returns:
        Created publication
    """
    manager = get_publication_manager()
    
    # Convert authors
    authors = None
    if publication_data.authors:
        authors = [
            Author(
                name=author.name,
                email=author.email,
                affiliation=author.affiliation,
                orcid=author.orcid,
                is_corresponding=author.is_corresponding
            )
            for author in publication_data.authors
        ]
    
    # Create publication from experiment
    publication = manager.create_publication_from_experiment(
        experiment_id=publication_data.experiment_id,
        title=publication_data.title,
        authors=authors,
        abstract=publication_data.abstract,
        keywords=publication_data.keywords
    )
    
    if publication is None:
        raise HTTPException(status_code=404, detail=f"Experiment with ID {publication_data.experiment_id} not found")
    
    return publication_to_response(publication)

def clean_old_exports(exports_dir: str, max_age_days: int = 7):
    """
    Clean up old exports.
    
    Args:
        exports_dir: Exports directory
        max_age_days: Maximum age of exports in days
    """
    now = datetime.datetime.now()
    max_age = datetime.timedelta(days=max_age_days)
    
    for filename in os.listdir(exports_dir):
        filepath = os.path.join(exports_dir, filename)
        
        if os.path.isfile(filepath):
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
            age = now - mtime
            
            if age > max_age:
                os.remove(filepath)
