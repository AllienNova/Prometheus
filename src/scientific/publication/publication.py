"""
Research publication module for the Prometheus AI Automation Platform.

This module provides classes and functions for preparing, formatting, and publishing
scientific research findings in various formats and styles.
"""

import os
import logging
import json
import datetime
import re
import uuid
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
import copy
import tempfile
import subprocess

# Configure logging
logger = logging.getLogger(__name__)

class PublicationFormat(Enum):
    """Format for research publication."""
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    JUPYTER = "jupyter"

class CitationStyle(Enum):
    """Citation style for references."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    NATURE = "nature"
    SCIENCE = "science"
    CELL = "cell"

class JournalTemplate(Enum):
    """Journal-specific templates."""
    GENERIC = "generic"
    NATURE = "nature"
    SCIENCE = "science"
    CELL = "cell"
    PLOS = "plos"
    FRONTIERS = "frontiers"
    IEEE = "ieee"
    ACM = "acm"
    ARXIV = "arxiv"

@dataclass
class Author:
    """Author information for a publication."""
    name: str
    email: Optional[str] = None
    affiliation: Optional[str] = None
    orcid: Optional[str] = None
    is_corresponding: bool = False

@dataclass
class Reference:
    """Reference information for a publication."""
    id: str
    type: str  # article, book, conference, etc.
    title: str
    authors: List[str]
    year: int
    source: str  # journal name, book publisher, etc.
    doi: Optional[str] = None
    url: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    def format_citation(self, style: CitationStyle) -> str:
        """
        Format the reference according to the specified citation style.
        
        Args:
            style: Citation style to use
            
        Returns:
            Formatted citation string
        """
        if style == CitationStyle.APA:
            # APA style
            authors_str = ", ".join(self.authors[:-1])
            if len(self.authors) > 1:
                authors_str += f", & {self.authors[-1]}"
            else:
                authors_str = self.authors[0]
            
            if self.type == "article":
                return f"{authors_str}. ({self.year}). {self.title}. {self.source}, {self.volume}({self.issue}), {self.pages}. {f'https://doi.org/{self.doi}' if self.doi else ''}"
            elif self.type == "book":
                return f"{authors_str}. ({self.year}). {self.title}. {self.publisher}."
            else:
                return f"{authors_str}. ({self.year}). {self.title}. {self.source}."
        
        elif style == CitationStyle.MLA:
            # MLA style
            authors_str = ", ".join(self.authors[:-1])
            if len(self.authors) > 1:
                authors_str += f", and {self.authors[-1]}"
            else:
                authors_str = self.authors[0]
            
            if self.type == "article":
                return f"{authors_str}. \"{self.title}.\" {self.source}, vol. {self.volume}, no. {self.issue}, {self.year}, pp. {self.pages}."
            elif self.type == "book":
                return f"{authors_str}. {self.title}. {self.publisher}, {self.year}."
            else:
                return f"{authors_str}. \"{self.title}.\" {self.source}, {self.year}."
        
        elif style == CitationStyle.CHICAGO:
            # Chicago style
            authors_str = ", ".join(self.authors[:-1])
            if len(self.authors) > 1:
                authors_str += f", and {self.authors[-1]}"
            else:
                authors_str = self.authors[0]
            
            if self.type == "article":
                return f"{authors_str}. \"{self.title}.\" {self.source} {self.volume}, no. {self.issue} ({self.year}): {self.pages}."
            elif self.type == "book":
                return f"{authors_str}. {self.title}. {self.publisher}, {self.year}."
            else:
                return f"{authors_str}. \"{self.title}.\" {self.source}, {self.year}."
        
        elif style == CitationStyle.IEEE:
            # IEEE style
            if len(self.authors) > 3:
                authors_str = f"{self.authors[0]} et al."
            else:
                authors_str = " and ".join(self.authors)
            
            if self.type == "article":
                return f"{authors_str}, \"{self.title},\" {self.source}, vol. {self.volume}, no. {self.issue}, pp. {self.pages}, {self.year}."
            elif self.type == "book":
                return f"{authors_str}, {self.title}. {self.publisher}, {self.year}."
            else:
                return f"{authors_str}, \"{self.title},\" {self.source}, {self.year}."
        
        elif style == CitationStyle.HARVARD:
            # Harvard style
            if len(self.authors) > 3:
                authors_str = f"{self.authors[0]} et al."
            else:
                authors_str = " and ".join(self.authors)
            
            if self.type == "article":
                return f"{authors_str} ({self.year}). {self.title}. {self.source}, {self.volume}({self.issue}), pp. {self.pages}."
            elif self.type == "book":
                return f"{authors_str} ({self.year}). {self.title}. {self.publisher}."
            else:
                return f"{authors_str} ({self.year}). {self.title}. {self.source}."
        
        elif style == CitationStyle.NATURE:
            # Nature style
            if len(self.authors) > 5:
                authors_str = f"{self.authors[0]} et al."
            else:
                authors_str = ", ".join(self.authors)
            
            if self.type == "article":
                return f"{authors_str} {self.title}. {self.source} {self.volume}, {self.pages} ({self.year})."
            elif self.type == "book":
                return f"{authors_str} {self.title}. ({self.publisher}, {self.year})."
            else:
                return f"{authors_str} {self.title}. {self.source} ({self.year})."
        
        elif style == CitationStyle.SCIENCE:
            # Science style
            if len(self.authors) > 5:
                authors_str = f"{self.authors[0]} et al."
            else:
                authors_str = ", ".join(self.authors)
            
            if self.type == "article":
                return f"{authors_str}, {self.title}. {self.source} {self.volume}, {self.pages} ({self.year})."
            elif self.type == "book":
                return f"{authors_str}, {self.title} ({self.publisher}, {self.year})."
            else:
                return f"{authors_str}, {self.title}. {self.source} ({self.year})."
        
        elif style == CitationStyle.CELL:
            # Cell style
            if len(self.authors) > 10:
                authors_str = ", ".join(self.authors[:10]) + ", et al."
            else:
                authors_str = ", ".join(self.authors)
            
            if self.type == "article":
                return f"{authors_str} ({self.year}). {self.title}. {self.source} {self.volume}, {self.pages}."
            elif self.type == "book":
                return f"{authors_str} ({self.year}). {self.title} ({self.publisher})."
            else:
                return f"{authors_str} ({self.year}). {self.title}. {self.source}."
        
        else:
            # Default to APA
            return self.format_citation(CitationStyle.APA)

@dataclass
class Figure:
    """Figure information for a publication."""
    id: str
    title: str
    caption: str
    path: str
    type: str = "image"  # image, chart, diagram, etc.
    width: Optional[str] = None
    height: Optional[str] = None
    source: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class Table:
    """Table information for a publication."""
    id: str
    title: str
    caption: str
    data: Union[List[List[str]], str]  # Either a 2D array or a path to a CSV file
    header: Optional[List[str]] = None
    footer: Optional[str] = None
    notes: Optional[str] = None

@dataclass
class Section:
    """Section information for a publication."""
    title: str
    content: str
    level: int = 1  # 1 = main section, 2 = subsection, etc.
    id: Optional[str] = None

@dataclass
class Publication:
    """
    Scientific publication for the Prometheus AI Automation Platform.
    
    This class represents a scientific publication with all its components,
    including metadata, content, figures, tables, and references.
    """
    title: str
    authors: List[Author]
    abstract: str
    keywords: List[str]
    sections: List[Section]
    references: List[Reference] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    journal: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[datetime.datetime] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    acknowledgments: Optional[str] = None
    funding: Optional[str] = None
    conflicts_of_interest: Optional[str] = None
    supplementary_materials: List[str] = field(default_factory=list)
    
    def add_section(self, section: Section) -> None:
        """
        Add a section to the publication.
        
        Args:
            section: Section to add
        """
        if section.id is None:
            section.id = f"section-{len(self.sections) + 1}"
        
        self.sections.append(section)
        self.updated_at = datetime.datetime.now()
        
        logger.info(f"Added section '{section.title}' to publication '{self.title}'")
    
    def add_reference(self, reference: Reference) -> None:
        """
        Add a reference to the publication.
        
        Args:
            reference: Reference to add
        """
        self.references.append(reference)
        self.updated_at = datetime.datetime.now()
        
        logger.info(f"Added reference '{reference.id}' to publication '{self.title}'")
    
    def add_figure(self, figure: Figure) -> None:
        """
        Add a figure to the publication.
        
        Args:
            figure: Figure to add
        """
        self.figures.append(figure)
        self.updated_at = datetime.datetime.now()
        
        logger.info(f"Added figure '{figure.id}' to publication '{self.title}'")
    
    def add_table(self, table: Table) -> None:
        """
        Add a table to the publication.
        
        Args:
            table: Table to add
        """
        self.tables.append(table)
        self.updated_at = datetime.datetime.now()
        
        logger.info(f"Added table '{table.id}' to publication '{self.title}'")
    
    def get_corresponding_author(self) -> Optional[Author]:
        """
        Get the corresponding author of the publication.
        
        Returns:
            Corresponding author, or None if not specified
        """
        for author in self.authors:
            if author.is_corresponding:
                return author
        
        # If no corresponding author is specified, return the first author
        if self.authors:
            return self.authors[0]
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the publication to a dictionary.
        
        Returns:
            Dictionary representation of the publication
        """
        # Convert to dictionary using dataclasses.asdict
        result = asdict(self)
        
        # Convert datetime objects to ISO format strings
        for key in ['created_at', 'updated_at', 'publication_date']:
            if result[key] is not None:
                result[key] = result[key].isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Publication':
        """
        Create a publication from a dictionary.
        
        Args:
            data: Dictionary representation of the publication
            
        Returns:
            Publication object
        """
        # Create a copy of the data to avoid modifying the original
        data = copy.deepcopy(data)
        
        # Convert ISO format strings to datetime objects
        for key in ['created_at', 'updated_at', 'publication_date']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.datetime.fromisoformat(data[key])
        
        # Convert author dictionaries to objects
        authors = []
        for author_dict in data.get('authors', []):
            authors.append(Author(**author_dict))
        data['authors'] = authors
        
        # Convert section dictionaries to objects
        sections = []
        for section_dict in data.get('sections', []):
            sections.append(Section(**section_dict))
        data['sections'] = sections
        
        # Convert reference dictionaries to objects
        references = []
        for reference_dict in data.get('references', []):
            references.append(Reference(**reference_dict))
        data['references'] = references
        
        # Convert figure dictionaries to objects
        figures = []
        for figure_dict in data.get('figures', []):
            figures.append(Figure(**figure_dict))
        data['figures'] = figures
        
        # Convert table dictionaries to objects
        tables = []
        for table_dict in data.get('tables', []):
            tables.append(Table(**table_dict))
        data['tables'] = tables
        
        # Create publication object
        return cls(**data)
    
    def save(self, directory: str) -> str:
        """
        Save the publication to a file.
        
        Args:
            directory: Directory to save the publication
            
        Returns:
            Path to the saved publication file
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Convert to dictionary
        data = self.to_dict()
        
        # Generate filename
        filename = f"{self.id}.json"
        filepath = os.path.join(directory, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved publication '{self.title}' to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'Publication':
        """
        Load a publication from a file.
        
        Args:
            filepath: Path to the publication file
            
        Returns:
            Loaded publication
        """
        # Load from file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create publication object
        publication = cls.from_dict(data)
        
        logger.info(f"Loaded publication '{publication.title}' from {filepath}")
        return publication
    
    def export(self, format: PublicationFormat, output_dir: str, 
              citation_style: CitationStyle = CitationStyle.APA,
              journal_template: JournalTemplate = JournalTemplate.GENERIC) -> str:
        """
        Export the publication to the specified format.
        
        Args:
            format: Output format
            output_dir: Directory to save the exported file
            citation_style: Citation style to use
            journal_template: Journal template to use
            
        Returns:
            Path to the exported file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base filename
        base_filename = re.sub(r'[^\w\-\.]', '_', self.title.lower())
        
        if format == PublicationFormat.MARKDOWN:
            return self._export_markdown(output_dir, base_filename, citation_style)
        elif format == PublicationFormat.LATEX:
            return self._export_latex(output_dir, base_filename, citation_style, journal_template)
        elif format == PublicationFormat.HTML:
            return self._export_html(output_dir, base_filename, citation_style)
        elif format == PublicationFormat.PDF:
            # Export to LaTeX first, then convert to PDF
            latex_path = self._export_latex(output_dir, base_filename, citation_style, journal_template)
            return self._convert_latex_to_pdf(latex_path, output_dir)
        elif format == PublicationFormat.DOCX:
            # Export to Markdown first, then convert to DOCX
            md_path = self._export_markdown(output_dir, base_filename, citation_style)
            return self._convert_markdown_to_docx(md_path, output_dir)
        elif format == PublicationFormat.JUPYTER:
            return self._export_jupyter(output_dir, base_filename, citation_style)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_markdown(self, output_dir: str, base_filename: str, 
                        citation_style: CitationStyle) -> str:
        """
        Export the publication to Markdown format.
        
        Args:
            output_dir: Directory to save the exported file
            base_filename: Base filename for the exported file
            citation_style: Citation style to use
            
        Returns:
            Path to the exported file
        """
        # Generate filename
        filename = f"{base_filename}.md"
        filepath = os.path.join(output_dir, filename)
        
        # Create a figures directory if needed
        figures_dir = os.path.join(output_dir, "figures")
        if self.figures and not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        # Copy figures to the figures directory
        for figure in self.figures:
            if os.path.exists(figure.path):
                figure_filename = os.path.basename(figure.path)
                figure_dest = os.path.join(figures_dir, figure_filename)
                shutil.copy2(figure.path, figure_dest)
        
        # Generate Markdown content
        content = []
        
        # Title and authors
        content.append(f"# {self.title}\n")
        
        authors_str = ", ".join([author.name for author in self.authors])
        content.append(f"{authors_str}\n")
        
        # Affiliations
        affiliations = {}
        for i, author in enumerate(self.authors):
            if author.affiliation:
                if author.affiliation not in affiliations:
                    affiliations[author.affiliation] = []
                affiliations[author.affiliation].append(i + 1)
        
        if affiliations:
            for affiliation, indices in affiliations.items():
                indices_str = ", ".join([str(i) for i in indices])
                content.append(f"{indices_str}. {affiliation}")
            content.append("")
        
        # Corresponding author
        corresponding_author = self.get_corresponding_author()
        if corresponding_author and corresponding_author.email:
            content.append(f"Corresponding author: {corresponding_author.name} ({corresponding_author.email})\n")
        
        # Abstract
        content.append("## Abstract\n")
        content.append(f"{self.abstract}\n")
        
        # Keywords
        if self.keywords:
            content.append(f"**Keywords**: {', '.join(self.keywords)}\n")
        
        # Sections
        for section in self.sections:
            # Add appropriate heading level
            heading = "#" * section.level
            content.append(f"{heading} {section.title}\n")
            content.append(f"{section.content}\n")
        
        # Figures
        if self.figures:
            content.append("## Figures\n")
            for figure in self.figures:
                figure_filename = os.path.basename(figure.path)
                content.append(f"**Figure {figure.id}: {figure.title}**\n")
                content.append(f"![{figure.caption}](figures/{figure_filename})\n")
                content.append(f"*{figure.caption}*\n")
        
        # Tables
        if self.tables:
            content.append("## Tables\n")
            for table in self.tables:
                content.append(f"**Table {table.id}: {table.title}**\n")
                
                if isinstance(table.data, str):
                    # Data is a path to a CSV file
                    if os.path.exists(table.data):
                        with open(table.data, 'r') as f:
                            lines = f.readlines()
                        
                        # Parse CSV data
                        import csv
                        csv_data = list(csv.reader(lines))
                        
                        if table.header is None and csv_data:
                            table_header = csv_data[0]
                            table_data = csv_data[1:]
                        else:
                            table_header = table.header
                            table_data = csv_data
                    else:
                        table_header = table.header
                        table_data = [["Data file not found"]]
                else:
                    # Data is a 2D array
                    table_header = table.header
                    table_data = table.data
                
                # Generate Markdown table
                if table_header:
                    content.append("| " + " | ".join(table_header) + " |")
                    content.append("| " + " | ".join(["---"] * len(table_header)) + " |")
                
                for row in table_data:
                    content.append("| " + " | ".join([str(cell) for cell in row]) + " |")
                
                content.append("")
                content.append(f"*{table.caption}*\n")
        
        # Acknowledgments
        if self.acknowledgments:
            content.append("## Acknowledgments\n")
            content.append(f"{self.acknowledgments}\n")
        
        # Funding
        if self.funding:
            content.append("## Funding\n")
            content.append(f"{self.funding}\n")
        
        # Conflicts of interest
        if self.conflicts_of_interest:
            content.append("## Conflicts of Interest\n")
            content.append(f"{self.conflicts_of_interest}\n")
        
        # References
        if self.references:
            content.append("## References\n")
            for i, reference in enumerate(self.references):
                content.append(f"{i+1}. {reference.format_citation(citation_style)}\n")
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write("\n".join(content))
        
        logger.info(f"Exported publication '{self.title}' to Markdown: {filepath}")
        return filepath
    
    def _export_latex(self, output_dir: str, base_filename: str, 
                     citation_style: CitationStyle,
                     journal_template: JournalTemplate) -> str:
        """
        Export the publication to LaTeX format.
        
        Args:
            output_dir: Directory to save the exported file
            base_filename: Base filename for the exported file
            citation_style: Citation style to use
            journal_template: Journal template to use
            
        Returns:
            Path to the exported file
        """
        # Generate filename
        filename = f"{base_filename}.tex"
        filepath = os.path.join(output_dir, filename)
        
        # Create a figures directory if needed
        figures_dir = os.path.join(output_dir, "figures")
        if self.figures and not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        # Copy figures to the figures directory
        for figure in self.figures:
            if os.path.exists(figure.path):
                figure_filename = os.path.basename(figure.path)
                figure_dest = os.path.join(figures_dir, figure_filename)
                shutil.copy2(figure.path, figure_dest)
        
        # Generate LaTeX content
        content = []
        
        # Document class and packages
        if journal_template == JournalTemplate.NATURE:
            content.append("\\documentclass{nature}")
            content.append("\\usepackage{natbib}")
        elif journal_template == JournalTemplate.SCIENCE:
            content.append("\\documentclass{sciencemag}")
            content.append("\\usepackage{natbib}")
        elif journal_template == JournalTemplate.IEEE:
            content.append("\\documentclass[conference]{IEEEtran}")
            content.append("\\usepackage{cite}")
        elif journal_template == JournalTemplate.ACM:
            content.append("\\documentclass{acmart}")
            content.append("\\usepackage{natbib}")
        else:
            # Generic template
            content.append("\\documentclass{article}")
            content.append("\\usepackage{natbib}")
        
        # Common packages
        content.append("\\usepackage{graphicx}")
        content.append("\\usepackage{amsmath}")
        content.append("\\usepackage{booktabs}")
        content.append("\\usepackage{hyperref}")
        content.append("\\usepackage{url}")
        content.append("\\usepackage{float}")
        content.append("")
        
        # Document metadata
        content.append("\\title{" + self._escape_latex(self.title) + "}")
        
        # Authors and affiliations
        if journal_template in [JournalTemplate.NATURE, JournalTemplate.SCIENCE]:
            # Nature/Science style
            authors_str = " \\and ".join([self._escape_latex(author.name) for author in self.authors])
            content.append("\\author{" + authors_str + "}")
            
            # Affiliations as footnotes
            affiliations = {}
            for i, author in enumerate(self.authors):
                if author.affiliation:
                    if author.affiliation not in affiliations:
                        affiliations[author.affiliation] = []
                    affiliations[author.affiliation].append(i + 1)
            
            if affiliations:
                content.append("\\maketitle")
                content.append("\\begin{abstract}")
                content.append(self._escape_latex(self.abstract))
                content.append("\\end{abstract}")
                
                content.append("\\begin{center}")
                for affiliation, indices in affiliations.items():
                    indices_str = ", ".join([str(i) for i in indices])
                    content.append(f"$^{{{indices_str}}}$ {self._escape_latex(affiliation)} \\\\")
                content.append("\\end{center}")
        else:
            # Standard style
            content.append("\\author{")
            for i, author in enumerate(self.authors):
                if i > 0:
                    content.append(" \\and ")
                content.append(self._escape_latex(author.name))
                if author.affiliation:
                    content.append("\\\\ " + self._escape_latex(author.affiliation))
                if author.email:
                    content.append("\\\\ \\texttt{" + author.email + "}")
            content.append("}")
            
            content.append("\\begin{document}")
            content.append("\\maketitle")
            
            # Abstract
            content.append("\\begin{abstract}")
            content.append(self._escape_latex(self.abstract))
            content.append("\\end{abstract}")
        
        # Keywords
        if self.keywords:
            content.append("\\begin{keywords}")
            content.append(self._escape_latex(", ".join(self.keywords)))
            content.append("\\end{keywords}")
        
        # Sections
        for section in self.sections:
            if section.level == 1:
                content.append("\\section{" + self._escape_latex(section.title) + "}")
            elif section.level == 2:
                content.append("\\subsection{" + self._escape_latex(section.title) + "}")
            elif section.level == 3:
                content.append("\\subsubsection{" + self._escape_latex(section.title) + "}")
            else:
                content.append("\\paragraph{" + self._escape_latex(section.title) + "}")
            
            content.append(self._escape_latex(section.content))
            content.append("")
        
        # Figures
        for figure in self.figures:
            figure_filename = os.path.basename(figure.path)
            content.append("\\begin{figure}[htbp]")
            content.append("\\centering")
            
            # Figure width and height
            options = []
            if figure.width:
                options.append(f"width={figure.width}")
            if figure.height:
                options.append(f"height={figure.height}")
            
            options_str = ",".join(options)
            if options_str:
                options_str = f"[{options_str}]"
            
            content.append(f"\\includegraphics{options_str}{{figures/{figure_filename}}}")
            content.append("\\caption{" + self._escape_latex(figure.caption) + "}")
            content.append("\\label{fig:" + figure.id + "}")
            content.append("\\end{figure}")
            content.append("")
        
        # Tables
        for table in self.tables:
            content.append("\\begin{table}[htbp]")
            content.append("\\centering")
            content.append("\\caption{" + self._escape_latex(table.caption) + "}")
            content.append("\\label{tab:" + table.id + "}")
            
            if isinstance(table.data, str):
                # Data is a path to a CSV file
                if os.path.exists(table.data):
                    with open(table.data, 'r') as f:
                        lines = f.readlines()
                    
                    # Parse CSV data
                    import csv
                    csv_data = list(csv.reader(lines))
                    
                    if table.header is None and csv_data:
                        table_header = csv_data[0]
                        table_data = csv_data[1:]
                    else:
                        table_header = table.header
                        table_data = csv_data
                else:
                    table_header = table.header
                    table_data = [["Data file not found"]]
            else:
                # Data is a 2D array
                table_header = table.header
                table_data = table.data
            
            # Generate LaTeX table
            if table_data:
                num_cols = len(table_data[0])
                content.append("\\begin{tabular}{" + "c" * num_cols + "}")
                content.append("\\toprule")
                
                if table_header:
                    content.append(" & ".join([self._escape_latex(str(cell)) for cell in table_header]) + " \\\\")
                    content.append("\\midrule")
                
                for row in table_data:
                    content.append(" & ".join([self._escape_latex(str(cell)) for cell in row]) + " \\\\")
                
                content.append("\\bottomrule")
                content.append("\\end{tabular}")
            
            content.append("\\end{table}")
            content.append("")
        
        # Acknowledgments
        if self.acknowledgments:
            content.append("\\section*{Acknowledgments}")
            content.append(self._escape_latex(self.acknowledgments))
            content.append("")
        
        # Funding
        if self.funding:
            content.append("\\section*{Funding}")
            content.append(self._escape_latex(self.funding))
            content.append("")
        
        # Conflicts of interest
        if self.conflicts_of_interest:
            content.append("\\section*{Conflicts of Interest}")
            content.append(self._escape_latex(self.conflicts_of_interest))
            content.append("")
        
        # References
        if self.references:
            # Create BibTeX file
            bib_filename = f"{base_filename}.bib"
            bib_filepath = os.path.join(output_dir, bib_filename)
            
            with open(bib_filepath, 'w') as f:
                for reference in self.references:
                    f.write(self._format_bibtex_entry(reference))
                    f.write("\n\n")
            
            # Add bibliography to LaTeX
            if citation_style == CitationStyle.APA:
                content.append("\\bibliographystyle{apalike}")
            elif citation_style == CitationStyle.IEEE:
                content.append("\\bibliographystyle{IEEEtran}")
            elif citation_style == CitationStyle.NATURE:
                content.append("\\bibliographystyle{naturemag}")
            elif citation_style == CitationStyle.SCIENCE:
                content.append("\\bibliographystyle{Science}")
            else:
                content.append("\\bibliographystyle{plain}")
            
            content.append("\\bibliography{" + base_filename + "}")
        
        # End document
        content.append("\\end{document}")
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write("\n".join(content))
        
        logger.info(f"Exported publication '{self.title}' to LaTeX: {filepath}")
        return filepath
    
    def _export_html(self, output_dir: str, base_filename: str, 
                    citation_style: CitationStyle) -> str:
        """
        Export the publication to HTML format.
        
        Args:
            output_dir: Directory to save the exported file
            base_filename: Base filename for the exported file
            citation_style: Citation style to use
            
        Returns:
            Path to the exported file
        """
        # Generate filename
        filename = f"{base_filename}.html"
        filepath = os.path.join(output_dir, filename)
        
        # Create a figures directory if needed
        figures_dir = os.path.join(output_dir, "figures")
        if self.figures and not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        # Copy figures to the figures directory
        for figure in self.figures:
            if os.path.exists(figure.path):
                figure_filename = os.path.basename(figure.path)
                figure_dest = os.path.join(figures_dir, figure_filename)
                shutil.copy2(figure.path, figure_dest)
        
        # Generate HTML content
        content = []
        
        # HTML header
        content.append("<!DOCTYPE html>")
        content.append("<html lang=\"en\">")
        content.append("<head>")
        content.append("    <meta charset=\"UTF-8\">")
        content.append("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">")
        content.append(f"    <title>{self._escape_html(self.title)}</title>")
        content.append("    <style>")
        content.append("        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }")
        content.append("        h1 { text-align: center; }")
        content.append("        .authors { text-align: center; margin-bottom: 20px; }")
        content.append("        .abstract { font-style: italic; margin-bottom: 20px; }")
        content.append("        .keywords { margin-bottom: 20px; }")
        content.append("        figure { text-align: center; margin: 20px 0; }")
        content.append("        figure img { max-width: 100%; }")
        content.append("        figcaption { font-style: italic; }")
        content.append("        table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        content.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        content.append("        th { background-color: #f2f2f2; }")
        content.append("        .references { margin-top: 40px; }")
        content.append("        .reference { margin-bottom: 10px; text-indent: -2em; padding-left: 2em; }")
        content.append("    </style>")
        content.append("</head>")
        content.append("<body>")
        
        # Title and authors
        content.append(f"    <h1>{self._escape_html(self.title)}</h1>")
        
        content.append("    <div class=\"authors\">")
        for author in self.authors:
            author_html = self._escape_html(author.name)
            if author.affiliation:
                author_html += f"<sup>{self.authors.index(author) + 1}</sup>"
            content.append(f"        <span>{author_html}</span>")
            if author != self.authors[-1]:
                content.append(", ")
        content.append("    </div>")
        
        # Affiliations
        affiliations = {}
        for i, author in enumerate(self.authors):
            if author.affiliation:
                if author.affiliation not in affiliations:
                    affiliations[author.affiliation] = []
                affiliations[author.affiliation].append(i + 1)
        
        if affiliations:
            content.append("    <div class=\"affiliations\">")
            for affiliation, indices in affiliations.items():
                indices_str = ", ".join([str(i) for i in indices])
                content.append(f"        <div><sup>{indices_str}</sup> {self._escape_html(affiliation)}</div>")
            content.append("    </div>")
        
        # Corresponding author
        corresponding_author = self.get_corresponding_author()
        if corresponding_author and corresponding_author.email:
            content.append("    <div class=\"corresponding\">")
            content.append(f"        Corresponding author: {self._escape_html(corresponding_author.name)} ({corresponding_author.email})")
            content.append("    </div>")
        
        # Abstract
        content.append("    <div class=\"abstract\">")
        content.append("        <h2>Abstract</h2>")
        content.append(f"        <p>{self._escape_html(self.abstract)}</p>")
        content.append("    </div>")
        
        # Keywords
        if self.keywords:
            content.append("    <div class=\"keywords\">")
            content.append(f"        <strong>Keywords:</strong> {self._escape_html(', '.join(self.keywords))}")
            content.append("    </div>")
        
        # Sections
        for section in self.sections:
            heading_level = min(section.level + 1, 6)  # h2 to h6
            content.append(f"    <h{heading_level} id=\"{section.id}\">{self._escape_html(section.title)}</h{heading_level}>")
            
            # Split content into paragraphs
            paragraphs = section.content.split("\n\n")
            for paragraph in paragraphs:
                content.append(f"    <p>{self._escape_html(paragraph)}</p>")
        
        # Figures
        if self.figures:
            content.append("    <div class=\"figures\">")
            content.append("        <h2>Figures</h2>")
            for figure in self.figures:
                figure_filename = os.path.basename(figure.path)
                content.append(f"        <figure id=\"figure-{figure.id}\">")
                content.append(f"            <img src=\"figures/{figure_filename}\" alt=\"{self._escape_html(figure.caption)}\">")
                content.append(f"            <figcaption><strong>Figure {figure.id}:</strong> {self._escape_html(figure.caption)}</figcaption>")
                content.append("        </figure>")
            content.append("    </div>")
        
        # Tables
        if self.tables:
            content.append("    <div class=\"tables\">")
            content.append("        <h2>Tables</h2>")
            for table in self.tables:
                content.append(f"        <div id=\"table-{table.id}\">")
                content.append(f"            <h3>Table {table.id}: {self._escape_html(table.title)}</h3>")
                
                if isinstance(table.data, str):
                    # Data is a path to a CSV file
                    if os.path.exists(table.data):
                        with open(table.data, 'r') as f:
                            lines = f.readlines()
                        
                        # Parse CSV data
                        import csv
                        csv_data = list(csv.reader(lines))
                        
                        if table.header is None and csv_data:
                            table_header = csv_data[0]
                            table_data = csv_data[1:]
                        else:
                            table_header = table.header
                            table_data = csv_data
                    else:
                        table_header = table.header
                        table_data = [["Data file not found"]]
                else:
                    # Data is a 2D array
                    table_header = table.header
                    table_data = table.data
                
                # Generate HTML table
                content.append("            <table>")
                
                if table_header:
                    content.append("                <thead>")
                    content.append("                    <tr>")
                    for cell in table_header:
                        content.append(f"                        <th>{self._escape_html(str(cell))}</th>")
                    content.append("                    </tr>")
                    content.append("                </thead>")
                
                content.append("                <tbody>")
                for row in table_data:
                    content.append("                    <tr>")
                    for cell in row:
                        content.append(f"                        <td>{self._escape_html(str(cell))}</td>")
                    content.append("                    </tr>")
                content.append("                </tbody>")
                
                content.append("            </table>")
                content.append(f"            <p><em>{self._escape_html(table.caption)}</em></p>")
                content.append("        </div>")
            content.append("    </div>")
        
        # Acknowledgments
        if self.acknowledgments:
            content.append("    <div class=\"acknowledgments\">")
            content.append("        <h2>Acknowledgments</h2>")
            content.append(f"        <p>{self._escape_html(self.acknowledgments)}</p>")
            content.append("    </div>")
        
        # Funding
        if self.funding:
            content.append("    <div class=\"funding\">")
            content.append("        <h2>Funding</h2>")
            content.append(f"        <p>{self._escape_html(self.funding)}</p>")
            content.append("    </div>")
        
        # Conflicts of interest
        if self.conflicts_of_interest:
            content.append("    <div class=\"conflicts\">")
            content.append("        <h2>Conflicts of Interest</h2>")
            content.append(f"        <p>{self._escape_html(self.conflicts_of_interest)}</p>")
            content.append("    </div>")
        
        # References
        if self.references:
            content.append("    <div class=\"references\">")
            content.append("        <h2>References</h2>")
            for i, reference in enumerate(self.references):
                citation = reference.format_citation(citation_style)
                content.append(f"        <div class=\"reference\">{i+1}. {self._escape_html(citation)}</div>")
            content.append("    </div>")
        
        # HTML footer
        content.append("</body>")
        content.append("</html>")
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write("\n".join(content))
        
        logger.info(f"Exported publication '{self.title}' to HTML: {filepath}")
        return filepath
    
    def _export_jupyter(self, output_dir: str, base_filename: str, 
                       citation_style: CitationStyle) -> str:
        """
        Export the publication to Jupyter Notebook format.
        
        Args:
            output_dir: Directory to save the exported file
            base_filename: Base filename for the exported file
            citation_style: Citation style to use
            
        Returns:
            Path to the exported file
        """
        # Generate filename
        filename = f"{base_filename}.ipynb"
        filepath = os.path.join(output_dir, filename)
        
        # Create a figures directory if needed
        figures_dir = os.path.join(output_dir, "figures")
        if self.figures and not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        # Copy figures to the figures directory
        for figure in self.figures:
            if os.path.exists(figure.path):
                figure_filename = os.path.basename(figure.path)
                figure_dest = os.path.join(figures_dir, figure_filename)
                shutil.copy2(figure.path, figure_dest)
        
        # Generate Jupyter Notebook content
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Title and authors
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {self.title}\n",
                "\n",
                ", ".join([author.name for author in self.authors]) + "\n",
                "\n"
            ]
        })
        
        # Affiliations
        affiliations = {}
        for i, author in enumerate(self.authors):
            if author.affiliation:
                if author.affiliation not in affiliations:
                    affiliations[author.affiliation] = []
                affiliations[author.affiliation].append(i + 1)
        
        if affiliations:
            affiliation_text = []
            for affiliation, indices in affiliations.items():
                indices_str = ", ".join([str(i) for i in indices])
                affiliation_text.append(f"{indices_str}. {affiliation}")
            
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": affiliation_text + ["\n"]
            })
        
        # Corresponding author
        corresponding_author = self.get_corresponding_author()
        if corresponding_author and corresponding_author.email:
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"Corresponding author: {corresponding_author.name} ({corresponding_author.email})\n",
                    "\n"
                ]
            })
        
        # Abstract
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Abstract\n",
                "\n",
                f"{self.abstract}\n",
                "\n"
            ]
        })
        
        # Keywords
        if self.keywords:
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"**Keywords**: {', '.join(self.keywords)}\n",
                    "\n"
                ]
            })
        
        # Sections
        for section in self.sections:
            # Add appropriate heading level
            heading = "#" * (section.level + 1)
            
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"{heading} {section.title}\n",
                    "\n",
                    f"{section.content}\n",
                    "\n"
                ]
            })
        
        # Figures
        if self.figures:
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Figures\n",
                    "\n"
                ]
            })
            
            for figure in self.figures:
                figure_filename = os.path.basename(figure.path)
                
                notebook["cells"].append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"**Figure {figure.id}: {figure.title}**\n",
                        "\n"
                    ]
                })
                
                # Add code cell to display the figure
                notebook["cells"].append({
                    "cell_type": "code",
                    "metadata": {},
                    "source": [
                        "from IPython.display import Image\n",
                        f"Image(filename='figures/{figure_filename}')"
                    ],
                    "execution_count": None,
                    "outputs": []
                })
                
                notebook["cells"].append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"*{figure.caption}*\n",
                        "\n"
                    ]
                })
        
        # Tables
        if self.tables:
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Tables\n",
                    "\n"
                ]
            })
            
            for table in self.tables:
                notebook["cells"].append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"**Table {table.id}: {table.title}**\n",
                        "\n"
                    ]
                })
                
                # Add code cell to display the table
                code_source = ["import pandas as pd\n"]
                
                if isinstance(table.data, str):
                    # Data is a path to a CSV file
                    if os.path.exists(table.data):
                        code_source.append(f"df = pd.read_csv('{table.data}')\n")
                    else:
                        code_source.append("df = pd.DataFrame([['Data file not found']])\n")
                else:
                    # Data is a 2D array
                    if table.header:
                        code_source.append(f"df = pd.DataFrame({table.data}, columns={table.header})\n")
                    else:
                        code_source.append(f"df = pd.DataFrame({table.data})\n")
                
                code_source.append("df")
                
                notebook["cells"].append({
                    "cell_type": "code",
                    "metadata": {},
                    "source": code_source,
                    "execution_count": None,
                    "outputs": []
                })
                
                notebook["cells"].append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"*{table.caption}*\n",
                        "\n"
                    ]
                })
        
        # Acknowledgments
        if self.acknowledgments:
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Acknowledgments\n",
                    "\n",
                    f"{self.acknowledgments}\n",
                    "\n"
                ]
            })
        
        # Funding
        if self.funding:
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Funding\n",
                    "\n",
                    f"{self.funding}\n",
                    "\n"
                ]
            })
        
        # Conflicts of interest
        if self.conflicts_of_interest:
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Conflicts of Interest\n",
                    "\n",
                    f"{self.conflicts_of_interest}\n",
                    "\n"
                ]
            })
        
        # References
        if self.references:
            reference_text = ["## References\n", "\n"]
            for i, reference in enumerate(self.references):
                reference_text.append(f"{i+1}. {reference.format_citation(citation_style)}\n")
            
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": reference_text
            })
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(notebook, f, indent=2)
        
        logger.info(f"Exported publication '{self.title}' to Jupyter Notebook: {filepath}")
        return filepath
    
    def _convert_latex_to_pdf(self, latex_path: str, output_dir: str) -> str:
        """
        Convert LaTeX to PDF using pdflatex.
        
        Args:
            latex_path: Path to the LaTeX file
            output_dir: Directory to save the PDF file
            
        Returns:
            Path to the PDF file
        """
        # Check if pdflatex is installed
        try:
            subprocess.run(["pdflatex", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("pdflatex is not installed. Cannot convert to PDF.")
            return latex_path
        
        # Get the base filename
        base_filename = os.path.splitext(os.path.basename(latex_path))[0]
        
        # Create a temporary directory for LaTeX compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy LaTeX file and figures to temp directory
            shutil.copy2(latex_path, os.path.join(temp_dir, os.path.basename(latex_path)))
            
            # Copy BibTeX file if it exists
            bib_path = os.path.join(os.path.dirname(latex_path), f"{base_filename}.bib")
            if os.path.exists(bib_path):
                shutil.copy2(bib_path, os.path.join(temp_dir, os.path.basename(bib_path)))
            
            # Copy figures directory if it exists
            figures_dir = os.path.join(os.path.dirname(latex_path), "figures")
            if os.path.exists(figures_dir):
                shutil.copytree(figures_dir, os.path.join(temp_dir, "figures"))
            
            # Run pdflatex
            try:
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", os.path.basename(latex_path)],
                    cwd=temp_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                # Run bibtex if references exist
                if os.path.exists(os.path.join(temp_dir, f"{base_filename}.bib")):
                    subprocess.run(
                        ["bibtex", base_filename],
                        cwd=temp_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    
                    # Run pdflatex twice more to resolve references
                    subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", os.path.basename(latex_path)],
                        cwd=temp_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    
                    subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", os.path.basename(latex_path)],
                        cwd=temp_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                
                # Copy PDF to output directory
                pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
                shutil.copy2(os.path.join(temp_dir, f"{base_filename}.pdf"), pdf_path)
                
                logger.info(f"Converted LaTeX to PDF: {pdf_path}")
                return pdf_path
            
            except subprocess.SubprocessError as e:
                logger.error(f"Error converting LaTeX to PDF: {e}")
                return latex_path
    
    def _convert_markdown_to_docx(self, md_path: str, output_dir: str) -> str:
        """
        Convert Markdown to DOCX using pandoc.
        
        Args:
            md_path: Path to the Markdown file
            output_dir: Directory to save the DOCX file
            
        Returns:
            Path to the DOCX file
        """
        # Check if pandoc is installed
        try:
            subprocess.run(["pandoc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("pandoc is not installed. Cannot convert to DOCX.")
            return md_path
        
        # Get the base filename
        base_filename = os.path.splitext(os.path.basename(md_path))[0]
        
        # Generate output path
        docx_path = os.path.join(output_dir, f"{base_filename}.docx")
        
        # Run pandoc
        try:
            subprocess.run(
                ["pandoc", md_path, "-o", docx_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            logger.info(f"Converted Markdown to DOCX: {docx_path}")
            return docx_path
        
        except subprocess.SubprocessError as e:
            logger.error(f"Error converting Markdown to DOCX: {e}")
            return md_path
    
    def _escape_latex(self, text: str) -> str:
        """
        Escape special characters in LaTeX.
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text
        """
        # Define LaTeX special characters and their escaped versions
        latex_special_chars = {
            '&': '\\&',
            '%': '\\%',
            '$': '\\$',
            '#': '\\#',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}',
            '\\': '\\textbackslash{}',
            '<': '\\textless{}',
            '>': '\\textgreater{}'
        }
        
        # Replace special characters
        for char, escaped in latex_special_chars.items():
            text = text.replace(char, escaped)
        
        return text
    
    def _escape_html(self, text: str) -> str:
        """
        Escape special characters in HTML.
        
        Args:
            text: Text to escape
            
        Returns:
            Escaped text
        """
        # Define HTML special characters and their escaped versions
        html_special_chars = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }
        
        # Replace special characters
        for char, escaped in html_special_chars.items():
            text = text.replace(char, escaped)
        
        return text
    
    def _format_bibtex_entry(self, reference: Reference) -> str:
        """
        Format a reference as a BibTeX entry.
        
        Args:
            reference: Reference to format
            
        Returns:
            BibTeX entry string
        """
        # Determine BibTeX entry type
        if reference.type == "article":
            entry_type = "article"
        elif reference.type == "book":
            entry_type = "book"
        elif reference.type == "conference":
            entry_type = "inproceedings"
        else:
            entry_type = "misc"
        
        # Format authors
        authors = " and ".join(reference.authors)
        
        # Build BibTeX entry
        lines = [f"@{entry_type}{{{reference.id},"]
        lines.append(f"  author = {{{authors}}},")
        lines.append(f"  title = {{{reference.title}}},")
        lines.append(f"  year = {{{reference.year}}},")
        
        if entry_type == "article":
            lines.append(f"  journal = {{{reference.source}}},")
            if reference.volume:
                lines.append(f"  volume = {{{reference.volume}}},")
            if reference.issue:
                lines.append(f"  number = {{{reference.issue}}},")
            if reference.pages:
                lines.append(f"  pages = {{{reference.pages}}},")
        elif entry_type == "book":
            if reference.publisher:
                lines.append(f"  publisher = {{{reference.publisher}}},")
        elif entry_type == "inproceedings":
            lines.append(f"  booktitle = {{{reference.source}}},")
            if reference.pages:
                lines.append(f"  pages = {{{reference.pages}}},")
        
        if reference.doi:
            lines.append(f"  doi = {{{reference.doi}}},")
        if reference.url:
            lines.append(f"  url = {{{reference.url}}},")
        
        # Remove trailing comma from last line
        lines[-1] = lines[-1][:-1]
        
        lines.append("}")
        
        return "\n".join(lines)


class PublicationManager:
    """
    Manager for scientific publications.
    
    This class provides methods for creating, managing, and exporting
    scientific publications in various formats.
    """
    
    def __init__(self, storage_dir: str = "publications"):
        """
        Initialize the PublicationManager.
        
        Args:
            storage_dir: Directory for storing publication data
        """
        self.storage_dir = storage_dir
        self.publications = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        logger.info(f"Initialized PublicationManager with storage directory: {storage_dir}")
    
    def create_publication(self, title: str, authors: List[Author], abstract: str, 
                          keywords: List[str], sections: List[Section]) -> Publication:
        """
        Create a new publication.
        
        Args:
            title: Publication title
            authors: List of authors
            abstract: Publication abstract
            keywords: List of keywords
            sections: List of sections
            
        Returns:
            Created publication
        """
        publication = Publication(
            title=title,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            sections=sections
        )
        
        self.publications[publication.id] = publication
        
        logger.info(f"Created publication '{title}' with ID {publication.id}")
        return publication
    
    def get_publication(self, publication_id: str) -> Optional[Publication]:
        """
        Get a publication by ID.
        
        Args:
            publication_id: Publication ID
            
        Returns:
            Publication object or None if not found
        """
        return self.publications.get(publication_id)
    
    def list_publications(self) -> List[Publication]:
        """
        List all publications.
        
        Returns:
            List of publications
        """
        return list(self.publications.values())
    
    def save_publication(self, publication_id: str) -> Optional[str]:
        """
        Save a publication to disk.
        
        Args:
            publication_id: Publication ID
            
        Returns:
            Path to the saved publication file, or None if publication not found
        """
        publication = self.get_publication(publication_id)
        if publication is None:
            logger.error(f"Publication with ID {publication_id} not found")
            return None
        
        return publication.save(self.storage_dir)
    
    def load_publication(self, filepath: str) -> Optional[Publication]:
        """
        Load a publication from disk.
        
        Args:
            filepath: Path to the publication file
            
        Returns:
            Loaded publication, or None if loading failed
        """
        try:
            publication = Publication.load(filepath)
            self.publications[publication.id] = publication
            return publication
        except Exception as e:
            logger.error(f"Error loading publication from {filepath}: {str(e)}")
            return None
    
    def load_all_publications(self) -> int:
        """
        Load all publications from the storage directory.
        
        Returns:
            Number of publications loaded
        """
        count = 0
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_dir, filename)
                publication = self.load_publication(filepath)
                if publication is not None:
                    count += 1
        
        logger.info(f"Loaded {count} publications from {self.storage_dir}")
        return count
    
    def delete_publication(self, publication_id: str) -> bool:
        """
        Delete a publication.
        
        Args:
            publication_id: Publication ID
            
        Returns:
            True if deletion was successful, False otherwise
        """
        publication = self.get_publication(publication_id)
        if publication is None:
            logger.error(f"Publication with ID {publication_id} not found")
            return False
        
        # Remove from publications dictionary
        del self.publications[publication_id]
        
        # Delete publication file if it exists
        filepath = os.path.join(self.storage_dir, f"{publication_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        logger.info(f"Deleted publication '{publication.title}' with ID {publication_id}")
        return True
    
    def export_publication(self, publication_id: str, format: PublicationFormat, 
                          output_dir: str, citation_style: CitationStyle = CitationStyle.APA,
                          journal_template: JournalTemplate = JournalTemplate.GENERIC) -> Optional[str]:
        """
        Export a publication to the specified format.
        
        Args:
            publication_id: Publication ID
            format: Output format
            output_dir: Directory to save the exported file
            citation_style: Citation style to use
            journal_template: Journal template to use
            
        Returns:
            Path to the exported file, or None if publication not found
        """
        publication = self.get_publication(publication_id)
        if publication is None:
            logger.error(f"Publication with ID {publication_id} not found")
            return None
        
        return publication.export(format, output_dir, citation_style, journal_template)
    
    def create_publication_from_experiment(self, experiment_id: str, 
                                          title: Optional[str] = None,
                                          authors: Optional[List[Author]] = None,
                                          abstract: Optional[str] = None,
                                          keywords: Optional[List[str]] = None) -> Optional[Publication]:
        """
        Create a publication from an experiment.
        
        Args:
            experiment_id: Experiment ID
            title: Publication title (defaults to experiment name)
            authors: List of authors (defaults to experiment author)
            abstract: Publication abstract (defaults to experiment description)
            keywords: List of keywords (defaults to experiment tags)
            
        Returns:
            Created publication, or None if experiment not found
        """
        # Import experiment manager
        try:
            from ..experiment_workflow.workflow import get_experiment_manager
            experiment_manager = get_experiment_manager()
        except ImportError:
            logger.error("Could not import experiment manager")
            return None
        
        # Get experiment
        experiment = experiment_manager.get_experiment(experiment_id)
        if experiment is None:
            logger.error(f"Experiment with ID {experiment_id} not found")
            return None
        
        # Set default values
        if title is None:
            title = f"Results of Experiment: {experiment.name}"
        
        if authors is None:
            authors = [Author(name=experiment.author, is_corresponding=True)]
        
        if abstract is None:
            abstract = experiment.description
        
        if keywords is None:
            keywords = experiment.tags
        
        # Create sections
        sections = []
        
        # Introduction section
        intro_content = f"This paper presents the results of the experiment '{experiment.name}' "
        intro_content += f"conducted on {experiment.created_at.strftime('%B %d, %Y')}. "
        intro_content += f"The experiment was designed to investigate {experiment.description}."
        
        sections.append(Section(
            title="Introduction",
            content=intro_content,
            level=1,
            id="introduction"
        ))
        
        # Methods section
        methods_content = "The experiment was conducted using the following methodology:\n\n"
        methods_content += "### Parameters\n\n"
        for param_name, param_value in experiment.parameters.items():
            methods_content += f"- **{param_name}**: {param_value}\n"
        
        methods_content += "\n### Procedure\n\n"
        for step in experiment.steps:
            methods_content += f"- **{step.name}**: {step.description}\n"
        
        sections.append(Section(
            title="Methods",
            content=methods_content,
            level=1,
            id="methods"
        ))
        
        # Results section
        results_content = "The experiment yielded the following results:\n\n"
        results_content += "### Metrics\n\n"
        for metric_name, metric_value in experiment.metrics.items():
            results_content += f"- **{metric_name}**: {metric_value}\n"
        
        results_content += "\n### Step Results\n\n"
        for step in experiment.steps:
            results_content += f"#### {step.name}\n\n"
            results_content += f"Status: {step.status}\n\n"
            
            if step.metrics:
                results_content += "Metrics:\n\n"
                for metric_name, metric_value in step.metrics.items():
                    results_content += f"- **{metric_name}**: {metric_value}\n"
            
            results_content += "\n"
        
        sections.append(Section(
            title="Results",
            content=results_content,
            level=1,
            id="results"
        ))
        
        # Discussion section
        discussion_content = "The results of this experiment demonstrate "
        if experiment.status == "completed":
            discussion_content += "successful completion of all planned steps. "
        else:
            discussion_content += f"partial completion with status '{experiment.status}'. "
        
        discussion_content += "Further analysis and interpretation of these results "
        discussion_content += "can provide insights into the underlying phenomena being studied."
        
        sections.append(Section(
            title="Discussion",
            content=discussion_content,
            level=1,
            id="discussion"
        ))
        
        # Conclusion section
        conclusion_content = "This experiment has provided valuable data on "
        conclusion_content += f"{experiment.name}. The findings suggest that "
        conclusion_content += "further research in this area could yield additional insights."
        
        sections.append(Section(
            title="Conclusion",
            content=conclusion_content,
            level=1,
            id="conclusion"
        ))
        
        # Create publication
        publication = self.create_publication(
            title=title,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            sections=sections
        )
        
        # Add figures from experiment artifacts
        for artifact in experiment.artifacts:
            if hasattr(artifact, 'path') and artifact.path and artifact.type == 'image':
                figure = Figure(
                    id=f"fig-{len(publication.figures) + 1}",
                    title=artifact.name,
                    caption=artifact.description,
                    path=artifact.path,
                    type="image"
                )
                publication.add_figure(figure)
        
        # Add experiment as reference
        reference = Reference(
            id=f"exp-{experiment.id[:8]}",
            type="misc",
            title=experiment.name,
            authors=[experiment.author] if experiment.author else ["Anonymous"],
            year=experiment.created_at.year,
            source="Prometheus AI Automation Platform",
            url=f"experiment://{experiment.id}"
        )
        publication.add_reference(reference)
        
        # Set acknowledgments
        publication.acknowledgments = "This research was conducted using the Prometheus AI Automation Platform."
        
        # Save publication
        self.save_publication(publication.id)
        
        logger.info(f"Created publication '{publication.title}' from experiment '{experiment.name}'")
        return publication


# Create a singleton instance for easy import
publication_manager = None

def get_publication_manager(storage_dir: str = "publications") -> PublicationManager:
    """
    Get the publication manager singleton instance.
    
    Args:
        storage_dir: Directory for storing publication data
        
    Returns:
        PublicationManager instance
    """
    global publication_manager
    
    if publication_manager is None:
        publication_manager = PublicationManager(storage_dir=storage_dir)
    
    return publication_manager
