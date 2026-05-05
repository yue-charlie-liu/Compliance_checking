"""
The schema definitions for the OpenAI API
calls are defined as pydantic models.
"""
from pydantic import BaseModel, Field, ConfigDict

class FlatSchemaItem(BaseModel):
    title: str = Field(default="", description="Title of the segment")
    text: str = Field(default="", description="Main text content of the segment as plain text.")
    page: int = Field(default=None, description="Page number where this segment starts")
    model_config = ConfigDict(extra="forbid")

class FlatSchema(BaseModel):
    content: list[FlatSchemaItem]
    model_config = ConfigDict(extra="forbid")

class TOCSchemaItem(BaseModel):
    index: int = Field(default=None, description="Index of the segment in the document")
    unique_id: str = Field(default=None, description="Unique identifier for the segment")
    type: str = Field(default=None, description="Type of the segment: 'section', 'subsection', 'article', etc.")
    level: int = Field(default=None, description="Nesting level of the segment in the document hierarchy")
    action: str = Field(default="add", description="Action to take: 'add', 'merge', 'remove'")
    model_config = ConfigDict(extra="forbid")

class TOCSchema(BaseModel):
    content: list[TOCSchemaItem]
    missing: str = Field(default=None, description="Any missing sections that were not found")
    model_config = ConfigDict(extra="forbid")

class MetadataSchema(BaseModel):
    title: str = Field(default=None, description="Main title of the document")
    enactedOn: str = Field(default=None, description="Date the document was enacted (DD-MM-YYYY)")
    jurisdiction: str
    implementedBy: str
    llmSummary: str = Field(default=None, description="Short machine-generated summary")
    model_config = ConfigDict(extra="forbid")
