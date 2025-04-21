"""
API module for the collaboration system in the Prometheus AI Automation Platform.

This module provides a RESTful API for managing collaborative research projects,
team members, permissions, and comments.
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

from .collaboration import (
    CollaborationRole, PermissionLevel, Collaborator, Comment, ActivityLog,
    ResourcePermission, CollaborationProject, get_collaboration_manager
)

# Create FastAPI app
app = FastAPI(
    title="Prometheus Collaboration API",
    description="API for managing collaborative research projects",
    version="0.1.0"
)

# OAuth2 password bearer for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models for API requests and responses
class CollaboratorModel(BaseModel):
    id: str
    name: str
    email: str
    role: str
    joined_at: datetime.datetime
    bio: Optional[str] = None
    organization: Optional[str] = None
    avatar_url: Optional[str] = None
    last_active: Optional[datetime.datetime] = None

class CommentModel(BaseModel):
    id: str
    author_id: str
    content: str
    created_at: datetime.datetime
    updated_at: Optional[datetime.datetime] = None
    parent_id: Optional[str] = None
    resolved: bool
    resource_type: str
    resource_id: str
    resource_section: Optional[str] = None

class ActivityLogModel(BaseModel):
    id: str
    user_id: str
    action: str
    timestamp: datetime.datetime
    resource_type: str
    resource_id: str
    details: Dict[str, Any] = {}

class ResourcePermissionModel(BaseModel):
    resource_type: str
    resource_id: str
    user_id: str
    permission: str
    granted_by: str
    granted_at: datetime.datetime
    expires_at: Optional[datetime.datetime] = None

class ProjectCreateModel(BaseModel):
    name: str
    description: str
    visibility: str = "private"

class ProjectUpdateModel(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    visibility: Optional[str] = None
    status: Optional[str] = None

class ProjectResponseModel(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    visibility: str
    status: str
    collaborators_count: int
    experiments_count: int
    publications_count: int
    datasets_count: int

class CollaboratorCreateModel(BaseModel):
    user_id: str
    name: str
    email: str
    role: str

class CommentCreateModel(BaseModel):
    content: str
    resource_type: str
    resource_id: str
    resource_section: Optional[str] = None
    parent_id: Optional[str] = None

class PermissionCreateModel(BaseModel):
    resource_type: str
    resource_id: str
    target_user_id: str
    permission: str
    expires_at: Optional[datetime.datetime] = None

# Helper functions
def project_to_response(project: CollaborationProject) -> ProjectResponseModel:
    """
    Convert a CollaborationProject object to a ProjectResponseModel.
    
    Args:
        project: CollaborationProject object
        
    Returns:
        ProjectResponseModel
    """
    return ProjectResponseModel(
        id=project.id,
        name=project.name,
        description=project.description,
        owner_id=project.owner_id,
        created_at=project.created_at,
        updated_at=project.updated_at,
        visibility=project.visibility,
        status=project.status,
        collaborators_count=len(project.collaborators),
        experiments_count=len(project.experiments),
        publications_count=len(project.publications),
        datasets_count=len(project.datasets)
    )

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    Get the current user ID from the authentication token.
    
    Args:
        token: Authentication token
        
    Returns:
        User ID
    """
    # In a real implementation, this would validate the token and extract the user ID
    # For now, we'll just return a placeholder user ID
    return "current_user_id"

# API routes
@app.get("/projects", response_model=List[ProjectResponseModel])
def list_projects(
    visibility: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    List projects.
    
    Args:
        visibility: Filter by visibility
        current_user: Current user ID
        
    Returns:
        List of projects
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    projects = manager.list_projects(user_id=current_user, visibility=visibility)
    return [project_to_response(project) for project in projects]

@app.post("/projects", response_model=ProjectResponseModel)
def create_project(
    project_data: ProjectCreateModel,
    current_user: str = Depends(get_current_user)
):
    """
    Create a new project.
    
    Args:
        project_data: Project data
        current_user: Current user ID
        
    Returns:
        Created project
    """
    manager = get_collaboration_manager()
    
    project = manager.create_project(
        name=project_data.name,
        description=project_data.description,
        owner_id=current_user,
        visibility=project_data.visibility
    )
    
    return project_to_response(project)

@app.get("/projects/{project_id}", response_model=ProjectResponseModel)
def get_project(
    project_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get a project by ID.
    
    Args:
        project_id: Project ID
        current_user: Current user ID
        
    Returns:
        Project
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to view the project
    if not project.is_collaborator(current_user) and project.visibility != "public":
        raise HTTPException(status_code=403, detail="You don't have permission to view this project")
    
    return project_to_response(project)

@app.put("/projects/{project_id}", response_model=ProjectResponseModel)
def update_project(
    project_id: str,
    project_data: ProjectUpdateModel,
    current_user: str = Depends(get_current_user)
):
    """
    Update a project.
    
    Args:
        project_id: Project ID
        project_data: Project data
        current_user: Current user ID
        
    Returns:
        Updated project
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to update the project
    user_role = project.get_user_role(current_user)
    if user_role not in [CollaborationRole.OWNER, CollaborationRole.ADMIN]:
        raise HTTPException(status_code=403, detail="You don't have permission to update this project")
    
    # Update fields
    if project_data.name is not None:
        project.name = project_data.name
    
    if project_data.description is not None:
        project.description = project_data.description
    
    if project_data.visibility is not None:
        project.visibility = project_data.visibility
    
    if project_data.status is not None:
        project.status = project_data.status
    
    # Update timestamp
    project.updated_at = datetime.datetime.now()
    
    # Save project
    manager.save_project(project.id)
    
    return project_to_response(project)

@app.delete("/projects/{project_id}")
def delete_project(
    project_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Delete a project.
    
    Args:
        project_id: Project ID
        current_user: Current user ID
        
    Returns:
        Success message
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user is the owner
    if project.owner_id != current_user:
        raise HTTPException(status_code=403, detail="Only the project owner can delete the project")
    
    success = manager.delete_project(project_id)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete project with ID {project_id}")
    
    return {"message": f"Project with ID {project_id} deleted"}

@app.get("/projects/{project_id}/collaborators", response_model=List[CollaboratorModel])
def list_collaborators(
    project_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    List collaborators for a project.
    
    Args:
        project_id: Project ID
        current_user: Current user ID
        
    Returns:
        List of collaborators
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to view the project
    if not project.is_collaborator(current_user) and project.visibility != "public":
        raise HTTPException(status_code=403, detail="You don't have permission to view this project")
    
    # Convert collaborators to response models
    collaborators = []
    for collaborator in project.collaborators:
        collaborators.append(CollaboratorModel(
            id=collaborator.id,
            name=collaborator.name,
            email=collaborator.email,
            role=collaborator.role.value,
            joined_at=collaborator.joined_at,
            bio=collaborator.bio,
            organization=collaborator.organization,
            avatar_url=collaborator.avatar_url,
            last_active=collaborator.last_active
        ))
    
    # Add owner if not already in the list
    owner_in_list = False
    for collaborator in collaborators:
        if collaborator.id == project.owner_id:
            owner_in_list = True
            break
    
    if not owner_in_list:
        # In a real implementation, we would fetch the owner's details from a user service
        collaborators.append(CollaboratorModel(
            id=project.owner_id,
            name="Project Owner",
            email="owner@example.com",
            role=CollaborationRole.OWNER.value,
            joined_at=project.created_at,
            bio=None,
            organization=None,
            avatar_url=None,
            last_active=None
        ))
    
    return collaborators

@app.post("/projects/{project_id}/collaborators", response_model=CollaboratorModel)
def add_collaborator(
    project_id: str,
    collaborator_data: CollaboratorCreateModel,
    current_user: str = Depends(get_current_user)
):
    """
    Add a collaborator to a project.
    
    Args:
        project_id: Project ID
        collaborator_data: Collaborator data
        current_user: Current user ID
        
    Returns:
        Added collaborator
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to add collaborators
    user_role = project.get_user_role(current_user)
    if user_role not in [CollaborationRole.OWNER, CollaborationRole.ADMIN]:
        raise HTTPException(status_code=403, detail="You don't have permission to add collaborators to this project")
    
    # Parse role
    try:
        role = CollaborationRole(collaborator_data.role)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role: {collaborator_data.role}. Valid roles: {', '.join([r.value for r in CollaborationRole])}"
        )
    
    # Add collaborator
    collaborator = manager.add_collaborator(
        project_id=project_id,
        user_id=collaborator_data.user_id,
        name=collaborator_data.name,
        email=collaborator_data.email,
        role=role
    )
    
    if collaborator is None:
        raise HTTPException(status_code=500, detail=f"Failed to add collaborator to project with ID {project_id}")
    
    return CollaboratorModel(
        id=collaborator.id,
        name=collaborator.name,
        email=collaborator.email,
        role=collaborator.role.value,
        joined_at=collaborator.joined_at,
        bio=collaborator.bio,
        organization=collaborator.organization,
        avatar_url=collaborator.avatar_url,
        last_active=collaborator.last_active
    )

@app.delete("/projects/{project_id}/collaborators/{user_id}")
def remove_collaborator(
    project_id: str,
    user_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Remove a collaborator from a project.
    
    Args:
        project_id: Project ID
        user_id: User ID to remove
        current_user: Current user ID
        
    Returns:
        Success message
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to remove collaborators
    user_role = project.get_user_role(current_user)
    if user_role not in [CollaborationRole.OWNER, CollaborationRole.ADMIN]:
        raise HTTPException(status_code=403, detail="You don't have permission to remove collaborators from this project")
    
    # Check if trying to remove the owner
    if user_id == project.owner_id:
        raise HTTPException(status_code=400, detail="Cannot remove the project owner")
    
    # Remove collaborator
    success = manager.remove_collaborator(project_id, user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Collaborator with ID {user_id} not found in project with ID {project_id}")
    
    return {"message": f"Collaborator with ID {user_id} removed from project with ID {project_id}"}

@app.get("/projects/{project_id}/comments", response_model=List[CommentModel])
def list_comments(
    project_id: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    resolved: Optional[bool] = None,
    current_user: str = Depends(get_current_user)
):
    """
    List comments for a project.
    
    Args:
        project_id: Project ID
        resource_type: Filter by resource type
        resource_id: Filter by resource ID
        resolved: Filter by resolved status
        current_user: Current user ID
        
    Returns:
        List of comments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to view the project
    if not project.is_collaborator(current_user) and project.visibility != "public":
        raise HTTPException(status_code=403, detail="You don't have permission to view this project")
    
    # Get comments
    comments = manager.get_project_comments(
        project_id=project_id,
        resource_type=resource_type,
        resource_id=resource_id,
        resolved=resolved
    )
    
    # Convert comments to response models
    return [CommentModel(
        id=comment.id,
        author_id=comment.author_id,
        content=comment.content,
        created_at=comment.created_at,
        updated_at=comment.updated_at,
        parent_id=comment.parent_id,
        resolved=comment.resolved,
        resource_type=comment.resource_type,
        resource_id=comment.resource_id,
        resource_section=comment.resource_section
    ) for comment in comments]

@app.post("/projects/{project_id}/comments", response_model=CommentModel)
def add_comment(
    project_id: str,
    comment_data: CommentCreateModel,
    current_user: str = Depends(get_current_user)
):
    """
    Add a comment to a project.
    
    Args:
        project_id: Project ID
        comment_data: Comment data
        current_user: Current user ID
        
    Returns:
        Added comment
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to comment
    if not project.is_collaborator(current_user):
        raise HTTPException(status_code=403, detail="You don't have permission to comment on this project")
    
    # Add comment
    comment = manager.add_comment(
        project_id=project_id,
        user_id=current_user,
        content=comment_data.content,
        resource_type=comment_data.resource_type,
        resource_id=comment_data.resource_id,
        resource_section=comment_data.resource_section,
        parent_id=comment_data.parent_id
    )
    
    if comment is None:
        raise HTTPException(status_code=500, detail=f"Failed to add comment to project with ID {project_id}")
    
    return CommentModel(
        id=comment.id,
        author_id=comment.author_id,
        content=comment.content,
        created_at=comment.created_at,
        updated_at=comment.updated_at,
        parent_id=comment.parent_id,
        resolved=comment.resolved,
        resource_type=comment.resource_type,
        resource_id=comment.resource_id,
        resource_section=comment.resource_section
    )

@app.put("/projects/{project_id}/comments/{comment_id}")
def update_comment(
    project_id: str,
    comment_id: str,
    content: str,
    current_user: str = Depends(get_current_user)
):
    """
    Update a comment.
    
    Args:
        project_id: Project ID
        comment_id: Comment ID
        content: New comment content
        current_user: Current user ID
        
    Returns:
        Success message
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Update comment
    success = project.update_comment(current_user, comment_id, content)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Comment with ID {comment_id} not found or you don't have permission to update it")
    
    # Save project
    manager.save_project(project_id)
    
    return {"message": f"Comment with ID {comment_id} updated"}

@app.delete("/projects/{project_id}/comments/{comment_id}")
def delete_comment(
    project_id: str,
    comment_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Delete a comment.
    
    Args:
        project_id: Project ID
        comment_id: Comment ID
        current_user: Current user ID
        
    Returns:
        Success message
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Delete comment
    success = project.delete_comment(current_user, comment_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Comment with ID {comment_id} not found or you don't have permission to delete it")
    
    # Save project
    manager.save_project(project_id)
    
    return {"message": f"Comment with ID {comment_id} deleted"}

@app.post("/projects/{project_id}/comments/{comment_id}/resolve")
def resolve_comment(
    project_id: str,
    comment_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Resolve a comment.
    
    Args:
        project_id: Project ID
        comment_id: Comment ID
        current_user: Current user ID
        
    Returns:
        Success message
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Resolve comment
    success = project.resolve_comment(current_user, comment_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Comment with ID {comment_id} not found or you don't have permission to resolve it")
    
    # Save project
    manager.save_project(project_id)
    
    return {"message": f"Comment with ID {comment_id} resolved"}

@app.get("/projects/{project_id}/activity", response_model=List[ActivityLogModel])
def get_activity_log(
    project_id: str,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    start_time: Optional[datetime.datetime] = None,
    end_time: Optional[datetime.datetime] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Get activity log for a project.
    
    Args:
        project_id: Project ID
        user_id: Filter by user ID
        action: Filter by action
        resource_type: Filter by resource type
        resource_id: Filter by resource ID
        start_time: Filter by start time
        end_time: Filter by end time
        current_user: Current user ID
        
    Returns:
        List of activity log entries
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to view the project
    if not project.is_collaborator(current_user) and project.visibility != "public":
        raise HTTPException(status_code=403, detail="You don't have permission to view this project")
    
    # Get activity log
    activities = manager.get_project_activity_log(
        project_id=project_id,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        start_time=start_time,
        end_time=end_time
    )
    
    # Convert activities to response models
    return [ActivityLogModel(
        id=activity.id,
        user_id=activity.user_id,
        action=activity.action,
        timestamp=activity.timestamp,
        resource_type=activity.resource_type,
        resource_id=activity.resource_id,
        details=activity.details
    ) for activity in activities]

@app.post("/projects/{project_id}/permissions", response_model=ResourcePermissionModel)
def set_permission(
    project_id: str,
    permission_data: PermissionCreateModel,
    current_user: str = Depends(get_current_user)
):
    """
    Set permission for a resource in a project.
    
    Args:
        project_id: Project ID
        permission_data: Permission data
        current_user: Current user ID
        
    Returns:
        Created resource permission
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Parse permission
    try:
        permission = PermissionLevel(permission_data.permission)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid permission: {permission_data.permission}. Valid permissions: {', '.join([p.value for p in PermissionLevel])}"
        )
    
    # Set permission
    resource_permission = manager.set_resource_permission(
        project_id=project_id,
        user_id=current_user,
        resource_type=permission_data.resource_type,
        resource_id=permission_data.resource_id,
        permission=permission,
        target_user_id=permission_data.target_user_id,
        expires_at=permission_data.expires_at
    )
    
    if resource_permission is None:
        raise HTTPException(status_code=500, detail=f"Failed to set permission in project with ID {project_id}")
    
    return ResourcePermissionModel(
        resource_type=resource_permission.resource_type,
        resource_id=resource_permission.resource_id,
        user_id=resource_permission.user_id,
        permission=resource_permission.permission.value,
        granted_by=resource_permission.granted_by,
        granted_at=resource_permission.granted_at,
        expires_at=resource_permission.expires_at
    )

@app.get("/projects/{project_id}/permissions/{resource_type}/{resource_id}/{user_id}")
def get_permission(
    project_id: str,
    resource_type: str,
    resource_id: str,
    user_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Get permission level for a resource in a project.
    
    Args:
        project_id: Project ID
        resource_type: Resource type
        resource_id: Resource ID
        user_id: User ID
        current_user: Current user ID
        
    Returns:
        Permission level
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to view the project
    if not project.is_collaborator(current_user) and project.visibility != "public":
        raise HTTPException(status_code=403, detail="You don't have permission to view this project")
    
    # Get permission
    permission = manager.get_resource_permission(project_id, user_id, resource_type, resource_id)
    
    return {"permission": permission.value}

@app.post("/projects/{project_id}/resources/{resource_type}/{resource_id}")
def add_resource(
    project_id: str,
    resource_type: str,
    resource_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Add a resource to a project.
    
    Args:
        project_id: Project ID
        resource_type: Resource type (experiment, publication, dataset)
        resource_id: Resource ID
        current_user: Current user ID
        
    Returns:
        Success message
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to add resources
    user_role = project.get_user_role(current_user)
    if user_role not in [CollaborationRole.OWNER, CollaborationRole.ADMIN, CollaborationRole.CONTRIBUTOR]:
        raise HTTPException(status_code=403, detail="You don't have permission to add resources to this project")
    
    # Add resource
    success = manager.add_resource(project_id, resource_type, resource_id)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to add {resource_type} with ID {resource_id} to project with ID {project_id}")
    
    return {"message": f"{resource_type.capitalize()} with ID {resource_id} added to project with ID {project_id}"}

@app.delete("/projects/{project_id}/resources/{resource_type}/{resource_id}")
def remove_resource(
    project_id: str,
    resource_type: str,
    resource_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    Remove a resource from a project.
    
    Args:
        project_id: Project ID
        resource_type: Resource type (experiment, publication, dataset)
        resource_id: Resource ID
        current_user: Current user ID
        
    Returns:
        Success message
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(project_id)
    
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project with ID {project_id} not found")
    
    # Check if user has permission to remove resources
    user_role = project.get_user_role(current_user)
    if user_role not in [CollaborationRole.OWNER, CollaborationRole.ADMIN]:
        raise HTTPException(status_code=403, detail="You don't have permission to remove resources from this project")
    
    # Remove resource
    success = manager.remove_resource(project_id, resource_type, resource_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"{resource_type.capitalize()} with ID {resource_id} not found in project with ID {project_id}")
    
    return {"message": f"{resource_type.capitalize()} with ID {resource_id} removed from project with ID {project_id}"}
