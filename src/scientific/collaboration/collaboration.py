"""
Collaboration module for the Prometheus AI Automation Platform.

This module provides classes and functions for enabling collaborative scientific research,
including team management, real-time collaboration, version control, and permissions.
"""

import os
import logging
import json
import datetime
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict

# Configure logging
logger = logging.getLogger(__name__)

class CollaborationRole(Enum):
    """Role for a collaborator in a research project."""
    OWNER = "owner"
    ADMIN = "admin"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"

class PermissionLevel(Enum):
    """Permission level for a resource."""
    READ = "read"
    COMMENT = "comment"
    EDIT = "edit"
    MANAGE = "manage"
    NONE = "none"

@dataclass
class Collaborator:
    """Collaborator information for a research project."""
    id: str
    name: str
    email: str
    role: CollaborationRole
    joined_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    bio: Optional[str] = None
    organization: Optional[str] = None
    avatar_url: Optional[str] = None
    last_active: Optional[datetime.datetime] = None

@dataclass
class Comment:
    """Comment on a research resource."""
    id: str
    author_id: str
    content: str
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: Optional[datetime.datetime] = None
    parent_id: Optional[str] = None
    resolved: bool = False
    resource_type: str = ""  # experiment, publication, dataset, etc.
    resource_id: str = ""
    resource_section: Optional[str] = None  # specific section or part of the resource

@dataclass
class ActivityLog:
    """Activity log entry for a research project."""
    id: str
    user_id: str
    action: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    resource_type: str = ""  # experiment, publication, dataset, etc.
    resource_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChangeRecord:
    """Record of a change to a research resource."""
    id: str
    user_id: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    resource_type: str
    resource_id: str
    field: str
    old_value: Any
    new_value: Any
    description: Optional[str] = None

@dataclass
class ResourcePermission:
    """Permission settings for a resource."""
    resource_type: str
    resource_id: str
    user_id: str
    permission: PermissionLevel
    granted_by: str
    granted_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    expires_at: Optional[datetime.datetime] = None

@dataclass
class CollaborationProject:
    """
    Collaborative research project for the Prometheus AI Automation Platform.
    
    This class represents a collaborative research project with team members,
    shared resources, and collaboration features.
    """
    id: str
    name: str
    description: str
    owner_id: str
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    collaborators: List[Collaborator] = field(default_factory=list)
    experiments: List[str] = field(default_factory=list)  # List of experiment IDs
    publications: List[str] = field(default_factory=list)  # List of publication IDs
    datasets: List[str] = field(default_factory=list)  # List of dataset IDs
    
    tags: List[str] = field(default_factory=list)
    status: str = "active"
    visibility: str = "private"  # private, team, public
    
    activity_log: List[ActivityLog] = field(default_factory=list)
    comments: List[Comment] = field(default_factory=list)
    permissions: List[ResourcePermission] = field(default_factory=list)
    
    def add_collaborator(self, user_id: str, name: str, email: str, role: CollaborationRole) -> Collaborator:
        """
        Add a collaborator to the project.
        
        Args:
            user_id: User ID
            name: User name
            email: User email
            role: Collaboration role
            
        Returns:
            Added collaborator
        """
        # Check if user is already a collaborator
        for collaborator in self.collaborators:
            if collaborator.id == user_id:
                logger.warning(f"User {user_id} is already a collaborator on project {self.id}")
                return collaborator
        
        # Create collaborator
        collaborator = Collaborator(
            id=user_id,
            name=name,
            email=email,
            role=role
        )
        
        # Add to collaborators list
        self.collaborators.append(collaborator)
        
        # Update project
        self.updated_at = datetime.datetime.now()
        
        # Log activity
        self.log_activity(
            user_id=self.owner_id,
            action="add_collaborator",
            resource_type="project",
            resource_id=self.id,
            details={
                "collaborator_id": user_id,
                "collaborator_name": name,
                "role": role.value
            }
        )
        
        logger.info(f"Added collaborator {name} ({user_id}) to project {self.name} ({self.id})")
        return collaborator
    
    def remove_collaborator(self, user_id: str) -> bool:
        """
        Remove a collaborator from the project.
        
        Args:
            user_id: User ID
            
        Returns:
            True if collaborator was removed, False otherwise
        """
        # Check if user is the owner
        if user_id == self.owner_id:
            logger.error(f"Cannot remove owner {user_id} from project {self.id}")
            return False
        
        # Find collaborator
        for i, collaborator in enumerate(self.collaborators):
            if collaborator.id == user_id:
                # Remove collaborator
                removed = self.collaborators.pop(i)
                
                # Update project
                self.updated_at = datetime.datetime.now()
                
                # Log activity
                self.log_activity(
                    user_id=self.owner_id,
                    action="remove_collaborator",
                    resource_type="project",
                    resource_id=self.id,
                    details={
                        "collaborator_id": user_id,
                        "collaborator_name": removed.name
                    }
                )
                
                logger.info(f"Removed collaborator {removed.name} ({user_id}) from project {self.name} ({self.id})")
                return True
        
        logger.warning(f"User {user_id} is not a collaborator on project {self.id}")
        return False
    
    def update_collaborator_role(self, user_id: str, role: CollaborationRole) -> bool:
        """
        Update a collaborator's role.
        
        Args:
            user_id: User ID
            role: New collaboration role
            
        Returns:
            True if role was updated, False otherwise
        """
        # Check if user is the owner
        if user_id == self.owner_id and role != CollaborationRole.OWNER:
            logger.error(f"Cannot change owner role for {user_id} in project {self.id}")
            return False
        
        # Find collaborator
        for collaborator in self.collaborators:
            if collaborator.id == user_id:
                # Update role
                old_role = collaborator.role
                collaborator.role = role
                
                # Update project
                self.updated_at = datetime.datetime.now()
                
                # Log activity
                self.log_activity(
                    user_id=self.owner_id,
                    action="update_collaborator_role",
                    resource_type="project",
                    resource_id=self.id,
                    details={
                        "collaborator_id": user_id,
                        "collaborator_name": collaborator.name,
                        "old_role": old_role.value,
                        "new_role": role.value
                    }
                )
                
                logger.info(f"Updated role for {collaborator.name} ({user_id}) from {old_role.value} to {role.value} in project {self.name} ({self.id})")
                return True
        
        logger.warning(f"User {user_id} is not a collaborator on project {self.id}")
        return False
    
    def add_resource(self, resource_type: str, resource_id: str) -> bool:
        """
        Add a resource to the project.
        
        Args:
            resource_type: Resource type (experiment, publication, dataset)
            resource_id: Resource ID
            
        Returns:
            True if resource was added, False otherwise
        """
        # Check resource type
        if resource_type == "experiment":
            resource_list = self.experiments
        elif resource_type == "publication":
            resource_list = self.publications
        elif resource_type == "dataset":
            resource_list = self.datasets
        else:
            logger.error(f"Invalid resource type: {resource_type}")
            return False
        
        # Check if resource is already in the project
        if resource_id in resource_list:
            logger.warning(f"{resource_type.capitalize()} {resource_id} is already in project {self.id}")
            return False
        
        # Add resource
        resource_list.append(resource_id)
        
        # Update project
        self.updated_at = datetime.datetime.now()
        
        # Log activity
        self.log_activity(
            user_id=self.owner_id,
            action=f"add_{resource_type}",
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                "project_id": self.id,
                "project_name": self.name
            }
        )
        
        logger.info(f"Added {resource_type} {resource_id} to project {self.name} ({self.id})")
        return True
    
    def remove_resource(self, resource_type: str, resource_id: str) -> bool:
        """
        Remove a resource from the project.
        
        Args:
            resource_type: Resource type (experiment, publication, dataset)
            resource_id: Resource ID
            
        Returns:
            True if resource was removed, False otherwise
        """
        # Check resource type
        if resource_type == "experiment":
            resource_list = self.experiments
        elif resource_type == "publication":
            resource_list = self.publications
        elif resource_type == "dataset":
            resource_list = self.datasets
        else:
            logger.error(f"Invalid resource type: {resource_type}")
            return False
        
        # Check if resource is in the project
        if resource_id not in resource_list:
            logger.warning(f"{resource_type.capitalize()} {resource_id} is not in project {self.id}")
            return False
        
        # Remove resource
        resource_list.remove(resource_id)
        
        # Update project
        self.updated_at = datetime.datetime.now()
        
        # Log activity
        self.log_activity(
            user_id=self.owner_id,
            action=f"remove_{resource_type}",
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                "project_id": self.id,
                "project_name": self.name
            }
        )
        
        logger.info(f"Removed {resource_type} {resource_id} from project {self.name} ({self.id})")
        return True
    
    def add_comment(self, user_id: str, content: str, resource_type: str, resource_id: str, 
                   resource_section: Optional[str] = None, parent_id: Optional[str] = None) -> Comment:
        """
        Add a comment to a resource.
        
        Args:
            user_id: User ID
            content: Comment content
            resource_type: Resource type (experiment, publication, dataset, project)
            resource_id: Resource ID
            resource_section: Specific section or part of the resource
            parent_id: Parent comment ID for threaded comments
            
        Returns:
            Added comment
        """
        # Check if user is a collaborator
        if not self.is_collaborator(user_id):
            logger.error(f"User {user_id} is not a collaborator on project {self.id}")
            raise ValueError(f"User {user_id} is not a collaborator on project {self.id}")
        
        # Check if parent comment exists
        if parent_id:
            parent_exists = False
            for comment in self.comments:
                if comment.id == parent_id:
                    parent_exists = True
                    break
            
            if not parent_exists:
                logger.error(f"Parent comment {parent_id} not found in project {self.id}")
                raise ValueError(f"Parent comment {parent_id} not found in project {self.id}")
        
        # Create comment
        comment = Comment(
            id=str(uuid.uuid4()),
            author_id=user_id,
            content=content,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_section=resource_section,
            parent_id=parent_id
        )
        
        # Add to comments list
        self.comments.append(comment)
        
        # Update project
        self.updated_at = datetime.datetime.now()
        
        # Log activity
        self.log_activity(
            user_id=user_id,
            action="add_comment",
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                "comment_id": comment.id,
                "parent_id": parent_id,
                "resource_section": resource_section
            }
        )
        
        logger.info(f"Added comment {comment.id} to {resource_type} {resource_id} in project {self.name} ({self.id})")
        return comment
    
    def update_comment(self, user_id: str, comment_id: str, content: str) -> bool:
        """
        Update a comment.
        
        Args:
            user_id: User ID
            comment_id: Comment ID
            content: New comment content
            
        Returns:
            True if comment was updated, False otherwise
        """
        # Check if user is a collaborator
        if not self.is_collaborator(user_id):
            logger.error(f"User {user_id} is not a collaborator on project {self.id}")
            return False
        
        # Find comment
        for comment in self.comments:
            if comment.id == comment_id:
                # Check if user is the author
                if comment.author_id != user_id:
                    # Check if user is an admin or owner
                    user_role = self.get_user_role(user_id)
                    if user_role not in [CollaborationRole.OWNER, CollaborationRole.ADMIN]:
                        logger.error(f"User {user_id} is not authorized to update comment {comment_id}")
                        return False
                
                # Update comment
                comment.content = content
                comment.updated_at = datetime.datetime.now()
                
                # Update project
                self.updated_at = datetime.datetime.now()
                
                # Log activity
                self.log_activity(
                    user_id=user_id,
                    action="update_comment",
                    resource_type=comment.resource_type,
                    resource_id=comment.resource_id,
                    details={
                        "comment_id": comment_id
                    }
                )
                
                logger.info(f"Updated comment {comment_id} in project {self.name} ({self.id})")
                return True
        
        logger.warning(f"Comment {comment_id} not found in project {self.id}")
        return False
    
    def delete_comment(self, user_id: str, comment_id: str) -> bool:
        """
        Delete a comment.
        
        Args:
            user_id: User ID
            comment_id: Comment ID
            
        Returns:
            True if comment was deleted, False otherwise
        """
        # Check if user is a collaborator
        if not self.is_collaborator(user_id):
            logger.error(f"User {user_id} is not a collaborator on project {self.id}")
            return False
        
        # Find comment
        for i, comment in enumerate(self.comments):
            if comment.id == comment_id:
                # Check if user is the author
                if comment.author_id != user_id:
                    # Check if user is an admin or owner
                    user_role = self.get_user_role(user_id)
                    if user_role not in [CollaborationRole.OWNER, CollaborationRole.ADMIN]:
                        logger.error(f"User {user_id} is not authorized to delete comment {comment_id}")
                        return False
                
                # Remove comment
                removed = self.comments.pop(i)
                
                # Update project
                self.updated_at = datetime.datetime.now()
                
                # Log activity
                self.log_activity(
                    user_id=user_id,
                    action="delete_comment",
                    resource_type=removed.resource_type,
                    resource_id=removed.resource_id,
                    details={
                        "comment_id": comment_id
                    }
                )
                
                logger.info(f"Deleted comment {comment_id} from project {self.name} ({self.id})")
                return True
        
        logger.warning(f"Comment {comment_id} not found in project {self.id}")
        return False
    
    def resolve_comment(self, user_id: str, comment_id: str) -> bool:
        """
        Mark a comment as resolved.
        
        Args:
            user_id: User ID
            comment_id: Comment ID
            
        Returns:
            True if comment was resolved, False otherwise
        """
        # Check if user is a collaborator
        if not self.is_collaborator(user_id):
            logger.error(f"User {user_id} is not a collaborator on project {self.id}")
            return False
        
        # Find comment
        for comment in self.comments:
            if comment.id == comment_id:
                # Check if comment is already resolved
                if comment.resolved:
                    logger.warning(f"Comment {comment_id} is already resolved")
                    return True
                
                # Resolve comment
                comment.resolved = True
                
                # Update project
                self.updated_at = datetime.datetime.now()
                
                # Log activity
                self.log_activity(
                    user_id=user_id,
                    action="resolve_comment",
                    resource_type=comment.resource_type,
                    resource_id=comment.resource_id,
                    details={
                        "comment_id": comment_id
                    }
                )
                
                logger.info(f"Resolved comment {comment_id} in project {self.name} ({self.id})")
                return True
        
        logger.warning(f"Comment {comment_id} not found in project {self.id}")
        return False
    
    def set_resource_permission(self, user_id: str, resource_type: str, resource_id: str, 
                              permission: PermissionLevel, target_user_id: str,
                              expires_at: Optional[datetime.datetime] = None) -> ResourcePermission:
        """
        Set permission for a resource.
        
        Args:
            user_id: User ID setting the permission
            resource_type: Resource type (experiment, publication, dataset)
            resource_id: Resource ID
            permission: Permission level
            target_user_id: User ID to grant permission to
            expires_at: Expiration date for the permission
            
        Returns:
            Created resource permission
        """
        # Check if user is authorized to set permissions
        user_role = self.get_user_role(user_id)
        if user_role not in [CollaborationRole.OWNER, CollaborationRole.ADMIN]:
            logger.error(f"User {user_id} is not authorized to set permissions in project {self.id}")
            raise ValueError(f"User {user_id} is not authorized to set permissions in project {self.id}")
        
        # Check if target user exists
        if not self.is_collaborator(target_user_id) and target_user_id != "public":
            logger.error(f"Target user {target_user_id} is not a collaborator on project {self.id}")
            raise ValueError(f"Target user {target_user_id} is not a collaborator on project {self.id}")
        
        # Check if resource exists in the project
        if resource_type == "experiment" and resource_id not in self.experiments:
            logger.error(f"Experiment {resource_id} not found in project {self.id}")
            raise ValueError(f"Experiment {resource_id} not found in project {self.id}")
        elif resource_type == "publication" and resource_id not in self.publications:
            logger.error(f"Publication {resource_id} not found in project {self.id}")
            raise ValueError(f"Publication {resource_id} not found in project {self.id}")
        elif resource_type == "dataset" and resource_id not in self.datasets:
            logger.error(f"Dataset {resource_id} not found in project {self.id}")
            raise ValueError(f"Dataset {resource_id} not found in project {self.id}")
        elif resource_type != "project" and resource_id != self.id:
            logger.error(f"Invalid resource type: {resource_type}")
            raise ValueError(f"Invalid resource type: {resource_type}")
        
        # Remove existing permission for the same resource and user
        for i, perm in enumerate(self.permissions):
            if (perm.resource_type == resource_type and 
                perm.resource_id == resource_id and 
                perm.user_id == target_user_id):
                self.permissions.pop(i)
                break
        
        # Create permission
        resource_permission = ResourcePermission(
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=target_user_id,
            permission=permission,
            granted_by=user_id,
            expires_at=expires_at
        )
        
        # Add to permissions list
        self.permissions.append(resource_permission)
        
        # Update project
        self.updated_at = datetime.datetime.now()
        
        # Log activity
        self.log_activity(
            user_id=user_id,
            action="set_permission",
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                "target_user_id": target_user_id,
                "permission": permission.value,
                "expires_at": expires_at.isoformat() if expires_at else None
            }
        )
        
        logger.info(f"Set {permission.value} permission for {target_user_id} on {resource_type} {resource_id} in project {self.name} ({self.id})")
        return resource_permission
    
    def get_resource_permission(self, user_id: str, resource_type: str, resource_id: str) -> PermissionLevel:
        """
        Get permission level for a resource.
        
        Args:
            user_id: User ID
            resource_type: Resource type (experiment, publication, dataset, project)
            resource_id: Resource ID
            
        Returns:
            Permission level
        """
        # Check if user is the owner
        if user_id == self.owner_id:
            return PermissionLevel.MANAGE
        
        # Check if user is an admin
        user_role = self.get_user_role(user_id)
        if user_role == CollaborationRole.ADMIN:
            return PermissionLevel.MANAGE
        
        # Check for specific permission
        for perm in self.permissions:
            if (perm.resource_type == resource_type and 
                perm.resource_id == resource_id and 
                perm.user_id == user_id):
                # Check if permission has expired
                if perm.expires_at and perm.expires_at < datetime.datetime.now():
                    continue
                
                return perm.permission
        
        # Check for public permission
        for perm in self.permissions:
            if (perm.resource_type == resource_type and 
                perm.resource_id == resource_id and 
                perm.user_id == "public"):
                # Check if permission has expired
                if perm.expires_at and perm.expires_at < datetime.datetime.now():
                    continue
                
                return perm.permission
        
        # Default permissions based on role
        if user_role == CollaborationRole.CONTRIBUTOR:
            return PermissionLevel.EDIT
        elif user_role == CollaborationRole.REVIEWER:
            return PermissionLevel.COMMENT
        elif user_role == CollaborationRole.VIEWER:
            return PermissionLevel.READ
        
        # Default to no permission
        return PermissionLevel.NONE
    
    def log_activity(self, user_id: str, action: str, resource_type: str, resource_id: str, 
                    details: Dict[str, Any] = None) -> ActivityLog:
        """
        Log an activity.
        
        Args:
            user_id: User ID
            action: Action performed
            resource_type: Resource type (experiment, publication, dataset, project)
            resource_id: Resource ID
            details: Additional details
            
        Returns:
            Created activity log
        """
        # Create activity log
        activity = ActivityLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {}
        )
        
        # Add to activity log
        self.activity_log.append(activity)
        
        logger.debug(f"Logged activity: {user_id} {action} {resource_type} {resource_id}")
        return activity
    
    def record_change(self, user_id: str, resource_type: str, resource_id: str, 
                     field: str, old_value: Any, new_value: Any, 
                     description: Optional[str] = None) -> ChangeRecord:
        """
        Record a change to a resource.
        
        Args:
            user_id: User ID
            resource_type: Resource type (experiment, publication, dataset, project)
            resource_id: Resource ID
            field: Field that was changed
            old_value: Old value
            new_value: New value
            description: Description of the change
            
        Returns:
            Created change record
        """
        # Create change record
        change = ChangeRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            field=field,
            old_value=old_value,
            new_value=new_value,
            description=description
        )
        
        # Log activity
        self.log_activity(
            user_id=user_id,
            action="record_change",
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                "change_id": change.id,
                "field": field,
                "description": description
            }
        )
        
        logger.info(f"Recorded change to {field} in {resource_type} {resource_id} by {user_id}")
        return change
    
    def is_collaborator(self, user_id: str) -> bool:
        """
        Check if a user is a collaborator on the project.
        
        Args:
            user_id: User ID
            
        Returns:
            True if user is a collaborator, False otherwise
        """
        if user_id == self.owner_id:
            return True
        
        for collaborator in self.collaborators:
            if collaborator.id == user_id:
                return True
        
        return False
    
    def get_user_role(self, user_id: str) -> Optional[CollaborationRole]:
        """
        Get a user's role in the project.
        
        Args:
            user_id: User ID
            
        Returns:
            User's role, or None if user is not a collaborator
        """
        if user_id == self.owner_id:
            return CollaborationRole.OWNER
        
        for collaborator in self.collaborators:
            if collaborator.id == user_id:
                return collaborator.role
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the project to a dictionary.
        
        Returns:
            Dictionary representation of the project
        """
        # Convert to dictionary using dataclasses.asdict
        result = asdict(self)
        
        # Convert datetime objects to ISO format strings
        for key in ['created_at', 'updated_at']:
            if result[key] is not None:
                result[key] = result[key].isoformat()
        
        # Convert collaborators
        for i, collaborator in enumerate(result['collaborators']):
            if collaborator['joined_at'] is not None:
                result['collaborators'][i]['joined_at'] = collaborator['joined_at'].isoformat()
            if collaborator['last_active'] is not None:
                result['collaborators'][i]['last_active'] = collaborator['last_active'].isoformat()
            result['collaborators'][i]['role'] = collaborator['role'].value
        
        # Convert comments
        for i, comment in enumerate(result['comments']):
            if comment['created_at'] is not None:
                result['comments'][i]['created_at'] = comment['created_at'].isoformat()
            if comment['updated_at'] is not None:
                result['comments'][i]['updated_at'] = comment['updated_at'].isoformat()
        
        # Convert activity log
        for i, activity in enumerate(result['activity_log']):
            if activity['timestamp'] is not None:
                result['activity_log'][i]['timestamp'] = activity['timestamp'].isoformat()
        
        # Convert permissions
        for i, permission in enumerate(result['permissions']):
            if permission['granted_at'] is not None:
                result['permissions'][i]['granted_at'] = permission['granted_at'].isoformat()
            if permission['expires_at'] is not None:
                result['permissions'][i]['expires_at'] = permission['expires_at'].isoformat()
            result['permissions'][i]['permission'] = permission['permission'].value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollaborationProject':
        """
        Create a project from a dictionary.
        
        Args:
            data: Dictionary representation of the project
            
        Returns:
            CollaborationProject object
        """
        # Create a copy of the data to avoid modifying the original
        data = data.copy()
        
        # Convert ISO format strings to datetime objects
        for key in ['created_at', 'updated_at']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.datetime.fromisoformat(data[key])
        
        # Convert collaborators
        collaborators = []
        for collaborator_dict in data.get('collaborators', []):
            # Convert datetime strings
            if collaborator_dict.get('joined_at') and isinstance(collaborator_dict['joined_at'], str):
                collaborator_dict['joined_at'] = datetime.datetime.fromisoformat(collaborator_dict['joined_at'])
            if collaborator_dict.get('last_active') and isinstance(collaborator_dict['last_active'], str):
                collaborator_dict['last_active'] = datetime.datetime.fromisoformat(collaborator_dict['last_active'])
            
            # Convert role string to enum
            if collaborator_dict.get('role') and isinstance(collaborator_dict['role'], str):
                collaborator_dict['role'] = CollaborationRole(collaborator_dict['role'])
            
            collaborators.append(Collaborator(**collaborator_dict))
        data['collaborators'] = collaborators
        
        # Convert comments
        comments = []
        for comment_dict in data.get('comments', []):
            # Convert datetime strings
            if comment_dict.get('created_at') and isinstance(comment_dict['created_at'], str):
                comment_dict['created_at'] = datetime.datetime.fromisoformat(comment_dict['created_at'])
            if comment_dict.get('updated_at') and isinstance(comment_dict['updated_at'], str):
                comment_dict['updated_at'] = datetime.datetime.fromisoformat(comment_dict['updated_at'])
            
            comments.append(Comment(**comment_dict))
        data['comments'] = comments
        
        # Convert activity log
        activity_log = []
        for activity_dict in data.get('activity_log', []):
            # Convert datetime strings
            if activity_dict.get('timestamp') and isinstance(activity_dict['timestamp'], str):
                activity_dict['timestamp'] = datetime.datetime.fromisoformat(activity_dict['timestamp'])
            
            activity_log.append(ActivityLog(**activity_dict))
        data['activity_log'] = activity_log
        
        # Convert permissions
        permissions = []
        for permission_dict in data.get('permissions', []):
            # Convert datetime strings
            if permission_dict.get('granted_at') and isinstance(permission_dict['granted_at'], str):
                permission_dict['granted_at'] = datetime.datetime.fromisoformat(permission_dict['granted_at'])
            if permission_dict.get('expires_at') and isinstance(permission_dict['expires_at'], str):
                permission_dict['expires_at'] = datetime.datetime.fromisoformat(permission_dict['expires_at'])
            
            # Convert permission string to enum
            if permission_dict.get('permission') and isinstance(permission_dict['permission'], str):
                permission_dict['permission'] = PermissionLevel(permission_dict['permission'])
            
            permissions.append(ResourcePermission(**permission_dict))
        data['permissions'] = permissions
        
        # Create project object
        return cls(**data)
    
    def save(self, directory: str) -> str:
        """
        Save the project to a file.
        
        Args:
            directory: Directory to save the project
            
        Returns:
            Path to the saved project file
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
        
        logger.info(f"Saved project '{self.title}' to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'CollaborationProject':
        """
        Load a project from a file.
        
        Args:
            filepath: Path to the project file
            
        Returns:
            Loaded project
        """
        # Load from file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create project object
        project = cls.from_dict(data)
        
        logger.info(f"Loaded project '{project.name}' from {filepath}")
        return project


class CollaborationManager:
    """
    Manager for collaborative research projects.
    
    This class provides methods for creating, managing, and accessing
    collaborative research projects.
    """
    
    def __init__(self, storage_dir: str = "collaboration"):
        """
        Initialize the CollaborationManager.
        
        Args:
            storage_dir: Directory for storing collaboration data
        """
        self.storage_dir = storage_dir
        self.projects = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        logger.info(f"Initialized CollaborationManager with storage directory: {storage_dir}")
    
    def create_project(self, name: str, description: str, owner_id: str, 
                      visibility: str = "private") -> CollaborationProject:
        """
        Create a new collaborative research project.
        
        Args:
            name: Project name
            description: Project description
            owner_id: Owner user ID
            visibility: Project visibility (private, team, public)
            
        Returns:
            Created project
        """
        # Create project
        project = CollaborationProject(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            owner_id=owner_id,
            visibility=visibility
        )
        
        # Add to projects dictionary
        self.projects[project.id] = project
        
        # Save project
        self.save_project(project.id)
        
        logger.info(f"Created project '{name}' with ID {project.id}")
        return project
    
    def get_project(self, project_id: str) -> Optional[CollaborationProject]:
        """
        Get a project by ID.
        
        Args:
            project_id: Project ID
            
        Returns:
            Project object or None if not found
        """
        return self.projects.get(project_id)
    
    def list_projects(self, user_id: Optional[str] = None, 
                     visibility: Optional[str] = None) -> List[CollaborationProject]:
        """
        List projects.
        
        Args:
            user_id: Filter by user ID (projects where user is a collaborator)
            visibility: Filter by visibility
            
        Returns:
            List of projects
        """
        projects = list(self.projects.values())
        
        # Filter by user ID
        if user_id:
            projects = [p for p in projects if p.is_collaborator(user_id)]
        
        # Filter by visibility
        if visibility:
            projects = [p for p in projects if p.visibility == visibility]
        
        return projects
    
    def save_project(self, project_id: str) -> Optional[str]:
        """
        Save a project to disk.
        
        Args:
            project_id: Project ID
            
        Returns:
            Path to the saved project file, or None if project not found
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return None
        
        return project.save(self.storage_dir)
    
    def load_project(self, filepath: str) -> Optional[CollaborationProject]:
        """
        Load a project from disk.
        
        Args:
            filepath: Path to the project file
            
        Returns:
            Loaded project, or None if loading failed
        """
        try:
            project = CollaborationProject.load(filepath)
            self.projects[project.id] = project
            return project
        except Exception as e:
            logger.error(f"Error loading project from {filepath}: {str(e)}")
            return None
    
    def load_all_projects(self) -> int:
        """
        Load all projects from the storage directory.
        
        Returns:
            Number of projects loaded
        """
        count = 0
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_dir, filename)
                project = self.load_project(filepath)
                if project is not None:
                    count += 1
        
        logger.info(f"Loaded {count} projects from {self.storage_dir}")
        return count
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if deletion was successful, False otherwise
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return False
        
        # Remove from projects dictionary
        del self.projects[project_id]
        
        # Delete project file if it exists
        filepath = os.path.join(self.storage_dir, f"{project_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        
        logger.info(f"Deleted project '{project.name}' with ID {project_id}")
        return True
    
    def add_collaborator(self, project_id: str, user_id: str, name: str, email: str, 
                        role: CollaborationRole) -> Optional[Collaborator]:
        """
        Add a collaborator to a project.
        
        Args:
            project_id: Project ID
            user_id: User ID
            name: User name
            email: User email
            role: Collaboration role
            
        Returns:
            Added collaborator, or None if project not found
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return None
        
        # Add collaborator
        collaborator = project.add_collaborator(user_id, name, email, role)
        
        # Save project
        self.save_project(project_id)
        
        return collaborator
    
    def remove_collaborator(self, project_id: str, user_id: str) -> bool:
        """
        Remove a collaborator from a project.
        
        Args:
            project_id: Project ID
            user_id: User ID
            
        Returns:
            True if collaborator was removed, False otherwise
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return False
        
        # Remove collaborator
        success = project.remove_collaborator(user_id)
        
        # Save project if successful
        if success:
            self.save_project(project_id)
        
        return success
    
    def add_resource(self, project_id: str, resource_type: str, resource_id: str) -> bool:
        """
        Add a resource to a project.
        
        Args:
            project_id: Project ID
            resource_type: Resource type (experiment, publication, dataset)
            resource_id: Resource ID
            
        Returns:
            True if resource was added, False otherwise
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return False
        
        # Add resource
        success = project.add_resource(resource_type, resource_id)
        
        # Save project if successful
        if success:
            self.save_project(project_id)
        
        return success
    
    def remove_resource(self, project_id: str, resource_type: str, resource_id: str) -> bool:
        """
        Remove a resource from a project.
        
        Args:
            project_id: Project ID
            resource_type: Resource type (experiment, publication, dataset)
            resource_id: Resource ID
            
        Returns:
            True if resource was removed, False otherwise
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return False
        
        # Remove resource
        success = project.remove_resource(resource_type, resource_id)
        
        # Save project if successful
        if success:
            self.save_project(project_id)
        
        return success
    
    def add_comment(self, project_id: str, user_id: str, content: str, 
                   resource_type: str, resource_id: str,
                   resource_section: Optional[str] = None, 
                   parent_id: Optional[str] = None) -> Optional[Comment]:
        """
        Add a comment to a resource in a project.
        
        Args:
            project_id: Project ID
            user_id: User ID
            content: Comment content
            resource_type: Resource type (experiment, publication, dataset, project)
            resource_id: Resource ID
            resource_section: Specific section or part of the resource
            parent_id: Parent comment ID for threaded comments
            
        Returns:
            Added comment, or None if project not found
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return None
        
        try:
            # Add comment
            comment = project.add_comment(
                user_id=user_id,
                content=content,
                resource_type=resource_type,
                resource_id=resource_id,
                resource_section=resource_section,
                parent_id=parent_id
            )
            
            # Save project
            self.save_project(project_id)
            
            return comment
        except ValueError as e:
            logger.error(f"Error adding comment to project {project_id}: {str(e)}")
            return None
    
    def set_resource_permission(self, project_id: str, user_id: str, 
                              resource_type: str, resource_id: str,
                              permission: PermissionLevel, target_user_id: str,
                              expires_at: Optional[datetime.datetime] = None) -> Optional[ResourcePermission]:
        """
        Set permission for a resource in a project.
        
        Args:
            project_id: Project ID
            user_id: User ID setting the permission
            resource_type: Resource type (experiment, publication, dataset, project)
            resource_id: Resource ID
            permission: Permission level
            target_user_id: User ID to grant permission to
            expires_at: Expiration date for the permission
            
        Returns:
            Created resource permission, or None if project not found
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return None
        
        try:
            # Set permission
            resource_permission = project.set_resource_permission(
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                permission=permission,
                target_user_id=target_user_id,
                expires_at=expires_at
            )
            
            # Save project
            self.save_project(project_id)
            
            return resource_permission
        except ValueError as e:
            logger.error(f"Error setting permission in project {project_id}: {str(e)}")
            return None
    
    def get_resource_permission(self, project_id: str, user_id: str, 
                              resource_type: str, resource_id: str) -> PermissionLevel:
        """
        Get permission level for a resource in a project.
        
        Args:
            project_id: Project ID
            user_id: User ID
            resource_type: Resource type (experiment, publication, dataset, project)
            resource_id: Resource ID
            
        Returns:
            Permission level
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return PermissionLevel.NONE
        
        return project.get_resource_permission(user_id, resource_type, resource_id)
    
    def get_user_projects(self, user_id: str) -> List[CollaborationProject]:
        """
        Get projects where a user is a collaborator.
        
        Args:
            user_id: User ID
            
        Returns:
            List of projects
        """
        return [p for p in self.projects.values() if p.is_collaborator(user_id)]
    
    def get_project_collaborators(self, project_id: str) -> List[Collaborator]:
        """
        Get collaborators for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of collaborators
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return []
        
        return project.collaborators
    
    def get_project_comments(self, project_id: str, 
                           resource_type: Optional[str] = None,
                           resource_id: Optional[str] = None,
                           resolved: Optional[bool] = None) -> List[Comment]:
        """
        Get comments for a project.
        
        Args:
            project_id: Project ID
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            resolved: Filter by resolved status
            
        Returns:
            List of comments
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return []
        
        comments = project.comments
        
        # Filter by resource type
        if resource_type:
            comments = [c for c in comments if c.resource_type == resource_type]
        
        # Filter by resource ID
        if resource_id:
            comments = [c for c in comments if c.resource_id == resource_id]
        
        # Filter by resolved status
        if resolved is not None:
            comments = [c for c in comments if c.resolved == resolved]
        
        return comments
    
    def get_project_activity_log(self, project_id: str,
                               user_id: Optional[str] = None,
                               action: Optional[str] = None,
                               resource_type: Optional[str] = None,
                               resource_id: Optional[str] = None,
                               start_time: Optional[datetime.datetime] = None,
                               end_time: Optional[datetime.datetime] = None) -> List[ActivityLog]:
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
            
        Returns:
            List of activity log entries
        """
        project = self.get_project(project_id)
        if project is None:
            logger.error(f"Project with ID {project_id} not found")
            return []
        
        activities = project.activity_log
        
        # Filter by user ID
        if user_id:
            activities = [a for a in activities if a.user_id == user_id]
        
        # Filter by action
        if action:
            activities = [a for a in activities if a.action == action]
        
        # Filter by resource type
        if resource_type:
            activities = [a for a in activities if a.resource_type == resource_type]
        
        # Filter by resource ID
        if resource_id:
            activities = [a for a in activities if a.resource_id == resource_id]
        
        # Filter by start time
        if start_time:
            activities = [a for a in activities if a.timestamp >= start_time]
        
        # Filter by end time
        if end_time:
            activities = [a for a in activities if a.timestamp <= end_time]
        
        return activities


# Create a singleton instance for easy import
collaboration_manager = None

def get_collaboration_manager(storage_dir: str = "collaboration") -> CollaborationManager:
    """
    Get the collaboration manager singleton instance.
    
    Args:
        storage_dir: Directory for storing collaboration data
        
    Returns:
        CollaborationManager instance
    """
    global collaboration_manager
    
    if collaboration_manager is None:
        collaboration_manager = CollaborationManager(storage_dir=storage_dir)
    
    return collaboration_manager
