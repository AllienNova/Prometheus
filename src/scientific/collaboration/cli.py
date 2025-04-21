"""
Command-line interface for the collaboration system in the Prometheus AI Automation Platform.

This module provides a command-line interface for managing collaborative research projects,
team members, permissions, and comments.
"""

import os
import sys
import json
import argparse
import datetime
from typing import List, Dict, Any, Optional

from .collaboration import (
    CollaborationRole, PermissionLevel, Collaborator, Comment, ActivityLog,
    ResourcePermission, CollaborationProject, get_collaboration_manager
)

def parse_datetime(datetime_str: str) -> datetime.datetime:
    """
    Parse a datetime string.
    
    Args:
        datetime_str: Datetime string in ISO format
        
    Returns:
        Datetime object
    """
    try:
        return datetime.datetime.fromisoformat(datetime_str)
    except ValueError:
        raise ValueError(f"Invalid datetime format: {datetime_str}. Expected ISO format (YYYY-MM-DDTHH:MM:SS).")

def format_datetime(dt: datetime.datetime) -> str:
    """
    Format a datetime object for display.
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted datetime string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def list_projects(args):
    """
    List projects.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    projects = manager.list_projects(user_id=args.user_id, visibility=args.visibility)
    
    if not projects:
        print("No projects found.")
        return
    
    print(f"Found {len(projects)} projects:")
    for project in projects:
        print(f"  - {project.id}: {project.name} ({project.visibility}, {project.status})")
        print(f"    Owner: {project.owner_id}")
        print(f"    Created: {format_datetime(project.created_at)}")
        print(f"    Updated: {format_datetime(project.updated_at)}")
        print(f"    Collaborators: {len(project.collaborators)}")
        print(f"    Resources: {len(project.experiments)} experiments, {len(project.publications)} publications, {len(project.datasets)} datasets")
        print()

def create_project(args):
    """
    Create a new project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    
    project = manager.create_project(
        name=args.name,
        description=args.description,
        owner_id=args.owner_id,
        visibility=args.visibility
    )
    
    print(f"Created project '{project.name}' with ID {project.id}")
    print(f"Owner: {project.owner_id}")
    print(f"Visibility: {project.visibility}")
    print(f"Created at: {format_datetime(project.created_at)}")

def show_project(args):
    """
    Show project details.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(args.project_id)
    
    if project is None:
        print(f"Project with ID {args.project_id} not found.")
        return
    
    print(f"Project: {project.name} ({project.id})")
    print(f"Description: {project.description}")
    print(f"Owner: {project.owner_id}")
    print(f"Status: {project.status}")
    print(f"Visibility: {project.visibility}")
    print(f"Created: {format_datetime(project.created_at)}")
    print(f"Updated: {format_datetime(project.updated_at)}")
    print()
    
    print("Collaborators:")
    if not project.collaborators:
        print("  No collaborators.")
    else:
        for collaborator in project.collaborators:
            print(f"  - {collaborator.name} ({collaborator.id}): {collaborator.role.value}")
            print(f"    Email: {collaborator.email}")
            print(f"    Joined: {format_datetime(collaborator.joined_at)}")
            if collaborator.last_active:
                print(f"    Last active: {format_datetime(collaborator.last_active)}")
            print()
    
    print("Experiments:")
    if not project.experiments:
        print("  No experiments.")
    else:
        for experiment_id in project.experiments:
            print(f"  - {experiment_id}")
    
    print("Publications:")
    if not project.publications:
        print("  No publications.")
    else:
        for publication_id in project.publications:
            print(f"  - {publication_id}")
    
    print("Datasets:")
    if not project.datasets:
        print("  No datasets.")
    else:
        for dataset_id in project.datasets:
            print(f"  - {dataset_id}")

def update_project(args):
    """
    Update a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(args.project_id)
    
    if project is None:
        print(f"Project with ID {args.project_id} not found.")
        return
    
    # Update fields
    if args.name is not None:
        project.name = args.name
    
    if args.description is not None:
        project.description = args.description
    
    if args.visibility is not None:
        project.visibility = args.visibility
    
    if args.status is not None:
        project.status = args.status
    
    # Update timestamp
    project.updated_at = datetime.datetime.now()
    
    # Save project
    manager.save_project(project.id)
    
    print(f"Updated project '{project.name}' ({project.id})")

def delete_project(args):
    """
    Delete a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(args.project_id)
    
    if project is None:
        print(f"Project with ID {args.project_id} not found.")
        return
    
    # Confirm deletion
    if not args.force:
        confirm = input(f"Are you sure you want to delete project '{project.name}' ({project.id})? [y/N] ")
        if confirm.lower() != 'y':
            print("Deletion cancelled.")
            return
    
    success = manager.delete_project(args.project_id)
    
    if success:
        print(f"Deleted project '{project.name}' ({project.id})")
    else:
        print(f"Failed to delete project with ID {args.project_id}")

def list_collaborators(args):
    """
    List collaborators for a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(args.project_id)
    
    if project is None:
        print(f"Project with ID {args.project_id} not found.")
        return
    
    print(f"Collaborators for project '{project.name}' ({project.id}):")
    
    # Add owner
    print(f"  - {project.owner_id} (owner)")
    
    # Add collaborators
    for collaborator in project.collaborators:
        print(f"  - {collaborator.name} ({collaborator.id}): {collaborator.role.value}")
        print(f"    Email: {collaborator.email}")
        print(f"    Joined: {format_datetime(collaborator.joined_at)}")
        if collaborator.last_active:
            print(f"    Last active: {format_datetime(collaborator.last_active)}")
        print()

def add_collaborator(args):
    """
    Add a collaborator to a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    # Parse role
    try:
        role = CollaborationRole(args.role)
    except ValueError:
        print(f"Invalid role: {args.role}")
        print(f"Valid roles: {', '.join([r.value for r in CollaborationRole])}")
        return
    
    # Add collaborator
    collaborator = manager.add_collaborator(
        project_id=args.project_id,
        user_id=args.user_id,
        name=args.name,
        email=args.email,
        role=role
    )
    
    if collaborator is None:
        print(f"Failed to add collaborator to project with ID {args.project_id}")
        return
    
    print(f"Added collaborator {collaborator.name} ({collaborator.id}) to project with ID {args.project_id}")
    print(f"Role: {collaborator.role.value}")
    print(f"Email: {collaborator.email}")
    print(f"Joined: {format_datetime(collaborator.joined_at)}")

def remove_collaborator(args):
    """
    Remove a collaborator from a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    # Remove collaborator
    success = manager.remove_collaborator(args.project_id, args.user_id)
    
    if success:
        print(f"Removed collaborator with ID {args.user_id} from project with ID {args.project_id}")
    else:
        print(f"Failed to remove collaborator with ID {args.user_id} from project with ID {args.project_id}")

def list_comments(args):
    """
    List comments for a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(args.project_id)
    
    if project is None:
        print(f"Project with ID {args.project_id} not found.")
        return
    
    # Get comments
    comments = manager.get_project_comments(
        project_id=args.project_id,
        resource_type=args.resource_type,
        resource_id=args.resource_id,
        resolved=args.resolved
    )
    
    if not comments:
        print("No comments found.")
        return
    
    print(f"Found {len(comments)} comments:")
    for comment in comments:
        print(f"  - {comment.id} by {comment.author_id}")
        print(f"    Created: {format_datetime(comment.created_at)}")
        if comment.updated_at:
            print(f"    Updated: {format_datetime(comment.updated_at)}")
        print(f"    Resource: {comment.resource_type} {comment.resource_id}")
        if comment.resource_section:
            print(f"    Section: {comment.resource_section}")
        if comment.parent_id:
            print(f"    Reply to: {comment.parent_id}")
        print(f"    Resolved: {comment.resolved}")
        print(f"    Content: {comment.content}")
        print()

def add_comment(args):
    """
    Add a comment to a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    # Add comment
    comment = manager.add_comment(
        project_id=args.project_id,
        user_id=args.user_id,
        content=args.content,
        resource_type=args.resource_type,
        resource_id=args.resource_id,
        resource_section=args.resource_section,
        parent_id=args.parent_id
    )
    
    if comment is None:
        print(f"Failed to add comment to project with ID {args.project_id}")
        return
    
    print(f"Added comment with ID {comment.id} to project with ID {args.project_id}")
    print(f"Author: {comment.author_id}")
    print(f"Created: {format_datetime(comment.created_at)}")
    print(f"Resource: {comment.resource_type} {comment.resource_id}")
    if comment.resource_section:
        print(f"Section: {comment.resource_section}")
    if comment.parent_id:
        print(f"Reply to: {comment.parent_id}")
    print(f"Content: {comment.content}")

def resolve_comment(args):
    """
    Resolve a comment.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(args.project_id)
    
    if project is None:
        print(f"Project with ID {args.project_id} not found.")
        return
    
    # Resolve comment
    success = project.resolve_comment(args.user_id, args.comment_id)
    
    if success:
        # Save project
        manager.save_project(project.id)
        print(f"Resolved comment with ID {args.comment_id}")
    else:
        print(f"Failed to resolve comment with ID {args.comment_id}")

def list_activity(args):
    """
    List activity log for a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(args.project_id)
    
    if project is None:
        print(f"Project with ID {args.project_id} not found.")
        return
    
    # Parse start and end times
    start_time = None
    if args.start_time:
        try:
            start_time = parse_datetime(args.start_time)
        except ValueError as e:
            print(str(e))
            return
    
    end_time = None
    if args.end_time:
        try:
            end_time = parse_datetime(args.end_time)
        except ValueError as e:
            print(str(e))
            return
    
    # Get activity log
    activities = manager.get_project_activity_log(
        project_id=args.project_id,
        user_id=args.user_id,
        action=args.action,
        resource_type=args.resource_type,
        resource_id=args.resource_id,
        start_time=start_time,
        end_time=end_time
    )
    
    if not activities:
        print("No activities found.")
        return
    
    print(f"Found {len(activities)} activities:")
    for activity in activities:
        print(f"  - {activity.id}")
        print(f"    User: {activity.user_id}")
        print(f"    Action: {activity.action}")
        print(f"    Timestamp: {format_datetime(activity.timestamp)}")
        print(f"    Resource: {activity.resource_type} {activity.resource_id}")
        if activity.details:
            print(f"    Details: {json.dumps(activity.details, indent=2)}")
        print()

def set_permission(args):
    """
    Set permission for a resource in a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    # Parse permission
    try:
        permission = PermissionLevel(args.permission)
    except ValueError:
        print(f"Invalid permission: {args.permission}")
        print(f"Valid permissions: {', '.join([p.value for p in PermissionLevel])}")
        return
    
    # Parse expiration
    expires_at = None
    if args.expires_at:
        try:
            expires_at = parse_datetime(args.expires_at)
        except ValueError as e:
            print(str(e))
            return
    
    # Set permission
    resource_permission = manager.set_resource_permission(
        project_id=args.project_id,
        user_id=args.user_id,
        resource_type=args.resource_type,
        resource_id=args.resource_id,
        permission=permission,
        target_user_id=args.target_user_id,
        expires_at=expires_at
    )
    
    if resource_permission is None:
        print(f"Failed to set permission in project with ID {args.project_id}")
        return
    
    print(f"Set {permission.value} permission for {args.target_user_id} on {args.resource_type} {args.resource_id}")
    print(f"Granted by: {resource_permission.granted_by}")
    print(f"Granted at: {format_datetime(resource_permission.granted_at)}")
    if resource_permission.expires_at:
        print(f"Expires at: {format_datetime(resource_permission.expires_at)}")

def get_permission(args):
    """
    Get permission level for a resource in a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    project = manager.get_project(args.project_id)
    
    if project is None:
        print(f"Project with ID {args.project_id} not found.")
        return
    
    # Get permission
    permission = project.get_resource_permission(args.user_id, args.resource_type, args.resource_id)
    
    print(f"Permission for {args.user_id} on {args.resource_type} {args.resource_id}: {permission.value}")

def add_resource(args):
    """
    Add a resource to a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    # Add resource
    success = manager.add_resource(args.project_id, args.resource_type, args.resource_id)
    
    if success:
        print(f"Added {args.resource_type} {args.resource_id} to project with ID {args.project_id}")
    else:
        print(f"Failed to add {args.resource_type} {args.resource_id} to project with ID {args.project_id}")

def remove_resource(args):
    """
    Remove a resource from a project.
    
    Args:
        args: Command-line arguments
    """
    manager = get_collaboration_manager()
    manager.load_all_projects()
    
    # Remove resource
    success = manager.remove_resource(args.project_id, args.resource_type, args.resource_id)
    
    if success:
        print(f"Removed {args.resource_type} {args.resource_id} from project with ID {args.project_id}")
    else:
        print(f"Failed to remove {args.resource_type} {args.resource_id} from project with ID {args.project_id}")

def main():
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="Prometheus Collaboration CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List projects
    list_projects_parser = subparsers.add_parser("list-projects", help="List projects")
    list_projects_parser.add_argument("--user-id", help="Filter by user ID")
    list_projects_parser.add_argument("--visibility", help="Filter by visibility")
    list_projects_parser.set_defaults(func=list_projects)
    
    # Create project
    create_project_parser = subparsers.add_parser("create-project", help="Create a new project")
    create_project_parser.add_argument("name", help="Project name")
    create_project_parser.add_argument("description", help="Project description")
    create_project_parser.add_argument("owner_id", help="Owner user ID")
    create_project_parser.add_argument("--visibility", default="private", help="Project visibility (private, team, public)")
    create_project_parser.set_defaults(func=create_project)
    
    # Show project
    show_project_parser = subparsers.add_parser("show-project", help="Show project details")
    show_project_parser.add_argument("project_id", help="Project ID")
    show_project_parser.set_defaults(func=show_project)
    
    # Update project
    update_project_parser = subparsers.add_parser("update-project", help="Update a project")
    update_project_parser.add_argument("project_id", help="Project ID")
    update_project_parser.add_argument("--name", help="New project name")
    update_project_parser.add_argument("--description", help="New project description")
    update_project_parser.add_argument("--visibility", help="New project visibility (private, team, public)")
    update_project_parser.add_argument("--status", help="New project status")
    update_project_parser.set_defaults(func=update_project)
    
    # Delete project
    delete_project_parser = subparsers.add_parser("delete-project", help="Delete a project")
    delete_project_parser.add_argument("project_id", help="Project ID")
    delete_project_parser.add_argument("--force", action="store_true", help="Force deletion without confirmation")
    delete_project_parser.set_defaults(func=delete_project)
    
    # List collaborators
    list_collaborators_parser = subparsers.add_parser("list-collaborators", help="List collaborators for a project")
    list_collaborators_parser.add_argument("project_id", help="Project ID")
    list_collaborators_parser.set_defaults(func=list_collaborators)
    
    # Add collaborator
    add_collaborator_parser = subparsers.add_parser("add-collaborator", help="Add a collaborator to a project")
    add_collaborator_parser.add_argument("project_id", help="Project ID")
    add_collaborator_parser.add_argument("user_id", help="User ID")
    add_collaborator_parser.add_argument("name", help="User name")
    add_collaborator_parser.add_argument("email", help="User email")
    add_collaborator_parser.add_argument("role", help="Collaboration role")
    add_collaborator_parser.set_defaults(func=add_collaborator)
    
    # Remove collaborator
    remove_collaborator_parser = subparsers.add_parser("remove-collaborator", help="Remove a collaborator from a project")
    remove_collaborator_parser.add_argument("project_id", help="Project ID")
    remove_collaborator_parser.add_argument("user_id", help="User ID")
    remove_collaborator_parser.set_defaults(func=remove_collaborator)
    
    # List comments
    list_comments_parser = subparsers.add_parser("list-comments", help="List comments for a project")
    list_comments_parser.add_argument("project_id", help="Project ID")
    list_comments_parser.add_argument("--resource-type", help="Filter by resource type")
    list_comments_parser.add_argument("--resource-id", help="Filter by resource ID")
    list_comments_parser.add_argument("--resolved", type=bool, help="Filter by resolved status")
    list_comments_parser.set_defaults(func=list_comments)
    
    # Add comment
    add_comment_parser = subparsers.add_parser("add-comment", help="Add a comment to a project")
    add_comment_parser.add_argument("project_id", help="Project ID")
    add_comment_parser.add_argument("user_id", help="User ID")
    add_comment_parser.add_argument("content", help="Comment content")
    add_comment_parser.add_argument("resource_type", help="Resource type")
    add_comment_parser.add_argument("resource_id", help="Resource ID")
    add_comment_parser.add_argument("--resource-section", help="Resource section")
    add_comment_parser.add_argument("--parent-id", help="Parent comment ID")
    add_comment_parser.set_defaults(func=add_comment)
    
    # Resolve comment
    resolve_comment_parser = subparsers.add_parser("resolve-comment", help="Resolve a comment")
    resolve_comment_parser.add_argument("project_id", help="Project ID")
    resolve_comment_parser.add_argument("user_id", help="User ID")
    resolve_comment_parser.add_argument("comment_id", help="Comment ID")
    resolve_comment_parser.set_defaults(func=resolve_comment)
    
    # List activity
    list_activity_parser = subparsers.add_parser("list-activity", help="List activity log for a project")
    list_activity_parser.add_argument("project_id", help="Project ID")
    list_activity_parser.add_argument("--user-id", help="Filter by user ID")
    list_activity_parser.add_argument("--action", help="Filter by action")
    list_activity_parser.add_argument("--resource-type", help="Filter by resource type")
    list_activity_parser.add_argument("--resource-id", help="Filter by resource ID")
    list_activity_parser.add_argument("--start-time", help="Filter by start time (ISO format)")
    list_activity_parser.add_argument("--end-time", help="Filter by end time (ISO format)")
    list_activity_parser.set_defaults(func=list_activity)
    
    # Set permission
    set_permission_parser = subparsers.add_parser("set-permission", help="Set permission for a resource in a project")
    set_permission_parser.add_argument("project_id", help="Project ID")
    set_permission_parser.add_argument("user_id", help="User ID setting the permission")
    set_permission_parser.add_argument("resource_type", help="Resource type")
    set_permission_parser.add_argument("resource_id", help="Resource ID")
    set_permission_parser.add_argument("permission", help="Permission level")
    set_permission_parser.add_argument("target_user_id", help="User ID to grant permission to")
    set_permission_parser.add_argument("--expires-at", help="Expiration date (ISO format)")
    set_permission_parser.set_defaults(func=set_permission)
    
    # Get permission
    get_permission_parser = subparsers.add_parser("get-permission", help="Get permission level for a resource in a project")
    get_permission_parser.add_argument("project_id", help="Project ID")
    get_permission_parser.add_argument("user_id", help="User ID")
    get_permission_parser.add_argument("resource_type", help="Resource type")
    get_permission_parser.add_argument("resource_id", help="Resource ID")
    get_permission_parser.set_defaults(func=get_permission)
    
    # Add resource
    add_resource_parser = subparsers.add_parser("add-resource", help="Add a resource to a project")
    add_resource_parser.add_argument("project_id", help="Project ID")
    add_resource_parser.add_argument("resource_type", help="Resource type (experiment, publication, dataset)")
    add_resource_parser.add_argument("resource_id", help="Resource ID")
    add_resource_parser.set_defaults(func=add_resource)
    
    # Remove resource
    remove_resource_parser = subparsers.add_parser("remove-resource", help="Remove a resource from a project")
    remove_resource_parser.add_argument("project_id", help="Project ID")
    remove_resource_parser.add_argument("resource_type", help="Resource type (experiment, publication, dataset)")
    remove_resource_parser.add_argument("resource_id", help="Resource ID")
    remove_resource_parser.set_defaults(func=remove_resource)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)

if __name__ == "__main__":
    main()
