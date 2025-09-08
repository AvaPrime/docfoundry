#!/usr/bin/env python3
"""
DocFoundry Automated Branching and Merging Strategy System

This module implements a comprehensive branching strategy with automated
merging, conflict resolution, and repository synchronization.

Author: AvaPrime
Version: 1.0.0
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation/logs/branching_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BranchType(Enum):
    """Branch type enumeration"""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    HOTFIX = "hotfix"
    RELEASE = "release"
    EXPERIMENTAL = "experimental"

class MergeStrategy(Enum):
    """Merge strategy enumeration"""
    MERGE_COMMIT = "merge"
    SQUASH = "squash"
    REBASE = "rebase"

@dataclass
class BranchConfig:
    """Branch configuration settings"""
    name: str
    branch_type: BranchType
    base_branch: str = "main"
    merge_strategy: MergeStrategy = MergeStrategy.SQUASH
    auto_merge: bool = False
    require_pr: bool = True
    delete_after_merge: bool = True
    protection_rules: Dict = None

class BranchingStrategy:
    """Main branching strategy management class"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.config_file = self.repo_path / "automation" / "branching_config.json"
        self.ensure_directories()
        self.load_config()
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        dirs = [
            self.repo_path / "automation" / "logs",
            self.repo_path / "automation" / "templates",
            self.repo_path / "automation" / "hooks"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def load_config(self):
        """Load branching configuration"""
        default_config = {
            "default_base_branch": "main",
            "auto_sync_interval": 300,  # 5 minutes
            "branch_naming_patterns": {
                "feature": "feature/{issue_id}-{description}",
                "bugfix": "bugfix/{issue_id}-{description}",
                "hotfix": "hotfix/{version}-{description}",
                "release": "release/{version}",
                "experimental": "experimental/{description}"
            },
            "merge_strategies": {
                "feature": "squash",
                "bugfix": "squash",
                "hotfix": "merge",
                "release": "merge",
                "experimental": "squash"
            },
            "auto_merge_rules": {
                "require_ci_pass": True,
                "require_reviews": 1,
                "require_up_to_date": True,
                "dismiss_stale_reviews": True
            },
            "protection_rules": {
                "main": {
                    "required_status_checks": ["ci"],
                    "enforce_admins": False,
                    "required_pull_request_reviews": {
                        "required_approving_review_count": 1,
                        "dismiss_stale_reviews": True
                    }
                }
            }
        }
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = default_config
            self.save_config()
            
    def save_config(self):
        """Save branching configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def run_command(self, cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
        """Run a shell command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except Exception as e:
            logger.error(f"Command failed: {' '.join(cmd)} - {e}")
            return 1, "", str(e)
            
    def get_current_branch(self) -> str:
        """Get current git branch"""
        code, stdout, stderr = self.run_command(["git", "branch", "--show-current"])
        if code == 0:
            return stdout
        raise Exception(f"Failed to get current branch: {stderr}")
        
    def branch_exists(self, branch_name: str, remote: bool = False) -> bool:
        """Check if branch exists locally or remotely"""
        if remote:
            code, _, _ = self.run_command(["git", "ls-remote", "--heads", "origin", branch_name])
        else:
            code, _, _ = self.run_command(["git", "show-ref", "--verify", f"refs/heads/{branch_name}"])
        return code == 0
        
    def create_branch(self, branch_config: BranchConfig) -> bool:
        """Create a new branch with specified configuration"""
        try:
            logger.info(f"Creating branch: {branch_config.name}")
            
            # Ensure we're on the base branch and it's up to date
            self.checkout_branch(branch_config.base_branch)
            self.sync_branch(branch_config.base_branch)
            
            # Create new branch
            code, stdout, stderr = self.run_command([
                "git", "checkout", "-b", branch_config.name
            ])
            
            if code != 0:
                logger.error(f"Failed to create branch {branch_config.name}: {stderr}")
                return False
                
            # Push branch to remote
            code, stdout, stderr = self.run_command([
                "git", "push", "-u", "origin", branch_config.name
            ])
            
            if code != 0:
                logger.error(f"Failed to push branch {branch_config.name}: {stderr}")
                return False
                
            logger.info(f"Successfully created and pushed branch: {branch_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating branch {branch_config.name}: {e}")
            return False
            
    def checkout_branch(self, branch_name: str) -> bool:
        """Checkout specified branch"""
        try:
            code, stdout, stderr = self.run_command(["git", "checkout", branch_name])
            if code != 0:
                logger.error(f"Failed to checkout branch {branch_name}: {stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking out branch {branch_name}: {e}")
            return False
            
    def sync_branch(self, branch_name: str) -> bool:
        """Sync branch with remote"""
        try:
            # Fetch latest changes
            code, stdout, stderr = self.run_command(["git", "fetch", "origin"])
            if code != 0:
                logger.warning(f"Failed to fetch from origin: {stderr}")
                
            # Pull latest changes if branch exists remotely
            if self.branch_exists(branch_name, remote=True):
                code, stdout, stderr = self.run_command([
                    "git", "pull", "origin", branch_name
                ])
                if code != 0:
                    logger.warning(f"Failed to pull branch {branch_name}: {stderr}")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error syncing branch {branch_name}: {e}")
            return False
            
    def merge_branch(self, source_branch: str, target_branch: str, 
                    strategy: MergeStrategy = MergeStrategy.SQUASH) -> bool:
        """Merge source branch into target branch"""
        try:
            logger.info(f"Merging {source_branch} into {target_branch} using {strategy.value}")
            
            # Checkout target branch and sync
            if not self.checkout_branch(target_branch):
                return False
            if not self.sync_branch(target_branch):
                return False
                
            # Perform merge based on strategy
            if strategy == MergeStrategy.SQUASH:
                cmd = ["git", "merge", "--squash", source_branch]
            elif strategy == MergeStrategy.REBASE:
                cmd = ["git", "rebase", source_branch]
            else:  # MERGE_COMMIT
                cmd = ["git", "merge", "--no-ff", source_branch]
                
            code, stdout, stderr = self.run_command(cmd)
            
            if code != 0:
                logger.error(f"Merge failed: {stderr}")
                return False
                
            # For squash merges, we need to commit
            if strategy == MergeStrategy.SQUASH:
                commit_msg = f"Merge {source_branch} into {target_branch}\n\nSquashed merge of {source_branch}"
                code, stdout, stderr = self.run_command([
                    "git", "commit", "-m", commit_msg
                ])
                if code != 0:
                    logger.error(f"Failed to commit squash merge: {stderr}")
                    return False
                    
            # Push merged changes
            code, stdout, stderr = self.run_command(["git", "push", "origin", target_branch])
            if code != 0:
                logger.error(f"Failed to push merged changes: {stderr}")
                return False
                
            logger.info(f"Successfully merged {source_branch} into {target_branch}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging {source_branch} into {target_branch}: {e}")
            return False
            
    def delete_branch(self, branch_name: str, remote: bool = True) -> bool:
        """Delete branch locally and optionally remotely"""
        try:
            logger.info(f"Deleting branch: {branch_name}")
            
            # Switch to main branch first
            current_branch = self.get_current_branch()
            if current_branch == branch_name:
                self.checkout_branch(self.config["default_base_branch"])
                
            # Delete local branch
            code, stdout, stderr = self.run_command(["git", "branch", "-D", branch_name])
            if code != 0:
                logger.warning(f"Failed to delete local branch {branch_name}: {stderr}")
                
            # Delete remote branch
            if remote:
                code, stdout, stderr = self.run_command([
                    "git", "push", "origin", "--delete", branch_name
                ])
                if code != 0:
                    logger.warning(f"Failed to delete remote branch {branch_name}: {stderr}")
                    
            logger.info(f"Successfully deleted branch: {branch_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting branch {branch_name}: {e}")
            return False
            
    def create_pull_request(self, source_branch: str, target_branch: str, 
                           title: str, body: str = "") -> Optional[str]:
        """Create a pull request using GitHub CLI"""
        try:
            logger.info(f"Creating PR: {source_branch} -> {target_branch}")
            
            cmd = [
                "gh", "pr", "create",
                "--base", target_branch,
                "--head", source_branch,
                "--title", title,
                "--body", body or f"Automated PR for {source_branch}"
            ]
            
            code, stdout, stderr = self.run_command(cmd)
            
            if code != 0:
                logger.error(f"Failed to create PR: {stderr}")
                return None
                
            # Extract PR URL from output
            pr_url = stdout.strip()
            logger.info(f"Successfully created PR: {pr_url}")
            return pr_url
            
        except Exception as e:
            logger.error(f"Error creating PR: {e}")
            return None
            
    def auto_merge_pr(self, pr_number: str, strategy: MergeStrategy = MergeStrategy.SQUASH) -> bool:
        """Auto-merge a pull request"""
        try:
            logger.info(f"Auto-merging PR #{pr_number} with {strategy.value}")
            
            merge_method = {
                MergeStrategy.MERGE_COMMIT: "merge",
                MergeStrategy.SQUASH: "squash",
                MergeStrategy.REBASE: "rebase"
            }[strategy]
            
            cmd = ["gh", "pr", "merge", pr_number, f"--{merge_method}", "--delete-branch"]
            
            code, stdout, stderr = self.run_command(cmd)
            
            if code != 0:
                logger.error(f"Failed to auto-merge PR #{pr_number}: {stderr}")
                return False
                
            logger.info(f"Successfully auto-merged PR #{pr_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error auto-merging PR #{pr_number}: {e}")
            return False
            
    def generate_branch_name(self, branch_type: BranchType, 
                           description: str, issue_id: Optional[str] = None,
                           version: Optional[str] = None) -> str:
        """Generate branch name based on type and description"""
        pattern = self.config["branch_naming_patterns"][branch_type.value]
        
        # Clean description
        clean_desc = description.lower().replace(" ", "-").replace("_", "-")
        clean_desc = ''.join(c for c in clean_desc if c.isalnum() or c == '-')
        
        # Replace placeholders
        branch_name = pattern.format(
            description=clean_desc,
            issue_id=issue_id or "no-issue",
            version=version or "v1.0.0"
        )
        
        return branch_name
        
    def workflow_create_feature_branch(self, description: str, issue_id: Optional[str] = None) -> bool:
        """Complete workflow to create a feature branch"""
        branch_name = self.generate_branch_name(BranchType.FEATURE, description, issue_id)
        
        config = BranchConfig(
            name=branch_name,
            branch_type=BranchType.FEATURE,
            base_branch=self.config["default_base_branch"],
            merge_strategy=MergeStrategy(self.config["merge_strategies"]["feature"]),
            auto_merge=False,
            require_pr=True,
            delete_after_merge=True
        )
        
        return self.create_branch(config)
        
    def workflow_merge_and_cleanup(self, source_branch: str, 
                                  target_branch: str = None,
                                  create_pr: bool = True) -> bool:
        """Complete workflow to merge branch and cleanup"""
        if target_branch is None:
            target_branch = self.config["default_base_branch"]
            
        try:
            if create_pr:
                # Create PR first
                pr_url = self.create_pull_request(
                    source_branch, target_branch,
                    f"Merge {source_branch} into {target_branch}",
                    f"Automated merge request for {source_branch}"
                )
                if not pr_url:
                    return False
                    
                logger.info(f"PR created: {pr_url}")
                logger.info("Manual review and merge required via GitHub interface")
                return True
            else:
                # Direct merge
                if not self.merge_branch(source_branch, target_branch):
                    return False
                    
                # Cleanup
                if not self.delete_branch(source_branch):
                    logger.warning(f"Failed to cleanup branch {source_branch}")
                    
                return True
                
        except Exception as e:
            logger.error(f"Error in merge workflow: {e}")
            return False

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="DocFoundry Branching Strategy Manager")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create branch command
    create_parser = subparsers.add_parser("create", help="Create a new branch")
    create_parser.add_argument("type", choices=[t.value for t in BranchType], help="Branch type")
    create_parser.add_argument("description", help="Branch description")
    create_parser.add_argument("--issue-id", help="Associated issue ID")
    create_parser.add_argument("--version", help="Version for release/hotfix branches")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge and cleanup branch")
    merge_parser.add_argument("source", help="Source branch name")
    merge_parser.add_argument("--target", help="Target branch name (default: main)")
    merge_parser.add_argument("--no-pr", action="store_true", help="Skip PR creation")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync current branch")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show repository status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    strategy = BranchingStrategy()
    
    if args.command == "create":
        branch_type = BranchType(args.type)
        if branch_type == BranchType.FEATURE:
            success = strategy.workflow_create_feature_branch(args.description, args.issue_id)
        else:
            branch_name = strategy.generate_branch_name(
                branch_type, args.description, args.issue_id, args.version
            )
            config = BranchConfig(
                name=branch_name,
                branch_type=branch_type,
                merge_strategy=MergeStrategy(strategy.config["merge_strategies"][branch_type.value])
            )
            success = strategy.create_branch(config)
            
        if success:
            print(f"✅ Successfully created {branch_type.value} branch")
        else:
            print(f"❌ Failed to create {branch_type.value} branch")
            sys.exit(1)
            
    elif args.command == "merge":
        success = strategy.workflow_merge_and_cleanup(
            args.source, args.target, not args.no_pr
        )
        if success:
            print(f"✅ Successfully processed merge workflow")
        else:
            print(f"❌ Failed to process merge workflow")
            sys.exit(1)
            
    elif args.command == "sync":
        current_branch = strategy.get_current_branch()
        success = strategy.sync_branch(current_branch)
        if success:
            print(f"✅ Successfully synced branch {current_branch}")
        else:
            print(f"❌ Failed to sync branch {current_branch}")
            sys.exit(1)
            
    elif args.command == "status":
        current_branch = strategy.get_current_branch()
        print(f"Current branch: {current_branch}")
        print(f"Repository path: {strategy.repo_path}")
        print(f"Configuration: {strategy.config_file}")

if __name__ == "__main__":
    main()