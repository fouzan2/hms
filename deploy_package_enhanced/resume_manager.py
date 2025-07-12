#!/usr/bin/env python3
"""
HMS Training Resume Manager
Manages training state and resume functionality
"""

import os
import sys
import json
import pickle
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class ResumeManager:
    def __init__(self):
        self.workspace = Path("/workspace")
        self.state_dir = self.workspace / "training_state"
        self.backup_dir = self.workspace / "backups"
        self.checkpoint_dir = self.workspace / "models" / "checkpoints"
        
        # Ensure directories exist
        for dir_path in [self.state_dir, self.backup_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, name: str = None) -> str:
        """Create a backup of current training state"""
        if name is None:
            name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / name
        backup_path.mkdir(exist_ok=True)
        
        # Backup training state
        if self.state_dir.exists():
            shutil.copytree(self.state_dir, backup_path / "training_state", dirs_exist_ok=True)
        
        # Backup checkpoints
        if self.checkpoint_dir.exists():
            shutil.copytree(self.checkpoint_dir, backup_path / "checkpoints", dirs_exist_ok=True)
        
        # Create backup info
        info = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "files_backed_up": len(list(backup_path.rglob("*"))),
            "size_mb": sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file()) / (1024*1024)
        }
        
        with open(backup_path / "backup_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✅ Backup created: {backup_path}")
        print(f"📊 Files: {info['files_backed_up']}, Size: {info['size_mb']:.1f}MB")
        
        return str(backup_path)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                info_file = backup_dir / "backup_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    backups.append(info)
                else:
                    # Create basic info for old backups
                    backups.append({
                        "name": backup_dir.name,
                        "timestamp": datetime.fromtimestamp(backup_dir.stat().st_mtime).isoformat(),
                        "files_backed_up": len(list(backup_dir.rglob("*"))),
                        "size_mb": sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file()) / (1024*1024)
                    })
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore from a backup"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            print(f"❌ Backup not found: {backup_name}")
            return False
        
        try:
            # Restore training state
            state_backup = backup_path / "training_state"
            if state_backup.exists():
                if self.state_dir.exists():
                    shutil.rmtree(self.state_dir)
                shutil.copytree(state_backup, self.state_dir)
            
            # Restore checkpoints
            checkpoint_backup = backup_path / "checkpoints"
            if checkpoint_backup.exists():
                if self.checkpoint_dir.exists():
                    shutil.rmtree(self.checkpoint_dir)
                shutil.copytree(checkpoint_backup, self.checkpoint_dir)
            
            print(f"✅ Backup restored: {backup_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restore backup: {e}")
            return False
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current training status"""
        state_file = self.state_dir / "training_state.pkl"
        
        if not state_file.exists():
            return {"status": "No training state found"}
        
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            return {
                "current_stage": state.current_stage,
                "stages_completed": state.stages_completed,
                "models_trained": state.models_trained,
                "best_accuracy": state.best_accuracy,
                "can_resume": True
            }
        except Exception as e:
            return {"status": f"Error reading state: {e}"}
    
    def clean_old_backups(self, keep_count: int = 10):
        """Clean old backups, keeping only the most recent ones"""
        backups = self.list_backups()
        
        if len(backups) > keep_count:
            to_remove = backups[keep_count:]
            
            for backup in to_remove:
                backup_path = self.backup_dir / backup['name']
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                    print(f"🗑️  Removed old backup: {backup['name']}")
            
            print(f"✅ Cleaned {len(to_remove)} old backups, kept {keep_count} most recent")

def main():
    parser = argparse.ArgumentParser(description='HMS Training Resume Manager')
    parser.add_argument('--backup', metavar='NAME', help='Create backup with optional name')
    parser.add_argument('--list', action='store_true', help='List all backups')
    parser.add_argument('--restore', metavar='NAME', help='Restore from backup')
    parser.add_argument('--status', action='store_true', help='Show current training status')
    parser.add_argument('--clean', action='store_true', help='Clean old backups')
    parser.add_argument('--auto-backup', action='store_true', help='Create automatic backup')
    
    args = parser.parse_args()
    manager = ResumeManager()
    
    if args.backup is not None:
        manager.create_backup(args.backup if args.backup else None)
    elif args.list:
        backups = manager.list_backups()
        print("\n📋 Available Backups:")
        print("=" * 80)
        for backup in backups:
            print(f"Name: {backup['name']}")
            print(f"Date: {backup['timestamp']}")
            print(f"Size: {backup['size_mb']:.1f}MB")
            print("-" * 40)
    elif args.restore:
        manager.restore_backup(args.restore)
    elif args.status:
        status = manager.get_current_status()
        print("\n📊 Current Training Status:")
        print(json.dumps(status, indent=2))
    elif args.clean:
        manager.clean_old_backups()
    elif args.auto_backup:
        manager.create_backup()
        manager.clean_old_backups()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
