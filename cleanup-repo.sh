#!/bin/bash

# Create a backup branch
git checkout -b backup_before_cleanup

# Move back to main branch
git checkout main

# Remove the large files from Git history
git filter-branch --force --index-filter \
  "git rm -r --cached --ignore-unmatch elizadb/ models/" \
  --prune-empty --tag-name-filter cat -- --all

# Clean up the repository
git for-each-ref --format="delete %(refname)" refs/original/ | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Repository cleaned. Check the size with 'du -sh .git'"
echo "To push these changes to origin use: git push origin --force --all" 