#!/bin/bash

# Configuration
REPO_URL="https://github.com/Ridreb05/project-amaze.git"
SUBDIR="site"
TEMP_DIR="temp_sync_repo"

echo "🚀 Starting push to $REPO_URL in directory /$SUBDIR..."

# 1. Clean up any previous temp dirs
rm -rf $TEMP_DIR

# 2. Clone the remote repository
git clone $REPO_URL $TEMP_DIR

if [ $? -ne 0 ]; then
    echo "❌ Failed to clone repository."
    exit 1
fi

# 3. Ensure the subdirectory exists
mkdir -p $TEMP_DIR/$SUBDIR

# 4. Sync files from current directory to the repo subdirectory
# Excluding git, node_modules, and the temp dir itself
rsync -av --exclude='.git' --exclude='node_modules' --exclude='$TEMP_DIR' --exclude='temp_repo*' --exclude='.DS_Store' ./ $TEMP_DIR/$SUBDIR/

# 5. Commit and push
cd $TEMP_DIR
git config user.email "ridreb05@gmail.com"
git config user.name "Ridreb05"
git add .
git commit -m "Update site from local hackfrnt"
git push origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully pushed to $REPO_URL/tree/main/$SUBDIR"
else
    echo "❌ Push failed."
fi

# 6. Clean up
cd ..
rm -rf $TEMP_DIR
