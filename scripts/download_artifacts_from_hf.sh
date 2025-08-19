#!/bin/bash

# Download artifacts from Hugging Face Hub
# Usage: ./scripts/download_artifacts_from_hf.sh [target_directory]

set -e

# Configuration
HF_REPO="canrager/dyn_rep"
SUBFOLDER="acts"
DEFAULT_TARGET_DIR="artifacts/interim"

# Get target directory from command line or use default
TARGET_DIR="${1:-$DEFAULT_TARGET_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}📥 Starting download of artifacts from Hugging Face Hub${NC}"
echo -e "${YELLOW}Repository: ${HF_REPO}${NC}"
echo -e "${YELLOW}Subfolder: ${SUBFOLDER}${NC}"
echo -e "${YELLOW}Target directory: ${TARGET_DIR}${NC}"

# Check if huggingface_hub is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}❌ Error: huggingface-cli not found. Install with: pip install huggingface_hub[cli]${NC}"
    exit 1
fi

# Backup existing target directory if it exists and has content
if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR")" ]; then
    BACKUP_DIR="${TARGET_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}🔄 Backing up existing directory to ${BACKUP_DIR}${NC}"
    mv "$TARGET_DIR" "$BACKUP_DIR"
fi

# Create target directory
echo -e "${YELLOW}📁 Creating target directory...${NC}"
mkdir -p "$TARGET_DIR"

# Download from HF Hub with retry logic
download_with_retry() {
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo -e "${YELLOW}📤 Download attempt $((retry_count + 1))/${max_retries}${NC}"
        
        if huggingface-cli download "$HF_REPO" \
            --include "${SUBFOLDER}/*" \
            --local-dir "$TARGET_DIR" \
            --local-dir-use-symlinks False; then
            echo -e "${GREEN}✅ Download successful!${NC}"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo -e "${YELLOW}⚠️  Download failed, retrying in 5 seconds...${NC}"
                sleep 5
            fi
        fi
    done
    
    echo -e "${RED}❌ Download failed after ${max_retries} attempts${NC}"
    return 1
}

# Perform the download
echo -e "${YELLOW}🌐 Downloading from Hugging Face Hub...${NC}"
if download_with_retry; then
    echo -e "${GREEN}✅ Download completed successfully!${NC}"
else
    echo -e "${RED}💥 Download failed${NC}"
    exit 1
fi

# Move files from subfolder to target directory root
DOWNLOADED_SUBFOLDER="${TARGET_DIR}/${SUBFOLDER}"
if [ -d "$DOWNLOADED_SUBFOLDER" ]; then
    echo -e "${YELLOW}📁 Moving files from subfolder to target directory...${NC}"
    mv "$DOWNLOADED_SUBFOLDER"/* "$TARGET_DIR"/ 2>/dev/null || true
    rmdir "$DOWNLOADED_SUBFOLDER" 2>/dev/null || true
fi

# Verify download
if [ ! -d "$TARGET_DIR" ] || [ -z "$(ls -A "$TARGET_DIR")" ]; then
    echo -e "${RED}❌ Error: Download failed or target directory is empty${NC}"
    exit 1
fi

# Count extracted files
FILE_COUNT=$(find "$TARGET_DIR" -type f | wc -l)
echo -e "${GREEN}✅ Extracted ${FILE_COUNT} files to ${TARGET_DIR}${NC}"

# Show directory size
DIR_SIZE=$(du -sh "$TARGET_DIR" | cut -f1)
echo -e "${BLUE}📊 Directory size: ${DIR_SIZE}${NC}"

echo -e "${GREEN}🎉 Download completed successfully!${NC}"
echo -e "${GREEN}📁 Artifacts available at: ${TARGET_DIR}${NC}"

# Show a sample of what was downloaded
echo -e "${BLUE}📋 Sample of downloaded files:${NC}"
find "$TARGET_DIR" -type f | head -5 | while read -r file; do
    echo -e "${BLUE}  - $(basename "$file")${NC}"
done

if [ $(find "$TARGET_DIR" -type f | wc -l) -gt 5 ]; then
    echo -e "${BLUE}  ... and $((FILE_COUNT - 5)) more files${NC}"
fi