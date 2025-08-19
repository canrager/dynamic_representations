#!/bin/bash

# Upload artifacts to Hugging Face Hub
# Usage: ./scripts/upload_artifacts_to_hf.sh

set -e

# Configuration
HF_REPO="canrager/dyn_rep"
ARTIFACTS_DIR="artifacts/interim"
SUBFOLDER="acts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting upload of artifacts to Hugging Face Hub${NC}"
echo -e "${YELLOW}Repository: ${HF_REPO}${NC}"
echo -e "${YELLOW}Subfolder: ${SUBFOLDER}${NC}"

# Check if artifacts directory exists
if [ ! -d "$ARTIFACTS_DIR" ]; then
    echo -e "${RED}❌ Error: Artifacts directory '$ARTIFACTS_DIR' not found${NC}"
    exit 1
fi

# Check if huggingface_hub is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}❌ Error: huggingface-cli not found. Install with: pip install huggingface_hub[cli]${NC}"
    exit 1
fi

# Check if logged in to HF
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${RED}❌ Error: Not logged in to Hugging Face. Run: huggingface-cli login${NC}"
    exit 1
fi

# Get artifacts directory size for progress tracking
ARTIFACTS_SIZE=$(du -sh "$ARTIFACTS_DIR" | cut -f1)
FILE_COUNT=$(find "$ARTIFACTS_DIR" -type f | wc -l)
echo -e "${GREEN}📊 Artifacts to upload: ${FILE_COUNT} files (${ARTIFACTS_SIZE})${NC}"

# Upload to HF Hub with retry logic
echo -e "${YELLOW}🌐 Uploading to Hugging Face Hub...${NC}"

upload_with_retry() {
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo -e "${YELLOW}📤 Upload attempt $((retry_count + 1))/${max_retries}${NC}"
        
        if huggingface-cli upload "$HF_REPO" "$ARTIFACTS_DIR" "$SUBFOLDER" \
            --repo-type model \
            --commit-message "Upload artifacts/interim folder structure"; then
            echo -e "${GREEN}✅ Upload successful!${NC}"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo -e "${YELLOW}⚠️  Upload failed, retrying in 5 seconds...${NC}"
                sleep 5
            fi
        fi
    done
    
    echo -e "${RED}❌ Upload failed after ${max_retries} attempts${NC}"
    return 1
}

# Try to create the repository first (will fail silently if it exists)
echo -e "${YELLOW}📋 Ensuring repository exists...${NC}"
huggingface-cli repo create "$HF_REPO" --type model || true

# Perform the upload
if upload_with_retry; then
    echo -e "${GREEN}🎉 Successfully uploaded artifacts to HF Hub!${NC}"
    echo -e "${GREEN}🔗 Repository: https://huggingface.co/${HF_REPO}${NC}"
    echo -e "${GREEN}📁 Folder location: ${SUBFOLDER}/${NC}"
else
    echo -e "${RED}💥 Upload failed${NC}"
    exit 1
fi

echo -e "${GREEN}✨ Upload process completed successfully!${NC}"