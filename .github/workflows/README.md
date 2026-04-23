# GitHub Workflows

This directory contains GitHub Actions workflows for automated testing and continuous integration.

## Workflows

### e2e_gcn.yml

**Purpose**: End-to-end testing of GCN model on Cora dataset

**Trigger**: Runs on push to main branch or pull requests targeting main branch

**Environment**: Ubuntu 22.04 LTS (CPU-only)

**Steps**:
1. Checkout code
2. Set up Python 3.10
3. Install system dependencies (build-essential, python3-dev)
4. Install Python dependencies from requirements_ci.txt
5. Run GCN example on Cora dataset with JITTOR_USE_CUDA=0

## Requirements Files

### requirements_ci.txt

CPU-only requirements file for GitHub Actions environment.

**Key differences from main requirements.txt**:
- Removed CUDA-dependent packages (cupy)
- All other dependencies remain the same
- Ensures compatibility with GitHub's CPU-only runners

## Usage

Workflows run automatically on push/PR events. To manually trigger:

1. Go to GitHub repository
2. Click "Actions" tab
3. Select the workflow
4. Click "Run workflow"

## Notes

- All workflows run in CPU-only mode
- JITTOR_USE_CUDA=0 is enforced for all tests
- Results are visible in GitHub Actions interface