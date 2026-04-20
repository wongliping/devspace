# HF Model Downloader

Download Hugging Face model files with glob-based include/exclude filtering and optional authenticated proxy support. Includes a Jenkins pipeline for automated download and upload to JFrog Artifactory.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python download_model.py <repo_id> [options]
```

### Examples

```bash
# Download all files
python download_model.py moonshotai/Kimi-K2.5 -o ./kimi-out

# Only safetensors + config files
python download_model.py moonshotai/Kimi-K2.5 --include "*.safetensors" "config.json" "tokenizer*"

# Exclude pytorch bins
python download_model.py moonshotai/Kimi-K2.5 --exclude "*.bin" "original/*"

# Via authenticated proxy (CLI)
python download_model.py moonshotai/Kimi-K2.5 \
  --proxy http://proxy.corp.com:8080 \
  --proxy-user myuser \
  --proxy-password mypass

# Via authenticated proxy (env vars)
export HTTPS_PROXY=http://proxy.corp.com:8080
export PROXY_USER=myuser
export PROXY_PASSWORD=mypass
python download_model.py moonshotai/Kimi-K2.5
```

## Options

| Option | Default | Description |
|---|---|---|
| `repo_id` | — | Hugging Face repo ID, e.g. `moonshotai/Kimi-K2.5` |
| `-o`, `--output-dir` | `./output` | Local output root; model files go to `OUTPUT_DIR/model`, manifests to `OUTPUT_DIR/results` |
| `--include PATTERN [...]` | — | Only download files matching these glob patterns |
| `--exclude PATTERN [...]` | — | Skip files matching these glob patterns |
| `--proxy URL` | `$HTTPS_PROXY` | Proxy URL, e.g. `http://proxy.corp.com:8080` |
| `--proxy-user USER` | `$PROXY_USER` | Proxy username |
| `--proxy-password PASS` | `$PROXY_PASSWORD` | Proxy password |

## Output

Files are written under the output root:

```
output/
  model/          ← downloaded model files
  results/
    downloaded.txt      ← paths of successfully downloaded files
    failed.txt          ← paths of files that failed to download
    uploaded.txt        ← JFrog URLs of successfully uploaded files
    upload_failed.txt   ← paths + HTTP status of files that failed to upload
```

Exit code is non-zero if any file failed to download.

---

## Jenkins Pipeline

The `Jenkinsfile` automates the full download-and-upload workflow.

### Stages

| Stage | Description |
|---|---|
| Validate | Ensures `REPO_ID` is provided |
| Setup | Installs Python dependencies |
| Download | Downloads model files from Hugging Face |
| Upload to JFrog | Uploads downloaded files to Artifactory (skipped if JFrog params are not set) |
| Archive | Archives all result manifests as Jenkins build artifacts |

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `REPO_ID` | — | Hugging Face repo ID, e.g. `moonshotai/Kimi-K2.5` |
| `OUTPUT_DIR` | `./out` | Local output root |
| `INCLUDE` | — | Space-separated glob patterns to include |
| `EXCLUDE` | — | Space-separated glob patterns to exclude |
| `PROXY` | — | Proxy URL, e.g. `http://proxy.corp.com:8080` |
| `JFROG_URL` | — | Artifactory base URL, e.g. `https://artifactory.corp.com/artifactory` |
| `JFROG_REPO` | — | Target Artifactory repository, e.g. `ml-models-local` |
| `JFROG_TARGET_PATH` | — | Path prefix within the repo, e.g. `huggingface/moonshotai/Kimi-K2.5` |

### Required Jenkins Credentials

| Credential ID | Type | Used for |
|---|---|---|
| `hf-token` | Secret text | Hugging Face API token |
| `proxy-user` | Secret text | Proxy username |
| `proxy-password` | Secret text | Proxy password |
| `jfrog-credentials` | Username/Password | Artifactory username and API key |

### Pipeline Report

At the end of every run (pass or fail), the pipeline prints a summary to the build log:

```
============================================================
PIPELINE REPORT — moonshotai/Kimi-K2.5
============================================================
DOWNLOAD
  Succeeded : 12
  Failed    : 1

Downloaded files:
  ./out/model/config.json
  ...

Failed downloads:
  some-file.bin
============================================================
UPLOAD TO JFROG
  Succeeded : 12
  Failed    : 0

Uploaded files:
  https://artifactory.corp.com/artifactory/ml-models-local/config.json
  ...
============================================================
```

If `JFROG_URL` or `JFROG_REPO` are not set, the upload section shows `Skipped`. All four result files are also archived as Jenkins build artifacts.
