# FORSAtic Hackathon - Algérie Télécom

## Project Overview

This repository contains our solution for the **FORSAtic Hackathon** (December 11-13, 2025) focused on improving customer satisfaction through AI and complaint analysis.

## Challenges

### Challenge 1: Social Media Comments Classification
- **Classes**: 9
- **Data**: ~30,000 comments from Facebook, Twitter, etc.
- **Format**: CSV with columns `comment`, `platform`, `class`

### Challenge 2: Call Center Tickets Classification
- **Classes**: 6
- **Data**: Structured call center tickets
- **Format**: CSV with column `class_int`

## Evaluation Metric
**Macro F1-Score** (average of both challenges)

## Repository Structure

```
forca-hack/
├── notebooks/          # Jupyter notebooks (run in Google Colab)
├── src/                # Python source code
├── submissions/        # Kaggle submission files
└── data/              # Datasets (local only, not in git)
```

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in Google Colab (see notebooks/ folder)

## Workflow

- **Code editing**: Use Cursor/VS Code locally
- **Training/Experiments**: Run in Google Colab
- **Data**: Store in Google Drive, mount in Colab

