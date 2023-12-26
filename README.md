# SEC-File-Analysis-with-LLM

## Introduction

This Python-based project analyzes SEC files (10-K, 10-Q) using Large Language Models (LLMs). It's designed to extract and process data from these documents, leveraging advanced Language Chain Generators, Data Extraction using prompt templates, and Natural Language Processing techniques. Modify the "ques_to_ask_[file_type].csv" to extract these answers from the SEC file.

## Features

- **PDF Processing:** Converts PDFs to text for analysis.
- **Chain Generation:** Creates a query chain for the LLM.
- **Data Extraction:** Extracts specific data points from SEC files.
- **File Management:** Handles file operations including reading, saving, and uploading.
- **Configurable for Various SEC Filings:** Suitable for documents like 10-K and 10-Q.

## Requirements

- Python 3.9
- Required Libraries: see requirements.txt

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/h-xiao/SEC-File-Analysis-with-LLM.git

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
 
## Usage

1. Place SEC files in the `pdfs/[ticker]/[file_type]/` directory, where `[ticker]` is the stock ticker symbol and `[file_type]` is the type of SEC filing (e.g., 10-K, 10-Q).

2. Configure the config.ini file with your settings, like dropbox token.

3. Execute the main script:
   ```bash
   python sec_file_analysis_with_llm.py

## Structure

- **PDFProcessor:** Loads and splits PDF documents.
- **ChainGenerator:** Generates a chain for querying LLMs.
- **DataExtractor:**  Extracts and formats the required data.
- **FileManager:** Manages file operations.
- **MainProcess:** Orchestrates the extraction and analysis process.

## Configuration

- **config.ini:** Configurable settings like paths and tokens.
- **ques_to_ask_[file_type].csv:** Questions for data extraction.



