# KG Schema Evolution Agents

This repository contains code, experiments, and documentation for knowledge graph schema evolution using PyKEEN and agentic reasoning approaches.

## Repository structure

- src/ - core project code and experiment scripts
  - Agentic_Memory/ - agentic memory and routing utilities
  - Nations_minimal_Run/ - PyKEEN/Nations embedding experiment scripts
- data_preprocessing/ - dataset preprocessing scripts and utilities
- esults/ - consolidated experiment outputs and checkpoint JSON files
- enchmarks/ - benchmark experiment data for writeback and no-writeback runs
- docs/ - architecture diagrams and project notes
- migration/ - migration notes and results
- papers/ - research papers and references
- equirements.txt - Python dependencies
- README.md - project overview and usage

## Getting started

1. Create a virtual environment:
   `powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   `
2. Install dependencies:
   `powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   `
3. Explore code in src/ and run experiment scripts from there.

## Notes

- All original files are preserved and reorganized into a cleaner structure.
- esults/ now contains JSON and TSV outputs previously stored at the repository root.
- enchmarks/ groups benchmark data under writeback and no-writeback scenarios.
- docs/ contains supplemental diagrams and design notes.
