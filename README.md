# IATA Coursework Group 2 — Task 2: Structured Information Extraction

## Overview

The project investigates how different NLP and AI approaches can extract structured clinical trial information from unstructured biomedical abstracts using the **EBM-NLP** dataset. The target schema follows the standard **PICO** framework:
- **P** — Population
- **I** — Intervention
- **C** — Comparator
- **O** — Outcome

The coursework explores multiple extraction aproaches which include clustering, rule-based systems, prompting strategies, transformer fine-tuning, decomposed QA pipelines and end-to-end extraction systems.

# Contributors: 
Weixuan Gu （2816084）

Zongjin Han (2764620)

Disen Zhu (2560315)

Chris Luckhurst

Sanya Kapoor (2696783)

Saisha Hiray


# Repository Structure

```text
IATA-Coursework-Group2/
│
├── Final Report 
├── dataset/                     # Dataset files and preprocessing resources
├── notebook/                    # Experimental notebooks
├── Report_ACL_Style/            # Final ACL-style report files
├── Report_Pending_merger/       # Draft report sections
├── README.md
```
# Dataset
The experiment uses the EBM-NLP dataset, which contains annotated clinical trial abstracts with token-level PICO labels and focuses on:
Biomedical abstracts, Clinical trial evidence and structured evidence extraction
Evaluation Metrics

# Evaluation Metrics
Different notebooks evaluate systems using combinations of:
Precision, Recall, F1-score, Coverage, Field-level and extraction accuracy.
Some notebooks also include qualitative error analysis and extraction examples.






