@'
# Aphasia APE Cue Generator

A category-wise Automatic Prompt Engineering (APE) pipeline for generating therapy cue sentences for older adults with aphasia.

## Project goal

This project helps older adults with aphasia recall target words through short, concrete, and category-appropriate cue sentences.

The system uses:
- therapist-scored cue datasets
- category-specific expert prompts
- local LLM generation
- category-wise prompt optimization
- final best-prompt-based cue generation

## Pipeline

1. Clean dataset
2. Select best human therapist cues
3. Build seed examples
4. Build category-wise prompt candidates
5. Generate initial cues with local LLM
6. Score prompt outputs
7. Run category-wise APE optimization
8. Select best prompt per category
9. Generate final cues using best prompt

## Main folders

- `src/` -> core Python files
- `notebooks/` -> step-by-step Jupyter notebooks
- `data/raw/` -> raw dataset
- `data/processed/` -> processed therapist-best cues and seed examples

## Main files

- `src/prompt_utils.py`
- `src/llm_utils.py`
- `src/scoring_utils.py`
- `src/ape_utils.py`
- `src/config.py`

## Outputs

Main outputs include:
- best prompt by category
- generated cues by category
- therapist cue vs generated cue comparison

## Notes

- Local Qwen model is used for offline generation.
- Large model files are not included in the repository.
- Generated cues may still require therapist review before practical use.

## Author

Ujjwal Sahu
'@ | Set-Content README.md
