# Aphasia APE Cue Generator

A category-wise Automatic Prompt Engineering (APE) pipeline for aphasia therapy cue generation, evaluation, local image generation, and mobile deployment.

## Overview

This project builds a full workflow to generate therapy cues for aphasia-related language support tasks.

The pipeline:
1. prepares and cleans therapist-scored data
2. selects strong seed examples
3. builds prompt candidates
4. applies category-wise APE
5. selects the best prompt for each category
6. generates final cues with a local LLM
7. prepares image prompts
8. generates images locally
9. exports mobile-ready cue and image assets

## Main Features

- Category-wise prompt optimization
- Local LLM-based cue generation
- Proxy scoring with similarity, brevity, and leakage penalty
- Systematic evaluation reports
- Local SDXL-based image generation
- Mobile-ready export for Android deployment


----------------------------------------------------------------------------------------------------------------------------------------------------------------

## Folder Description
 - data/

Stores raw, processed, interim, and final project data.

 - models/

Stores local models such as:

Qwen cue-generation model

SDXL image-generation model

These are kept local and are not uploaded to GitHub because of size limits.

 - notebooks/

Contains the step-by-step experimental and production pipeline notebooks.

 - results/

Stores generated images, test outputs, and result artifacts.

 - src/

Contains reusable utility modules used by the notebooks.

## Notebook Pipeline
- Data preparation

01_clean_dataset.ipynb

02_prepare_gold_scoring_tables.ipynb

03_build_seed_examples.ipynb

- Prompt generation and APE

04_build_initial_prompt_candidates.ipynb

05_generate_initial_outputs.ipynb

06_ape_round0_scoring.ipynb

07_category_ape_iterative_optimization.ipynb

08_select_final_best_prompt.ipynb

- Final cue generation and evaluation

09_generate_final_cues_with_best_prompt.ipynb

10_systematic_evaluation_report.ipynb

- Image generation and export

11_prepare_final_cue_and_image_prompt_table.ipynb

12_generate_images_for_final_words.ipynb

13_export_final_demo_package.ipynb

14_prepare_mobile_assets.ipynb

- Local image model testing

Test_local_SDXL_model.ipynb

## Source Files

Main Python modules in src/:

  - config.py

  - data_utils.py

  - prompt_utils.py

  - llm_utils.py

  - scoring_utils.py

  - ape_utils.py

## APE Workflow

This project uses category-wise APE instead of a single global prompt.

- Workflow summary

1. build therapist-best cue examples

2. group words by category

3. generate several prompt candidates for each category

4. use the LLM to generate cues

5. score generated cues using:

       - semantic similarity

       - brevity score

       - target leakage penalty

6. summarize prompt performance

7. select the best prompt per category

8. generate final cues

9. perform systematic evaluation

10. generate image prompts and images

11. export mobile-ready assets

## Example Categories

Examples of categories used in the dataset include:

水果

動物

動作

交通工具

生活用品

電器用品

食物

文具

家具

輔具

娛樂場所

休閒娛樂

## Models Used
- Cue generation model

Local Qwen model, for example:

models/Qwen2.5-1.5B-Instruct/

- Image generation model

Local SDXL model, for example:

models/sdxl-base-1.0/

## Local Setup

Create and activate your environment:

conda activate aphasia_ape

Install the required packages you use in the project, such as:

pandas

numpy

torch

transformers

sentence-transformers

scikit-learn

diffusers

accelerate

safetensors

pillow

tqdm

## Recommended Run Order

Run the notebooks in this order:

01_clean_dataset.ipynb

02_prepare_gold_scoring_tables.ipynb

03_build_seed_examples.ipynb

04_build_initial_prompt_candidates.ipynb

05_generate_initial_outputs.ipynb

06_ape_round0_scoring.ipynb

07_category_ape_iterative_optimization.ipynb

08_select_final_best_prompt.ipynb

09_generate_final_cues_with_best_prompt.ipynb

10_systematic_evaluation_report.ipynb

11_prepare_final_cue_and_image_prompt_table.ipynb

12_generate_images_for_final_words.ipynb

13_export_final_demo_package.ipynb

14_prepare_mobile_assets.ipynb

## Final Exports

Important outputs are written under:

data/final/

Typical exported files include:

  - best_prompt_by_category.csv

  - final_generated_cues_by_category.csv

  - mobile_app_export.csv

  - mobile_app_export.json

  - final_demo_package.csv

  - professor_review_package.csv

## Mobile Deployment

This project supports mobile deployment by exporting:

  - target word

  - category

  - final generated cue

  - image path

The mobile app should only load the final exported JSON and images.

## The phone app should not run:

  - APE

  - LLM inference

  - SDXL inference

These heavy steps are done locally on PC before deployment.

## Android App Integration

For Android deployment, use:

  - JSON in assets/data/mobile_app_export.json

  - images in assets/images/...

Typical Android-side files:

  - CueItem.kt

  - JsonUtils.kt

  - AssetImageUtils.kt

  - MainActivity.kt

  - activity_main.xml

## Notes on Large Files

Large models and generated image outputs are excluded from GitHub because of repository size limits.

The following are intentionally kept local:

  - models/

  - generated raw images

  - generated mobile images

  - large binary weights

--------------------------------------------------------------------------------------------------------------------------------------------------------------

## Project Structure

```text
Aphasia_APE_Cue_Generator/
├── data/
│   ├── final/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── results/
├── src/
├── .gitignore
└── README.md


Author

Ujjwal Sahu
