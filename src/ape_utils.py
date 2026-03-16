import uuid
import pandas as pd
from tqdm import tqdm

from prompt_utils import sample_demo_examples, format_demo_block
from llm_utils import (
    generate_therapy_cue,
    generate_resampled_prompts,
    is_valid_prompt_instruction,
)
from scoring_utils import compute_proxy_score


def generate_outputs_for_prompt_df(prompt_df, eval_df, seed_df, ape_round=0):
    rows = []

    for _, p_row in tqdm(prompt_df.iterrows(), total=len(prompt_df), desc=f"APE round {ape_round} prompts"):
        prompt_id = p_row["prompt_id"]
        prompt_family = p_row["prompt_family"]
        prompt_text = p_row["prompt_text"]
        subcategory = p_row["subcategory"]
        parent_prompt_id = p_row.get("parent_prompt_id", "")

        if not is_valid_prompt_instruction(prompt_text):
            continue

        eval_subset = eval_df[eval_df["subcategory"] == subcategory].copy()
        if eval_subset.empty:
            continue

        for _, ex in eval_subset.iterrows():
            target_word = ex["target_word"]
            pos = ex["pos"]
            reference_cue = ex["cue_text"]

            demo_examples = sample_demo_examples(
                seed_df=seed_df,
                subcategory=subcategory,
                n_examples=3,
                random_state=42,
            )
            demo_block = format_demo_block(demo_examples)

            generated_cue = generate_therapy_cue(
                instruction=prompt_text,
                target_word=target_word,
                pos=pos,
                subcategory=subcategory,
                demo_block=demo_block,
            )

            score_dict = compute_proxy_score(
                generated_cue=generated_cue,
                target_word=target_word,
                reference_cue=reference_cue,
            )

            rows.append({
                "ape_round": ape_round,
                "parent_prompt_id": parent_prompt_id,
                "subcategory": subcategory,
                "prompt_id": prompt_id,
                "prompt_family": prompt_family,
                "prompt_text": prompt_text,
                "target_word": target_word,
                "pos": pos,
                "reference_cue": reference_cue,
                "generated_cue": generated_cue,
                **score_dict,
            })

    return pd.DataFrame(rows)


def summarize_prompt_scores(results_df):
    if results_df.empty:
        return pd.DataFrame(columns=[
            "ape_round", "parent_prompt_id", "subcategory", "prompt_id", "prompt_family",
            "prompt_text", "mean_proxy_score", "mean_similarity", "mean_brevity", "mean_target_leak"
        ])

    summary = (
        results_df.groupby(
            ["ape_round", "parent_prompt_id", "subcategory", "prompt_id", "prompt_family", "prompt_text"],
            as_index=False
        )
        .agg(
            mean_proxy_score=("proxy_score", "mean"),
            mean_similarity=("semantic_similarity", "mean"),
            mean_brevity=("brevity_score", "mean"),
            mean_target_leak=("target_leak_penalty", "mean"),
        )
        .sort_values(["subcategory", "ape_round", "mean_proxy_score"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    return summary


def generate_resampled_prompt_df(top_prompt_summary_df, n_new_prompts=3, ape_round=1):
    rows = []

    for subcategory, group in top_prompt_summary_df.groupby("subcategory"):
        top_prompt_rows = group.to_dict("records")
        new_prompt_texts = generate_resampled_prompts(
            top_prompt_rows=top_prompt_rows,
            n_new_prompts=n_new_prompts,
        )

        seen = set()
        for p_text in new_prompt_texts:
            if not p_text:
                continue
            if not is_valid_prompt_instruction(p_text):
                continue
            if p_text in seen:
                continue
            seen.add(p_text)

            rows.append({
                "subcategory": subcategory,
                "prompt_id": f"{subcategory}_ape_{ape_round}_{uuid.uuid4().hex[:8]}",
                "prompt_family": "category_resampled",
                "prompt_text": p_text,
                "parent_prompt_id": top_prompt_rows[0]["prompt_id"] if top_prompt_rows else "",
            })

    return pd.DataFrame(rows)