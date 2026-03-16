from pathlib import Path
from typing import Optional, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# LOCAL QWEN MODEL PATH
# ============================================================
LOCAL_MODEL_PATH = Path(
    r"C:\Users\804\Dtx_Projects\Aphasia_APE_Cue_Generator\models\Qwen2.5-1.5B-Instruct"
)

print(f"Using local model path: {LOCAL_MODEL_PATH}")

# ============================================================
# LOAD TOKENIZER
# ============================================================
try:
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )
except Exception as e:
    print("Fast tokenizer load failed. Retrying with use_fast=False ...")
    print("Original error:", e)
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# LOAD MODEL
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
).eval()


# ============================================================
# CATEGORY RULES
# ============================================================
def get_category_guidance(subcategory: str, pos: str = "") -> str:
    sub = str(subcategory).strip()
    pos = str(pos).strip()

    if "動詞" in pos or sub in ["動作", "休閒娛樂", "日常活動", "日常行為"]:
        return (
            "若為動作類，請優先描述典型情境、動作目的、常見使用時機。"
            "不要寫成叫人去做的命令句。"
        )

    if sub == "水果":
        return "若為水果，請優先描述顏色、形狀、口感、味道、是否成串。"

    if sub == "蔬菜":
        return "若為蔬菜，請優先描述顏色、形狀、部位、常見料理用途。"

    if sub in ["食物", "飲料", "餐具", "食材"]:
        return "若為食物相關類，請優先描述外觀、味道、口感、用途或常見吃法。"

    if sub in ["生活用品", "文具", "輔具", "電器用品", "家電", "日常用品"]:
        return "若為用品或電器類，請優先描述用途、功能、外觀或使用情境。"

    if sub in ["娛樂場所", "地點", "場所", "公共場所"]:
        return "若為地點或場所，請優先描述人們在那裡常做什麼、會看到什麼或該地功能。"

    if sub == "交通場所":
        return "若為交通場所，請優先描述報到、候車、買票、安檢、等候、領行李等熟悉交通情境。"

    if sub in ["交通工具", "運輸工具"]:
        return "若為交通工具，請優先描述載人用途、搭乘方式、常見情境或固定路線。"

    if sub in ["動物", "自然環境", "昆蟲"]:
        return "若為動物或自然類，請優先描述外觀、叫聲、生活環境、移動方式或常見特徵。"

    if "名詞" in pos:
        return "若為名詞類，請優先描述外觀、用途、類別、位置或常見屬性。"

    return "請選擇最容易讓人聯想到答案的具體特徵，優先使用外觀、用途、類別、情境等線索。"


# ============================================================
# PROMPT BUILDERS
# ============================================================
def build_generation_prompt(
    instruction: str,
    target_word: str,
    pos: str = "",
    subcategory: str = "",
    demo_block: Optional[str] = None,
) -> str:
    category_guidance = get_category_guidance(subcategory=subcategory, pos=pos)

    parts = []
    parts.append("你要產生的是「失語症語言治療提示句」。")
    parts.append("目標是幫助高齡失語症患者聯想到答案，但不能直接說出答案。")
    parts.append("")
    parts.append("本次任務完整提示：")
    parts.append(instruction.strip())
    parts.append("")

    if demo_block:
        parts.append("以下是合格示例：")
        parts.append(demo_block.strip())
        parts.append("")

    parts.append("請再遵守以下生成規則：")
    parts.append("1. 只能輸出一行")
    parts.append("2. 只能輸出提示句本身")
    parts.append("3. 不可直接包含目標詞")
    parts.append("4. 不可寫成命令句、對話句、故事句、宣傳句")
    parts.append("5. 句子要短、具體、自然、容易理解")
    parts.append("6. 優先選最有辨識度的特徵")
    parts.append("7. 優先用高齡者熟悉的生活語言")
    parts.append("")

    parts.append("類別補充提示：")
    parts.append(category_guidance)
    parts.append("")

    parts.append("任務資訊：")
    if pos:
        parts.append(f"詞性：{pos}")
    if subcategory:
        parts.append(f"細項：{subcategory}")
    parts.append(f"目標詞彙：{target_word}")
    parts.append("")

    parts.append("請直接輸出提示句，不要解釋：")

    return "\n".join(parts)


def build_resampling_prompt(
    top_prompt_rows: List[Dict],
    n_new_prompts: int = 5,
) -> str:
    lines = []
    for i, row in enumerate(top_prompt_rows, start=1):
        if "prompt_text" in row:
            lines.append(f"{i}. {row['prompt_text']}")
        elif "instruction" in row:
            lines.append(f"{i}. {row['instruction']}")

    top_block = "\n".join(lines)

    return f"""
你是一位 Automatic Prompt Engineering 專家。

以下是目前表現最好的提示指令：
{top_block}

這些提示指令的任務固定為：
「為失語症患者生成簡短、具體、容易聯想到答案的語意提示句」。

請根據高分提示的共同優點，產生 {n_new_prompts} 個新的改良版提示指令。

嚴格要求：
1. 新提示指令必須仍然是在描述『如何生成失語症治療提示句』
2. 不可變成散文、想像畫面、故事、情境描寫或詩句
3. 必須強調：簡短、具體、自然、容易理解
4. 必須強調：不可直接說出目標詞
5. 必須強調：優先選最有辨識度的特徵
6. 必須盡量適用於水果、蔬菜、動物、地點、日常用品、交通工具、交通場所、動作等
7. 每行只輸出一個新提示指令
8. 不要加任何解釋

請直接逐行輸出新的提示指令：
""".strip()


# ============================================================
# FILTER PROMPT QUALITY
# ============================================================
def is_valid_prompt_instruction(text: str) -> bool:
    if not text:
        return False

    text = str(text).strip()

    bad_patterns = [
        "湖邊", "鳥鳴", "散步", "漫步", "詩", "故事", "畫面", "療癒",
        "自然句子", "動物和地點", "想像", "風景", "情緒",
        "不要直接說出「失語症」這個詞",
        "為失語症患者生成具體、自然的語意提示句"
    ]
    if any(bp in text for bp in bad_patterns):
        return False

    good_keywords = ["提示句", "目標詞", "不可", "產生", "生成", "描述", "輸出"]
    if not any(k in text for k in good_keywords):
        return False

    return True


# ============================================================
# OUTPUT CLEANERS / PARSERS
# ============================================================
def parse_resampled_prompts(raw_text: str) -> List[str]:
    prompts = []

    for line in str(raw_text).splitlines():
        line = line.strip()
        if not line:
            continue

        line = line.lstrip("0123456789.、- ").strip()

        if is_valid_prompt_instruction(line):
            prompts.append(line)

    unique_prompts = []
    seen = set()
    for p in prompts:
        if p not in seen:
            unique_prompts.append(p)
            seen.add(p)

    return unique_prompts


def clean_generated_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    prefixes = [
        "提示句：",
        "提示：",
        "答案：",
        "輸出：",
        "語意提示：",
        "線索：",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    text = lines[0]

    if len(text) >= 2 and text[0] in ['"', '“', '「'] and text[-1] in ['"', '”', '」']:
        text = text[1:-1].strip()

    text = text.rstrip("。！？!?. ").strip()
    return text


def postprocess_therapy_cue(text: str, target_word: str) -> str:
    """
    Final validation / filtering for therapy cue output.
    Prefer simple, natural, elder-friendly cues.
    """
    if not text:
        return ""

    text = text.strip()

    # direct target leakage
    if target_word in text:
        return ""

    # command / dialogue / unusable style
    banned_starts = [
        "吃", "喝", "去", "來", "快", "一起", "我們", "你可以", "今天", "這是"
    ]
    for prefix in banned_starts:
        if text.startswith(prefix):
            return ""

    banned_contains = [
        "吧", "！", "!", "嗎", "一起", "可以", "請", "建議"
    ]
    for token in banned_contains:
        if token in text:
            return ""

    # too short / too long
    if len(text) < 2:
        return ""
    if len(text) > 22:
        return ""

    # overly formal / robotic
    formal_tokens = ["旅客", "辦理", "進行", "提供", "相關", "設施", "功能", "指引", "登機口", "值機"]
    if sum(token in text for token in formal_tokens) >= 2:
        return ""

    # poetic / lifestyle / non-therapy style
    poetic_tokens = ["沿途風景", "享受", "浪漫", "美景", "輕鬆", "自由自在", "快樂", "舒服"]
    if any(token in text for token in poetic_tokens):
        return ""

    # weak generic cue
    weak_generic_tokens = ["需要錢", "才能開動", "很方便", "很好用", "很常見", "常常使用"]
    if any(token in text for token in weak_generic_tokens):
        return ""

    return text


# ============================================================
# CORE LOCAL QWEN GENERATION
# ============================================================
def llm_generate_text(
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    system_message: str = "你是一位專門為失語症語言治療設計提示句的專家。",
) -> str:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_device = next(model.parameters()).device
    inputs = tokenizer(chat_text, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )

    if temperature is None or temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return result


# ============================================================
# HELPERS
# ============================================================
def generate_therapy_cue(
    instruction: str,
    target_word: str,
    pos: str = "",
    subcategory: str = "",
    demo_block: Optional[str] = None,
) -> str:
    prompt = build_generation_prompt(
        instruction=instruction,
        target_word=target_word,
        pos=pos,
        subcategory=subcategory,
        demo_block=demo_block,
    )

    valid_outputs = []

    decoding_settings = [
        {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.12},
        {"temperature": 0.3, "top_p": 0.9, "repetition_penalty": 1.12},
        {"temperature": 0.5, "top_p": 0.9, "repetition_penalty": 1.10},
    ]

    for cfg in decoding_settings:
        raw_output = llm_generate_text(
            prompt=prompt,
            max_new_tokens=16,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            repetition_penalty=cfg["repetition_penalty"],
            system_message=(
                "你是一位失語症語言治療提示句專家。"
                "你只能輸出一行簡短提示句。"
                "不能直接說出目標詞。"
                "不能輸出命令句、對話句、故事句、宣傳句、完整說明。"
            ),
        )

        clean_output = clean_generated_text(raw_output)
        clean_output = postprocess_therapy_cue(clean_output, target_word)

        if clean_output:
            valid_outputs.append(clean_output)

    if valid_outputs:
        def cue_rank_key(x: str):
            penalty = 0

            bad_tokens = ["沿途風景", "享受", "需要錢", "才能開動", "很方便", "很常見"]
            for token in bad_tokens:
                if token in x:
                   penalty += 5

            return (penalty, abs(len(x) - 10), len(x), x)

        valid_outputs = sorted(valid_outputs, key=cue_rank_key)
        return valid_outputs[0]

    return ""


def generate_resampled_prompts(
    top_prompt_rows: List[Dict],
    n_new_prompts: int = 5,
) -> List[str]:
    prompt = build_resampling_prompt(
        top_prompt_rows=top_prompt_rows,
        n_new_prompts=n_new_prompts,
    )

    raw_output = llm_generate_text(
        prompt=prompt,
        max_new_tokens=220,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        system_message="你是一位擅長 Automatic Prompt Engineering 的提示設計專家。",
    )

    parsed = parse_resampled_prompts(raw_output)
    return parsed


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    instruction = "請根據目標詞彙產生一個簡短的語意提示句，內容需具體、常見、容易理解，不可直接包含目標詞。"

    demo_block = (
        "目標詞彙：香蕉\n"
        "提示：黃色長形水果\n\n"
        "目標詞彙：葡萄\n"
        "提示：一串一串小水果\n\n"
        "目標詞彙：草莓\n"
        "提示：紅色小小有籽"
    )

    final_cue = generate_therapy_cue(
        instruction=instruction,
        target_word="蘋果",
        pos="名詞",
        subcategory="水果",
        demo_block=demo_block,
    )

    print("FINAL:", final_cue)