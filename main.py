import os
import json
import csv
from typing import List
from tqdm import tqdm

import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from huggingface_hub import InferenceClient


# ================================
# 1. API key / Client setting
# ================================

# API key
OPENAI_API_KEY = "api-key"
GOOGLE_API_KEY = "api-key"
ANTHROPIC_API_KEY = "api-key"
HUGGINGFACE_API_KEY = "api-key"

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

llama_client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=HUGGINGFACE_API_KEY
)


# ================================
# 2. Prompt
# ================================

# 2-1: High_Econ

def high_econ_base():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S1, S2, ..., total S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: High school
    - Subject: Economics
    - Class size: Medium (around 30 students)
    - Teacher: a Korean L1 teacher with 8 years of teaching experience
    - Students: Korean L1 learners of English

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


def high_econ_int():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: High school
    - Subject: Economics
    - Class size: Medium (around 30 students)
    - Teacher: a Korean L1 teacher with 8 years of teaching experience
    - Students: Korean L1 learners of English

    Language-related episodes (LREs) can occur:
    - Pre-emptively: the teacher provides explanations or guidance about language use without any errors.
    - Reactively: in response to student errors or comprehension problems, the teacher may reformulate the student's utterance, request clarification, prompt self-correction, or explicitly correct the error.
    - Language focus: vocabulary (general/technical), grammar, pronunciation, expressions

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """

def high_econ_freq():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: High school
    - Subject: Economics
    - Class size: Medium (around 30 students)
    - Teacher: a Korean L1 teacher with 8 years of teaching experience
    - Students: Korean L1 learners of English

    In this classroom setting, language-related episodes (LREs) tend to occur approximately every 10–11 turns (about 9–10% of the interaction) of the classroom discourse.

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


#0406version
def high_econ_int_freq():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: High school
    - Subject: Economics
    - Class size: Medium (around 30 students)
    - Teacher: a Korean L1 teacher with 8 years of teaching experience
    - Students: Korean L1 learners of English

    In this classroom setting, language-related episodes (LREs) tend to occur approximately every 10–11 turns (about 9–10% of the interaction) of the classroom discourse.

    Language-related episodes (LREs) can occur:
    - Pre-emptively: the teacher provides explanations or guidance about language use without any errors.
    - Reactively: in response to student errors or comprehension problems, the teacher may reformulate the student's utterance, request clarification, prompt self-correction, or explicitly correct the error.
    - Language focus: vocabulary (general/technical), grammar, pronunciation, expressions

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


# 2-2: High_Math

def high_math_base():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S1, S2, ..., total S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: High school
    - Subject: Mathematics
    - Class size: Small (around 20 students)
    - Teacher: an English L1 teacher with 7 years of teaching experience
    - Students: Korean L1 learners of English

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


def high_math_int():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: High school
    - Subject: Mathematics
    - Class size: Small (around 20 students)
    - Teacher: an English L1 teacher with 7 years of teaching experience
    - Students: Korean L1 learners of English

    Language-related episodes (LREs) can occur:
    - Pre-emptively: the teacher provides explanations or guidance about language use without any errors.
    - Reactively: in response to student errors or comprehension problems, the teacher may reformulate the student's utterance, request clarification, prompt self-correction, or explicitly correct the error.
    - Language focus: vocabulary (general/technical), grammar, pronunciation, expressions

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


def high_math_freq():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: High school
    - Subject: Mathematics
    - Class size: Small (around 20 students)
    - Teacher: an English L1 teacher with 7 years of teaching experience
    - Students: Korean L1 learners of English

    In this classroom setting, language-related episodes (LREs) tend to occur approximately every 10–11 turns (about 9–10% of the interaction) of the classroom discourse.

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


def high_math_int_freq():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: High school
    - Subject: Mathematics
    - Class size: Small (around 20 students)
    - Teacher: an English L1 teacher with 7 years of teaching experience
    - Students: Korean L1 learners of English

    In this classroom setting, language-related episodes (LREs) tend to occur approximately every 10–11 turns (about 9–10% of the interaction) of the classroom discourse.

    Language-related episodes (LREs) can occur:
    - Pre-emptively: the teacher provides explanations or guidance about language use without any errors.
    - Reactively: in response to student errors or comprehension problems, the teacher may reformulate the student's utterance, request clarification, prompt self-correction, or explicitly correct the error.
    - Language focus: vocabulary (general/technical), grammar, pronunciation, expressions

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


# 2-3: Uni_Accounting

def uni_acc_base():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S1, S2, ..., total S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: University
    - Subject: Accounting
    - Class size: Large (around 40 students)
    - Teacher: a Chinese L1 teacher with 6 years of teaching experience
    - Students: Korean L1 learners of English

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


def uni_acc_int():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: University
    - Subject: Accounting
    - Class size: Large (around 40 students)
    - Teacher: a Chinese L1 teacher with 6 years of teaching experience
    - Students: Korean L1 learners of English

    Language-related episodes (LREs) can occur:
    - Pre-emptively: the teacher provides explanations or guidance about language use without any errors.
    - Reactively: in response to student errors or comprehension problems, the teacher may reformulate the student's utterance, request clarification, prompt self-correction, or explicitly correct the error.
    - Language focus: vocabulary (general/technical), grammar, pronunciation, expressions

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """

def uni_acc_freq():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: University
    - Subject: Accounting
    - Class size: Large (around 40 students)
    - Teacher: a Chinese L1 teacher with 6 years of teaching experience
    - Students: Korean L1 learners of English

    In this classroom setting, language-related episodes (LREs) tend to occur approximately every 30-40 turns (about 2-3% of the interaction) of the classroom discourse.

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """

def uni_acc_int_freq():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: University
    - Subject: Accounting
    - Class size: Large (around 40 students)
    - Teacher: a Chinese L1 teacher with 6 years of teaching experience
    - Students: Korean L1 learners of English

    In this classroom setting, language-related episodes (LREs) tend to occur approximately every 30-40 turns (about 2-3% of the interaction) of the classroom discourse.

    Language-related episodes (LREs) can occur:
    - Pre-emptively: the teacher provides explanations or guidance about language use without any errors.
    - Reactively: in response to student errors or comprehension problems, the teacher may reformulate the student's utterance, request clarification, prompt self-correction, or explicitly correct the error.
    - Language focus: vocabulary (general/technical), grammar, pronunciation, expressions

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


# 2-4: Uni_Math

def uni_math_base():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S1, S2, ..., total S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: University
    - Subject: Mathematics
    - Class size: Medium (around 30 students)
    - Teacher: an English L1 teacher with 7 years of teaching experience
    - Students: Korean L1 learners of English

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """


def uni_math_int():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: University
    - Subject: Mathematics
    - Class size: Medium (around 30 students)
    - Teacher: an English L1 teacher with 7 years of teaching experience
    - Students: Korean L1 learners of English

    Language-related episodes (LREs) can occur:
    - Pre-emptively: the teacher provides explanations or guidance about language use without any errors.
    - Reactively: in response to student errors or comprehension problems, the teacher may reformulate the student's utterance, request clarification, prompt self-correction, or explicitly correct the error.
    - Language focus: vocabulary (general/technical), grammar, pronunciation, expressions

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """

def uni_math_freq():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: University
    - Subject: Mathematics
    - Class size: Medium (around 30 students)
    - Teacher: an English L1 teacher with 7 years of teaching experience
    - Students: Korean L1 learners of English

    In this classroom setting, language-related episodes (LREs) tend to occur approximately every 30-40 turns (about 2-3% of the interaction) of the classroom discourse.

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """

def uni_math_int_freq():
    return """
    You are a context-aware conversational agent.
    You are given a dialogue between one or more students (S) and a teacher (T) in an EMI (English-Medium Instruction) classroom setting.

    Class information:
    - Class level: University
    - Subject: Mathematics
    - Class size: Medium (around 30 students)
    - Teacher: an English L1 teacher with 7 years of teaching experience
    - Students: Korean L1 learners of English

    In this classroom setting, language-related episodes (LREs) tend to occur approximately every 30-40 turns (about 2-3% of the interaction) of the classroom discourse.

    Language-related episodes (LREs) can occur:
    - Pre-emptively: the teacher provides explanations or guidance about language use without any errors.
    - Reactively: in response to student errors or comprehension problems, the teacher may reformulate the student's utterance, request clarification, prompt self-correction, or explicitly correct the error.
    - Language focus: vocabulary (general/technical), grammar, pronunciation, expressions

    Given the class information, continue the classroom conversation in a natural and contextually appropriate way with LREs embedded. Maintain the same alternating speaker pattern and conversational style.
    Make up the classroom discourse relevant to the subject. Do not repeat utterance once it is made. Embed LREs in the classroom discourse referring to the original LREs. Do not repeat LREs once they are made.

    Continue the dialogue:
    """

# ================================
# 3. JSON → Prompt
# ================================

def build_prompt(base_prompt: str, dialogue: list) -> str:
    lines = []
    for turn in dialogue:
        speaker = turn["speaker"]
        utt = turn["utterance"]
        lines.append(f"{speaker}: {utt}")
    dialogue_text = "\n".join(lines)

    return base_prompt + "\n\nDialogue:\n" + dialogue_text + "\n\nContinue the dialogue:"


# ================================
# 4. Model calls
# ================================

def call_gpt4o(prompt: str) -> str:
    resp = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(prompt)
    return resp.text if hasattr(resp, "text") else str(resp)


def call_claude(prompt: str) -> str:
    resp = client_anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_llama(prompt: str) -> str:
    response = llama_client.chat_completion(
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7,
    )
    return response.choices[0].message["content"]

def model_call(model_name: str, prompt: str) -> str:
    if model_name == "gpt-4o":
        return call_gpt4o(prompt)
    elif model_name == "gemini":
        return call_gemini(prompt)
    elif model_name == "claude":
        return call_claude(prompt)
    elif model_name == "llama3":
        return call_llama(prompt)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ================================
# 5. Parse & Save
# ================================

def parse_agent_reply(reply: str) -> list:
    turns = []
    for line in reply.splitlines():
        if ":" in line:
            speaker, utt = line.split(":", 1)
            turns.append({
                "index": "",
                "speaker": speaker.strip(),
                "utterance": utt.strip(),
                "LRE": "",
                "source": "agent",
            })
    return turns


def save_dialogue_csv(original_dialogue: list, agent_dialogue: list, output_filename: str):
    final_output = [
        {
            "index": t.get("index", ""),
            "speaker": t["speaker"],
            "utterance": t["utterance"],
            "LRE": t.get("LRE", ""),
            "source": "original",
        }
        for t in original_dialogue
    ] + agent_dialogue

    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "speaker", "utterance", "LRE", "source"]
        )
        writer.writeheader()
        writer.writerows(final_output)


# ================================
# 6. Main function
# ================================

def agent_dialogue(json_filename: str, num_cycles: int, prompt, models: list):

    with open(json_filename, "r", encoding="utf-8") as f:
        dialogue = json.load(f)

    original_dialogue = dialogue.copy()
    base_prompt = prompt()
    full_prompt = build_prompt(base_prompt, original_dialogue)
    prompt_name = prompt.__name__

    for model_name in models:
        print(f"\n🚀 Model: {model_name}")
        agent_turns_all_cycles = []

        for cycle in tqdm(range(num_cycles), desc=f"{model_name}", leave=True):
            reply = model_call(model_name, full_prompt)
            agent_turns = parse_agent_reply(reply)
            agent_turns_all_cycles.extend(agent_turns)

        base_name = os.path.splitext(os.path.basename(json_filename))[0]
        output_filename = f"agent_{prompt_name}_{model_name}.csv"

        save_dialogue_csv(original_dialogue, agent_turns_all_cycles, output_filename)
        print(f"✅ Saved: {output_filename}")

# ================================
# 7. Run
# ================================
models = ["gpt-4o", "gemini", "claude", "llama3"]
high_econ = [high_econ_base, high_econ_int, high_econ_freq, high_econ_int_freq]
high_math = [high_math_base, high_math_int, high_math_freq, high_math_int_freq]
uni_acc = [uni_acc_base, uni_acc_int, uni_acc_freq, uni_acc_int_freq]
uni_math = [uni_math_base, uni_math_int, uni_math_freq, uni_math_int_freq]

# run1: high_econ
for model in models:
    for p in high_econ:
        agent_dialogue(
            json_filename="high_econ.json",
            num_cycles=20,
            prompt=p,
            models=[model]
        )

# run2: high_math
for model in models:
    for p in high_math:
        agent_dialogue(
            json_filename="high_math.json",
            num_cycles=20,
            prompt=p,
            models=[model]
        )

# run3: uni_acc
for model in models:
    for p in uni_acc:
        agent_dialogue(
            json_filename="uni_acc.json",
            num_cycles=20,
            prompt=p,
            models=[model]
        )

# run4: uni_math
for model in models:
    for p in uni_math:
        agent_dialogue(
            json_filename="uni_math.json",
            num_cycles=20,
            prompt=p,
            models=[model]
        )
