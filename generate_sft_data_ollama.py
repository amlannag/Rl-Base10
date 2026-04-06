import json
import time
import random
import argparse
import requests
from datasets import load_dataset
from tqdm import tqdm


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:8b"

PROMPT_TEMPLATE = """Solve the following problem step by step.
You must respond in EXACTLY this format and nothing else:

<reasoning>
[step by step working]
</reasoning>
<answer>
[integer only, no commas, no units]
</answer>

Problem: {question}"""


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "")


def generate_solution(question: str, retries: int = 3) -> str | None:
    prompt = PROMPT_TEMPLATE.format(question=question)

    for attempt in range(retries):
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3, 
                    "num_predict": 512,
                }
            }, timeout=60)

            response.raise_for_status()
            text = response.json()["response"].strip()

            # Validate format
            if "<reasoning>" in text and "</reasoning>" in text and \
               "<answer>" in text and "</answer>" in text:
                return text
            else:
                print(f"  Invalid format on attempt {attempt + 1}, retrying...")

        except requests.exceptions.ConnectionError:
            print("  Cannot connect to ollama. Is it running? Try: ollama serve")
            time.sleep(2)
        except Exception as e:
            print(f"  Error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset using Llama 3 8B via ollama")
    parser.add_argument("--num_examples", type=int, default=750, help="Number of examples to generate")
    parser.add_argument("--output_file", type=str, default="sft_dataset.json", help="Output JSON file")
    parser.add_argument("--model", type=str, default="llama3:8b", help="Ollama model to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    # Load GSM8K
    print("Loading GSM8K dataset...")
    data = load_dataset("openai/gsm8k", "main")["train"]

    # Filter valid examples
    valid = []
    for item in data:
        ans = extract_hash_answer(item["answer"])
        if ans is not None:
            valid.append({"question": item["question"], "answer": ans})

    random.seed(args.seed)
    sampled = random.sample(valid, min(args.num_examples, len(valid)))

    # Resume from existing progress
    results = []
    import os
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} existing examples")
        sampled = sampled[len(results):]

    print(f"Generating {len(sampled)} examples with {MODEL}...")

    failed = 0
    for item in tqdm(sampled, desc="Generating"):
        response = generate_solution(item["question"])

        if response:
            results.append({
                "question": item["question"],
                "answer": item["answer"],
                "response": response
            })
        else:
            failed += 1
            print(f"  Failed: {item['question'][:60]}...")

        # Save every 10 examples
        if len(results) % 10 == 0:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Generated {len(results)} examples, {failed} failed")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
