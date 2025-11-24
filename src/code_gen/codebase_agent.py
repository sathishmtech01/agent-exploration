from dotenv import load_dotenv
load_dotenv()
import os
import json
import zipfile
from pathlib import Path
from openai import OpenAI

# -----------------------------------------
# CONFIG
# -----------------------------------------
client = OpenAI()
MODEL = "gpt-4o-mini"


# -----------------------------------------
# LLM CALL (NEW OPENAI API)
# -----------------------------------------
def ask_llm(prompt: str) -> dict:
    """
    Sends the prompt to the LLM and returns a JSON-structured representation
    of a full codebase (files + content).
    """

    system_prompt = """
You are a code-generation agent.
Return ONLY valid JSON describing a project codebase.

FORMAT:
{
  "files": [
    { "path": "path/to/file", "content": "file contents" }
  ]
}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # NEW SYNTAX: response.choices[0].message.content
    raw = response.choices[0].message.content.strip()

    # Parse JSON
    return json.loads(raw)


# -----------------------------------------
# FILE GENERATION
# -----------------------------------------
def save_codebase(structure: dict, output_dir: str):
    root = Path(output_dir)
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(root.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()

    root.mkdir(parents=True, exist_ok=True)

    for file in structure["files"]:
        full_path = root / file["path"]
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(file["content"], encoding="utf-8")

    return root


# -----------------------------------------
# ZIP PACKAGE
# -----------------------------------------
def zip_codebase(folder: Path, zip_name: str) -> str:
    zip_path = f"{zip_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in folder.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(folder))
    return zip_path


# -----------------------------------------
# MAIN AGENT ENTRYPOINT
# -----------------------------------------
def generate_codebase_from_prompt(prompt: str) -> str:
    out_dir="generated_project"
    zip_name="project"
    print("🔍 Calling LLM to generate codebase...")
    structure = ask_llm(prompt)

    print("📁 Writing generated files...")
    folder = save_codebase(structure, out_dir)

    print("📦 Zipping project...")
    zip_path = zip_codebase(folder, zip_name)

    print(f"✅ Codebase ZIP ready at: {zip_path}")
    return "success"


# -----------------------------------------
# CLI MODE
# -----------------------------------------
# if __name__ == "__main__":
#     prompt = input("Enter your project prompt:\n> ")
#     generate_codebase_from_prompt(prompt)
from google.adk.tools import FunctionTool
generate_codebase= FunctionTool(generate_codebase_from_prompt)