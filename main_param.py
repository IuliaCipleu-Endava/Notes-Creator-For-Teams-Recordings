import sys
import json
import nltk
import docx
import re
from pathlib import Path
from collections import Counter
from langdetect import detect, DetectorFactory
import dateparser

DetectorFactory.seed = 0
nltk.download("punkt", quiet=False)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Phi-2 model
print("Loading Phi-2 model... (first time may take ~30s)")
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
phi_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float32,   # CPU-friendly
    trust_remote_code=True
)

def load_wordlist(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# -------------------------------------
# STOPWORDS
# -------------------------------------

CONFIG_DIR = Path("config")

STOPWORDS_RO = set(load_wordlist(CONFIG_DIR / "stopwords_ro.txt"))
STOPWORDS_EN = set(load_wordlist(CONFIG_DIR / "stopwords_en.txt"))

TODO_RO = load_wordlist(CONFIG_DIR / "todo_keywords_ro.txt")
TODO_EN = load_wordlist(CONFIG_DIR / "todo_keywords_en.txt")

SOLVED_RO = load_wordlist(CONFIG_DIR / "solved_keywords_ro.txt")
SOLVED_EN = load_wordlist(CONFIG_DIR / "solved_keywords_en.txt")

SUGGEST_RO = load_wordlist(CONFIG_DIR / "suggestions_keywords_ro.txt")
SUGGEST_EN = load_wordlist(CONFIG_DIR / "suggestions_keywords_en.txt")

ISSUES_RO = load_wordlist(CONFIG_DIR / "issues_keywords_ro.txt")
ISSUES_EN = load_wordlist(CONFIG_DIR / "issues_keywords_en.txt")


# -------------------------------------
# HELPERS
# -------------------------------------
def detect_language(text):
    try:
        return "romanian" if detect(text) == "ro" else "english"
    except:
        return "english"
    
def phi_generate(prompt, max_tokens=512):
    inputs = phi_tokenizer(prompt, return_tensors="pt")
    outputs = phi_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
    )
    return phi_tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_dates(text, language="english"):
    """
    Extract absolute and relative dates from transcript text
    and return them normalized.
    """
    detected = set()
    results = []

    # Split into sentences (helps avoid duplicate parsing)
    sentences = nltk.sent_tokenize(text)

    for s in sentences:
        # Use dateparser with multilingual support
        dt = dateparser.parse(
            s,
            languages=['ro', 'en'],
            settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": None,
            }
        )
        if dt:
            # Avoid duplicates and nonsense (like all sentences detecting "now")
            key = dt.date().isoformat()
            if key not in detected:
                detected.add(key)
                results.append((s, dt.date()))

    return results

def extract_with_phi(clean_text, language):
    prompt = f"""
You are an assistant that extracts structured meeting notes.

Meeting transcript:
{clean_text}

Return only JSON with fields:
- summary
- todo
- solved
- issues
- suggestions
- decisions

Use Romanian or English depending on input.
"""

    result = phi_generate(prompt, max_tokens=300)

    # Try to extract the JSON from the model output
    match = re.search(r"\{.*\}", result, flags=re.S)
    if match:
        return match.group(0)
    else:
        return result

def extract_keywords(text, language="english", n=5):
    words = nltk.word_tokenize(text.lower())
    words = [w for w in words if w.isalnum()]

    stop = STOPWORDS_RO if language == "romanian" else STOPWORDS_EN
    words = [w for w in words if w not in stop and len(w) > 3]

    freq = Counter(words)
    return [w for w, _ in freq.most_common(n)]

def clean_transcript(text):
    lines = text.split("\n")
    lines = [ln.strip() for ln in lines if ln.strip()]
    lines = [ln for ln in lines if "transcription" not in ln.lower()]

    cleaned = []
    for ln in lines:
        ln = re.sub(r"\s+\d+:\d+", "", ln).strip()
        cleaned.append(ln)

    return "\n".join(cleaned)

def categorize_sentences(text, language):
    todos, solved, suggestions, issues = [], [], [], []

    sentences = nltk.sent_tokenize(text)

    if language == "romanian":
        todo_words = TODO_RO
        solved_words = SOLVED_RO
        suggestion_words = SUGGEST_RO
        issue_words = ISSUES_RO
    else:
        todo_words = TODO_EN
        solved_words = SOLVED_EN
        suggestion_words = SUGGEST_EN
        issue_words = ISSUES_EN

    for s in sentences:
        low = s.lower()

        if any(w in low for w in todo_words):
            todos.append(s)
            continue
        
        if any(w in low for w in solved_words):
            solved.append(s)
            continue
        
        if any(w in low for w in suggestion_words):
            suggestions.append(s)
            continue
        
        if any(w in low for w in issue_words):
            issues.append(s)
            continue

    return todos, solved, suggestions, issues


def read_vtt(path: Path):
    """
    Clean Teams .vtt files:
    - Remove WEBVTT headers
    - Remove NOTE blocks
    - Remove numeric cue IDs
    - Remove timecode lines
    - Extract <v Speaker>Text</v>
    - Merge with previous speaker if needed
    """
    lines = []
    current_speaker = None

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            if not line:
                continue
            if line == "WEBVTT":
                continue
            if line.startswith("NOTE"):
                continue

            # Remove line numbers like "1", "2", "3", etc.
            if line.isdigit():
                continue

            # Skip timecodes
            if re.match(r"\d{2}:\d{2}:\d{2}\.\d{3} -->", line):
                continue

            # Full <v Speaker>Text</v>
            m = re.match(r"<v ([^>]+)>(.*?)</v>", line)
            if m:
                speaker = m.group(1).strip()
                speaker = speaker.split(".")[0]
                text = m.group(2).strip()
                lines.append(f"{speaker}: {text}")
                current_speaker = speaker
                continue

            # <v Speaker>Text (no closing tag)
            m = re.match(r"<v ([^>]+)>(.*)", line)
            if m:
                current_speaker = m.group(1).strip()
                rest = m.group(2).strip()
                if rest:
                    lines.append(f"{current_speaker}: {rest}")
                continue

            # Continuation of previous speaker
            if current_speaker:
                lines.append(f"{current_speaker}: {line}")
                continue

            # Fallback (rare)
            lines.append(line)

    # Final clean-up
    cleaned = []
    for ln in lines:
        ln = ln.replace("</v>", "").strip()
        cleaned.append(ln)

    return "\n".join(cleaned)


def read_docx(path: Path):
    doc = docx.Document(path)
    raw = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    # Remove started/stopped transcription
    raw = re.sub(r"started transcription", "", raw, flags=re.I)
    raw = re.sub(r"stopped transcription", "", raw, flags=re.I)

    # Remove timestamps (0:11)
    raw = re.sub(r"\b\d{1,2}:\d{2}\b", "", raw)

    # Remove multiple blank lines
    raw = re.sub(r"\n{2,}", "\n", raw)

    return raw.strip()



def extract_participants(text):
    participants = set()

    for line in text.splitlines():
        # Name:
        m = re.match(r"^([A-ZȘȚĂÂÎ][\wăâîșț]+(?: [A-ZȘȚĂÂÎ][\wăâîșț]+)*)\s*:", line)
        if m:
            participants.add(m.group(1))
            continue

        # Name timestamp
        m = re.match(r"^([A-ZȘȚĂÂÎ][\wăâîșț]+(?: [A-ZȘȚĂÂÎ][\wăâîșț]+)*)\s+\d{1,2}:\d{2}", line)
        if m:
            participants.add(m.group(1))
            continue

    return sorted(participants)

# -------------------------------------
# MAIN
# -------------------------------------
def main():
    args = sys.argv

    TRANSCRIPTS_FOLDER = Path(args[1])
    OUTPUT_FOLDER = Path(args[2])
    PREV = int(args[3]) if len(args) > 3 else 0
    CURR = int(args[4]) if len(args) > 4 else 0

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Number of new files to process
    N_FILES = max(CURR - PREV, 1)

    # Collect docx + vtt
    files = list(TRANSCRIPTS_FOLDER.glob("*.docx")) + list(TRANSCRIPTS_FOLDER.glob("*.vtt"))
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    files_to_process = files[:N_FILES]

    for f in files_to_process:
        print(f"Processing: {f.name}")

        if f.suffix.lower() == ".docx":
            raw = read_docx(f)
        else:
            raw = read_vtt(f)

        if not raw.strip():
            print("Empty transcript.")
            continue

        heuristic_clean = clean_transcript(raw)
        heuristic_language = detect_language(heuristic_clean)
        important_dates = extract_dates(heuristic_clean, heuristic_language)
        heuristic_participants = extract_participants(heuristic_clean)
        heuristic_todos, heuristic_solved, heuristic_suggestions, heuristic_issues = categorize_sentences(heuristic_clean, heuristic_language)
        phi_struct = extract_with_phi(heuristic_clean, heuristic_language)
        print("Phi-2 structured extraction:")
        print(phi_struct)
        
        sentences = nltk.sent_tokenize(heuristic_clean)
        summary = " ".join(sentences[:5])

        keywords = extract_keywords(heuristic_clean, heuristic_language)

        try:
            data = json.loads(phi_struct)
            summary = data.get("summary", "")
            todos = data.get("todo", [])
            solved = data.get("solved", [])
            issues = data.get("issues", [])
            suggestions = data.get("suggestions", [])
        except:
            summary = summary           # fallback to extractive
            todos = heuristic_todos
            solved = heuristic_solved
            issues = heuristic_issues
            suggestions = heuristic_suggestions

        # Build output
        txt = []
        txt.append("--- Meeting Notes Summary ---")
        txt.append(f"Language: {heuristic_language.capitalize()}")
        txt.append(f"Participants: {', '.join(heuristic_participants) if heuristic_participants else 'N/A'}")
        txt.append("")
        
        # --- Keywords ---
        txt.append("Key Points:")
        txt.extend(f"- {kw}" for kw in keywords)

        # --- Summary ---
        txt.append("Summary:")
        txt.append(summary)
        txt.append("")

        # --- Action Items / To Do ---
        txt.append("To Do:")
        if todos:
            txt.extend(f"- {t}" for t in todos)
        else:
            txt.append("- None identified")
        txt.append("")

        # --- Solved ---
        txt.append("Solved / Completed:")
        if solved:
            txt.extend(f"- {s}" for s in solved)
        else:
            txt.append("- None mentioned")
        txt.append("")

        # --- Suggestions ---
        txt.append("Suggestions:")
        if suggestions:
            txt.extend(f"- {sg}" for sg in suggestions)
        else:
            txt.append("- None identified")
        txt.append("")

        # --- Issues / Problems ---
        txt.append("Issues:")
        if issues:
            txt.extend(f"- {i}" for i in issues)
        else:
            txt.append("- No issues flagged")
        txt.append("")
        
        txt.append("Important Dates:")
        if important_dates:
            for sent, d in important_dates:
                txt.append(f"- {d}: {sent}")
        else:
            txt.append("- No important dates mentioned")
        txt.append("")

        final_txt = "\n".join(txt)

        base = f"Summary-{f.stem}"

        # Save TXT
        (OUTPUT_FOLDER / f"{base}.txt").write_text(final_txt, encoding="utf-8")

        # Save DOCX
        doc_out = docx.Document()
        doc_out.add_heading("Meeting Notes Summary", level=0)
        doc_out.add_heading("Language", level=1)
        doc_out.add_paragraph(heuristic_language.capitalize())
        doc_out.add_heading("Participants", level=1)
        doc_out.add_paragraph(", ".join(heuristic_participants) if heuristic_participants else "N/A")
        doc_out.add_heading("Summary", level=1)
        doc_out.add_paragraph(summary)
        doc_out.add_heading("Key Points", level=1)
        for kw in keywords:
            doc_out.add_paragraph(kw, style="List Bullet")
        doc_out.add_heading("To Do", level=1)
        if todos:
            for t in todos: doc_out.add_paragraph(t, style="List Bullet")
        else:
            doc_out.add_paragraph("No action items identified.")

        doc_out.add_heading("Solved / Completed", level=1)
        if solved:
            for s in solved: doc_out.add_paragraph(s, style="List Bullet")
        else:
            doc_out.add_paragraph("None mentioned.")

        doc_out.add_heading("Suggestions", level=1)
        if suggestions:
            for s in suggestions: doc_out.add_paragraph(s, style="List Bullet")
        else:
            doc_out.add_paragraph("No suggestions mentioned.")

        doc_out.add_heading("Issues", level=1)
        if issues:
            for i in issues: doc_out.add_paragraph(i, style="List Bullet")
        else:
            doc_out.add_paragraph("No issues detected.")
        
        doc_out.add_heading("Important Dates", level=1)
        if important_dates:
            for sent, d in important_dates:
                doc_out.add_paragraph(f"{d}: {sent}", style="List Bullet")
        else:
            doc_out.add_paragraph("No important dates mentioned.")


        doc_out.save(OUTPUT_FOLDER / f"{base}.docx")

        print(f"Saved: {base}.txt / .docx")

    print("Done.")

if __name__ == "__main__":
    main()
