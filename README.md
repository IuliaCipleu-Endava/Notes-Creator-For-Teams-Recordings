
# Notes_Creator_For_Teams_Recordings

> **Note:** This project is configured to run as part of a Power Automate flow for automated meeting note generation and processing.

Generates structured meeting notes from Microsoft Teams auto-generated transcripts (.docx or .vtt).

## Features

- Extracts summary, to-do items, solved/completed actions, issues, suggestions, decisions, and important dates.
- Supports both English and Romanian.
- Uses a local GGUF LLM model (via llama.cpp backend) for advanced note extraction.
- Heuristic keyword-based extraction as fallback.
- Outputs both .txt and .docx summaries.
- Simple HTTP API for integration.

## Requirements

- Python 3.8+
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Phi-2](https://huggingface.co/afrideva/dolphin-2_6-phi-2_oasst2_chatML_V2-GGUF)

Install dependencies:

```sh
pip install -r requirements.txt
```

## Setup

1. Place your GGUF model file in the project directory.
2. Set the `GGUF_MODEL_PATH` in `.env` to the path of your GGUF model.
3. Ensure the `config/` directory contains the keyword and stopword lists.

## Usage

### Command Line

The main processing logic is in `main_param.py`. To process meetings, use the HTTP API or call the functions directly.

### HTTP API

Start the server:

```sh
python main_param.py
```

The server listens on `http://127.0.0.1:8000/process`.

#### Example request

```json
POST /process
Content-Type: application/json

{
	"input_folder": "path/to/transcripts",
	"output_folder": "path/to/output",
	"previous_count": 0,
	"current_count": 10
}
```

- `input_folder`: Folder containing Teams `.docx` or `.vtt` transcripts.
- `output_folder`: Where summaries will be saved.
- `previous_count`, `current_count`: Used to process only new files.

### Output

For each processed file, you get:

- `Summary-<filename>.txt`: Human-readable summary.
- `Summary-<filename>.docx`: Formatted Word document.
- `Summary-<filename>.raw.txt`: Raw LLM output for debugging.

## Configuration

Keyword and stopword lists are in the `config/` directory. You can customize these for your language or organization.

## License

MIT License. See [LICENSE](LICENSE).
