# Developer Tools

`tools/` contains local utility scripts that help developers inspect or smoke-test the project.

These scripts are not imported by the FastAPI application. Keep them small, explicit and safe to run from the repository root.

Current tools:

- `test_local_llm.py`: checks Ollama connectivity and sends a small controlled MASI chatbot prompt.
