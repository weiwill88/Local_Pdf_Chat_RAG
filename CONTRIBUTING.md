# Contributing to Local PDF Chat RAG

Thanks for helping improve this project. Contributions should keep the RAG pipeline understandable, runnable, and easy to inspect.

## Before opening an issue

- Search existing issues and pull requests.
- Confirm the problem on the current `main` branch.
- Remove API keys, private documents, personal data, and internal URLs from logs and examples.
- For usage questions, include the operating system, Python version, model backend, and the exact command you ran.

Security vulnerabilities must follow [`SECURITY.md`](SECURITY.md) and must not be posted in a public issue.

## Development setup

```bash
git clone https://github.com/weiwill88/Local_Pdf_Chat_RAG.git
cd Local_Pdf_Chat_RAG
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

## Pull request expectations

1. Create a focused branch from the latest `main`.
2. Keep the change small enough to review and explain why it belongs in this educational reference implementation.
3. Add or update tests for behavior changes.
4. Update both `README.md` and `README_EN.md` when public usage changes.
5. Run the compile and test commands above before opening the pull request.
6. Complete the pull request template and link related issues.

Please avoid unrelated formatting rewrites, generated dependency folders, model files, credentials, and private test documents.

## Design principles

- Keep parsing, retrieval, reranking, and generation boundaries visible.
- Prefer explicit, replaceable modules over hidden global behavior.
- Do not require paid services for unit tests.
- Fail with actionable messages when a model, credential, or optional dependency is missing.
- Preserve source metadata when changing retrieval behavior.

By contributing, you agree that your contribution is licensed under this repository's MIT License.
