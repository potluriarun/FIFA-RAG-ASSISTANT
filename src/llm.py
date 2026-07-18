"""Pipeline B step 2: wrap the `claude -p` CLI call for grounded answers.

Uses headless Claude Code (billed against the Claude Pro subscription, not
a metered API key). The prompt is piped via stdin rather than passed as a
CLI argument: retrieved-chunk prompts are large enough to risk Windows's
~8191-char command-line length limit, and stdin sidesteps shell-quoting
issues with the rulebook text (quotes, dashes, etc.) entirely. Tool access
is disabled (--tools "") since this is a pure text completion over
already-retrieved context, not an agentic task.

Run: python src/llm.py   (manual checkpoint: one hand-built prompt -> answer)
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PROMPT = """You are a football rules assistant. Answer ONLY using the \
provided context chunks from the official IFAB Laws of the Game and FIFA \
World Cup regulations.

Rules:
- Cite the source for every claim, e.g. (Law 14 | The Penalty Kick, p.129) \
or (Article 5, p.11), using the labels given in the context.
- If the context does not contain the answer, say so plainly - do not \
guess or fall back on outside knowledge.
- Be concise and direct."""

CLAUDE_TIMEOUT_SECONDS = 120


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = chunk.get("law") or chunk.get("source", "")
        context_blocks.append(
            f"[{i}] ({label}, {chunk['source']} p.{chunk['page']})\n{chunk['text']}"
        )
    context = "\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def ask_claude(prompt: str, model: str | None = None) -> str:
    cmd = [
        "claude", "-p",
        "--no-session-persistence",
        "--tools", "",
        "--append-system-prompt", SYSTEM_PROMPT,
    ]
    if model:
        cmd += ["--model", model]

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=CLAUDE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def answer_question(question: str, chunks: list[dict]) -> str:
    return ask_claude(build_prompt(question, chunks))


if __name__ == "__main__":
    from retrieve import retrieve

    question = "When exactly is a penalty kick retaken?"
    chunks = retrieve(question, top_k=5)

    print("Retrieved chunks:")
    for c in chunks:
        print(f"  - {c['source']} p.{c['page']} | {c['law']}")
    print()

    print("Asking Claude...\n")
    answer = answer_question(question, chunks)
    print("--- Answer ---")
    print(answer)
