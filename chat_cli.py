import argparse
import openai
import sys


EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>"
)


def stream_first_pass(client, model, messages, thinking_budget):
    """
    First-pass generation:
    - Streams `reasoning_content` live (printed under "Thinking: ").
    - Buffers any `content` (the post-</think> answer) without printing it yet.
    - Returns the concatenated text (reasoning + any early answer), the finish_reason,
      and a tuple (thinking_text, answer_seed_text).
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=thinking_budget,
        # Keep thinking enabled so the server applies the Qwen3 reasoning template.
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    thinking_text = []
    answer_seed = []
    finish_reason = None

    printed_thinking_header = False

    for chunk in response:
        choice = chunk.choices[0]
        delta = choice.delta

        # vLLM streams reasoning tokens separately as `reasoning_content`
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            if not printed_thinking_header:
                print("Thinking: ", end="", flush=True)
                printed_thinking_header = True
            print(rc, end="", flush=True)
            thinking_text.append(rc)

        # The final answer (after </think>) comes via normal `content`
        c = delta.content or ""
        if c:
            # Buffer answer text; we will print it later under "Assistant:"
            answer_seed.append(c)

        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

    if printed_thinking_header:
        print()  # newline after streaming "Thinking: ..."

    full_text = "".join(thinking_text) + "".join(answer_seed)
    return full_text, finish_reason, ("".join(thinking_text), "".join(answer_seed))


def split_thinking_and_answer(text):
    """
    Splits concatenated text into (thinking_part, answer_part) at the *last* </think>.
    If </think> is missing, returns (text, "").
    """
    end_tag = "</think>"
    idx = text.rfind(end_tag)
    if idx == -1:
        return text, ""
    cut = idx + len(end_tag)
    return text[:cut], text[cut:]


def stream_second_pass(client, model, messages, max_tokens):
    """
    Second-pass generation:
    - Continues from the assistant message we just appended.
    - Streams only the final answer (content) and prints under "Assistant:".
    - Returns the additional content appended.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    printed_answer_header = False
    additional = []

    for chunk in response:
        delta = chunk.choices[0].delta
        # On continuation we only expect/print answer content
        c = delta.content or ""
        if c:
            if not printed_answer_header:
                print("Assistant: ", end="", flush=True)
                printed_answer_header = True
            print(c, end="", flush=True)
            additional.append(c)

    if printed_answer_header:
        print()  # newline after streaming "Assistant: ..."

    return "".join(additional)


def main():
    parser = argparse.ArgumentParser(
        description="CLI for chatting with Qwen3-VL-32B-Thinking via a vLLM server (OpenAI-compatible)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://172.17.0.1:8000/v1",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cpatonn/Qwen3-VL-32B-Thinking-AWQ-4bit",
        help="Model name served by vLLM",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=8192,
        help="Thinking budget in tokens (0 to disable thinking)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for final answer continuation",
    )
    args = parser.parse_args()

    # Empty key is accepted by vLLM's OpenAI-compatible server.
    client = openai.OpenAI(base_url=args.host, api_key="EMPTY")

    messages = []
    print("Chat CLI started. Type '/exit' to quit.")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() == "/exit":
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # Hard switch to disable thinking entirely (useful for quick tests or non-reasoning models).
        if args.thinking_budget == 0:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=True,
                max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            print("Assistant: ", end="", flush=True)
            full_content = []
            for chunk in response:
                delta = chunk.choices[0].delta
                c = delta.content or ""
                if c:
                    print(c, end="", flush=True)
                    full_content.append(c)
            print()
            messages.append({"role": "assistant", "content": "".join(full_content)})
            sys.stdout.flush()
            continue

        # --- Thinking mode (two-pass thinking budget approach) ---
        # First pass: up to thinking_budget tokens. If we don't see </think>, we append an early-stopping cue.
        full_text, finish_reason, (thinking_text, answer_seed) = stream_first_pass(
            client=client,
            model=args.model,
            messages=messages,
            thinking_budget=args.thinking_budget,
        )

        # If thinking section hasn't closed and we hit budget, inject early-stopping text to force a summary.
        needs_early_stop = (finish_reason == "max_tokens") and ("</think>" not in full_text)
        if needs_early_stop:
            # Print the cue as part of "Thinking:" for user visibility (mirrors Qwen docs).
            print(EARLY_STOPPING_TEXT, flush=True)
            full_text = full_text + EARLY_STOPPING_TEXT

        # Print any already-produced answer seed (anything after </think> from pass 1).
        thinking_part, answer_part = split_thinking_and_answer(full_text)
        if answer_part:
            print("Assistant: ", end="", flush=True)
            print(answer_part, end="", flush=True)
            print()

        # Decide if we should continue. We continue if the first pass stopped by budget.
        need_continue = (finish_reason == "max_tokens")

        if need_continue:
            # Append first-pass assistant content (including any early-stop cue) so the model continues.
            messages.append({"role": "assistant", "content": full_text})
            additional = stream_second_pass(
                client=client,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
            )
            # Keep a full transcript of the assistant turn.
            messages[-1]["content"] = messages[-1]["content"] + additional
        else:
            # First pass already finished naturally; record what we have.
            messages.append({"role": "assistant", "content": full_text})

        sys.stdout.flush()


if __name__ == "__main__":
    main()
