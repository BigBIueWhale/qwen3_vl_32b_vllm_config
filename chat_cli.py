import argparse
import openai
import sys

def main():
    parser = argparse.ArgumentParser(description="CLI for chatting with Qwen3-VL-32B-Thinking via vLLM server.")
    parser.add_argument("--host", type=str, default="http://172.17.0.1:8000/v1", help="vLLM server base URL")
    parser.add_argument("--model", type=str, default="cpatonn/Qwen3-VL-32B-Thinking-AWQ-4bit", help="Model name")
    parser.add_argument("--thinking-budget", type=int, default=8192, help="Thinking budget in tokens (0 to disable thinking)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for generation")
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.host, api_key="EMPTY")

    messages = []
    print("Chat CLI started. Type '/exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "/exit":
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        if args.thinking_budget == 0:
            # Design decision: Allow disabling thinking via flag, even though it's a thinking model, for flexibility. User can choose.
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=True,
                max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            full_content = ""
            print("Assistant: ", end="", flush=True)
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                full_content += delta
            print()
            messages.append({"role": "assistant", "content": full_content})
        else:
            # Design decision: Use streaming for the first generation (thinking phase) to provide real-time output. Collect full content and finish_reason from the stream.
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=True,
                max_tokens=args.thinking_budget,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
            full_thinking = ""
            finish_reason = None
            print("Thinking: ", end="", flush=True)
            for chunk in response:
                if chunk.choices[0].finish_reason is not None:
                    finish_reason = chunk.choices[0].finish_reason
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
                full_thinking += delta
            print()

            # Design decision: Check if thinking is incomplete based on finish_reason and presence of '</think>'. Use 'max_tokens' as the indicator for reaching limit in vLLM/OpenAI compat.
            incomplete = (finish_reason == 'max_tokens') and ('</think>' not in full_thinking)

            assistant_content = full_thinking

            if incomplete:
                # Design decision: Append the early-stopping prompt as per Qwen docs to force summary generation. Print it as continuation of thinking for user visibility.
                early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>"
                assistant_content += early_stopping_text
                print(early_stopping_text, end="", flush=True)
                print()  # New line after early stop

            # Design decision: Split the content at '</think>' to separate thinking and initial answer part, if any.
            if '</think>' in assistant_content:
                index = assistant_content.rfind('</think>') + len('</think>')
                thinking_part = assistant_content[:index]
                answer_part = assistant_content[index:]
            else:
                thinking_part = assistant_content
                answer_part = ""

            print("Assistant: ", end="", flush=True)
            if answer_part:
                print(answer_part, end="", flush=True)

            # Design decision: Continue generation if reached max_tokens in first gen, to complete the answer or generate summary.
            need_continue = (finish_reason == 'max_tokens')

            if need_continue:
                messages.append({"role": "assistant", "content": assistant_content})
                response2 = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=True,
                    max_tokens=args.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
                )
                additional_content = ""
                for chunk in response2:
                    delta = chunk.choices[0].delta.content or ""
                    print(delta, end="", flush=True)
                    additional_content += delta
                print()
                messages[-1]["content"] += additional_content
            else:
                messages.append({"role": "assistant", "content": assistant_content})

        sys.stdout.flush()

if __name__ == "__main__":
    main()
