from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import re

llm = ChatOpenAI(
    openai_api_base="http://127.0.0.1:1234/v1",
    openai_api_key="lm-studio",
    model="deepseek-coder-6.7b-instruct"
)

tavily = TavilyClient(api_key="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")

def extract_python_code(text):
    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        return '\n'.join(code_blocks).strip()
    lines = text.split('\n')
    code_lines = []
    skip_phrases = ['Here is', 'When you', 'The ', 'In this']
    for line in lines:
        if (line.strip() and
            not any(line.startswith(phrase) for phrase in skip_phrases) and
            not line.strip().startswith('[') and
            ('=' in line or 'def ' in line or 'print(' in line or 'return ' in line or line.strip().startswith('#'))):
            code_lines.append(line)
    return '\n'.join(code_lines).strip() if code_lines else text.strip()

def agent2_coder(task_request):
    print("[Agent 2] Generating code for the task...")
    response = llm.invoke([
        HumanMessage(content=(
            "Write Python code that solves this task and shows the result. "
            "Only return executable Python code, no explanations or markdown:\n"
            f"{task_request}"
        ))
    ])
    return response.content.strip()

def agent3_websearch(query):
    print("[Agent 3] Performing web search...")
    results = tavily.search(query=query, search_depth="basic", max_results=1)
    if results and results.get("results"):
        return results["results"][0]["content"].strip()
    return "No relevant web information found."

def should_search(user_input):
    print("[Agent 1] Asking LLM if web search is needed...")
    response = llm.invoke([
        HumanMessage(content=(
            "You are a decision agent. Decide if the user's question needs a web search.\n"
            "Answer with only 'YES' or 'NO'.\n\n"
            f"User's Question: {user_input}\n\n"
            "Does this question require information that might not be known locally or might change over time?"
        ))
    ])
    decision = response.content.strip().upper()
    print(f"[Agent 1] LLM decision: {decision}")
    return decision == "YES"

def agent1_messenger(user_input):
    print("\n[Agent 1] Received user input.")
    if should_search(user_input):
        print("[Agent 1] Web search approved. Fetching information...")
        web_info = agent3_websearch(user_input)
        combined_request = f"{user_input}\n\nUse this information if needed:\n{web_info}"
        coder_response = agent2_coder(combined_request)
    else:
        print("[Agent 1] No web search required. Passing prompt directly to code generator.")
        coder_response = agent2_coder(user_input)
    print("[Agent 1] Code generation complete.\n")
    return coder_response

def main():
    print("🤖 Three-Agent Python Code Generator (LLM-Driven Web Search + Tavily + LM Studio)\n")
    while True:
        user_prompt = input("You: ").strip()
        if user_prompt.lower() in ["exit", "quit"]:
            print("👋 Exiting. Goodbye!")
            break
        generated_code = agent1_messenger(user_prompt)
        print(f"\n💡 Generated Code:\n{generated_code}\n")
        try:
            extracted_code = extract_python_code(generated_code)
            if extracted_code:
                print("📊 Execution Result:")
                exec(extracted_code)
                print()
            else:
                print("⚠ No executable code detected.\n")
        except Exception as e:
            print(f"⚠ Error during code execution: {e}\n")

if __name__ == "__main__":
    main()
