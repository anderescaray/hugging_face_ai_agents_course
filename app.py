import os
import pandas as pd
import gradio as gr
import requests
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool

# --- Professional Environment Setup ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Model selection: Coder-32B is more precise for formatting than the 72B ---
model = LiteLLMModel(
    model_id="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
    api_key=HF_TOKEN
)

# --- Standard Toolset ---
search_tool = DuckDuckGoSearchTool()
visit_page_tool = VisitWebpageTool()

# --- Advanced Agent Configuration ---
# We use CodeAgent because it's the gold standard for GAIA (it calculates and parses).
smol_agent = CodeAgent(
    tools=[search_tool, visit_page_tool],
    model=model,
    max_steps=12,
    verbosity_level=1,
    additional_authorized_imports=["pandas", "numpy", "re", "math", "datetime"]
)

class GAIAAssistant:
    """Enterprise-grade assistant focused on exact-match accuracy for GAIA."""
    
    def __init__(self):
        # The key is to tell the agent to use the final_answer tool with ONLY the value.
        self.strategy = (
            "You are a precise data extraction agent. \n"
            "1. Use 'search' and 'visit_webpage' to find the specific fact requested.\n"
            "2. Once found, provide ONLY the raw value (number, date, or name).\n"
            "3. DO NOT include units, 'The answer is', or explanations.\n"
        )

    def __call__(self, question: str) -> str:
        print(f"\n[EVALUATING]: {question[:100]}...")
        
        # We enforce strict formatting in the final prompt execution
        formatted_prompt = f"{self.strategy}\nQuestion: {question}"
        
        try:
            # We call the agent's run method. 
            # smolagents will try to return the result of the 'final_answer' tool.
            result = smol_agent.run(formatted_prompt)
            
            # Post-processing: remove common 'noise' words
            answer = str(result).strip()
            # If the agent returns a full sentence, this is a safety net
            if "is " in answer.lower() and len(answer) > 20:
                 # Simple heuristic: take the last word if it's a long sentence
                 answer = answer.split()[-1].rstrip('.')
            
            print(f"[FINAL OUTPUT]: {answer}")
            return answer
        except Exception as e:
            print(f"[RUNTIME ERROR]: {e}")
            return "Error"

# --- Framework Code (Keep as is, but ensuring clean submission) ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def run_evaluation_suite(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    if not profile:
        return "Authentication required.", None

    assistant = GAIAAssistant()
    
    try:
        response = requests.get(f"{DEFAULT_API_URL}/questions", timeout=15)
        questions = response.json()
    except Exception as e:
        return f"Fetch error: {e}", None

    payload = []
    log = []
    
    for item in questions:
        ans = assistant(item.get("question"))
        payload.append({"task_id": item.get("task_id"), "submitted_answer": ans})
        log.append({"Task": item.get("question"), "Answer": ans})

    try:
        res = requests.post(f"{DEFAULT_API_URL}/submit", json={
            "username": profile.username,
            "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
            "answers": payload
        }, timeout=60)
        data = res.json()
        return f"Score: {data.get('score')}% - {data.get('message')}", pd.DataFrame(log)
    except Exception as e:
        return f"Submit error: {e}", pd.DataFrame(log)

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# GAIA High-Accuracy Solver")
    gr.LoginButton()
    btn = gr.Button("Run Benchmark", variant="primary")
    status = gr.Textbox(label="Status")
    table = gr.DataFrame()
    btn.click(fn=run_evaluation_suite, outputs=[status, table])

if __name__ == "__main__":
    demo.launch()