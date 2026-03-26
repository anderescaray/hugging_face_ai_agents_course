import os
import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool

# 1. CARGA DE CONFIGURACIÓN
load_dotenv()
# Intentamos obtener el token del Secret de HF o del archivo .env local
hf_token = os.getenv("HF_TOKEN")

# 2. CONFIGURACIÓN DEL AGENTE (smolagents)
# Usamos Qwen 2.5 Coder 32B por su gran capacidad de razonamiento
model = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token
)

search_tool = DuckDuckGoSearchTool()

# Inicializamos el agente fuera de la clase para optimizar rendimiento
smol_agent = CodeAgent(
    tools=[search_tool],
    model=model,
    max_steps=10,
    verbosity_level=1,
    additional_authorized_imports=["pandas", "numpy", "re", "math"]
)

# 3. CONSTANTES Y CLASE PARA EL CURSO
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class BasicAgent:
    def __init__(self):
        print("--- Agente smolagents inicializado con éxito ---")

    def __call__(self, question: str) -> str:
        """
        Esta función es la que llama el evaluador del curso.
        """
        print(f"\n[PROCESANDO PREGUNTA]: {question[:100]}...")
        
        # Prompt específico para que el agente sea muy directo (clave para GAIA)
        prompt = f"""Solve the following question precisely. 
        IMPORTANT: Your final answer must be extremely concise. 
        If the answer is a number, date, or name, return ONLY that value. 
        Do not include any conversational text or explanations.

        Question: {question}"""
        
        try:
            # Ejecutamos la lógica de smolagents
            result = smol_agent.run(prompt)
            final_answer = str(result).strip()
            print(f"[RESPUESTA GENERADA]: {final_answer}")
            return final_answer
        except Exception as e:
            print(f"Error en el agente: {e}")
            return "Error procesando la respuesta."

# 4. LÓGICA DE EVALUACIÓN Y ENVÍO (Mantenemos la del curso)
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if not profile:
        return "Por favor, inicia sesión con Hugging Face usando el botón de arriba.", None

    username = profile.username
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # Instanciamos nuestro agente
    try:
        agent = BasicAgent()
    except Exception as e:
        return f"Error inicializando el agente: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    # Obtener preguntas
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
    except Exception as e:
        return f"Error obteniendo preguntas: {e}", None

    # Ejecutar agente en todas las preguntas
    results_log = []
    answers_payload = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        try:
            answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Answer": answer})
        except Exception as e:
            results_log.append({"Task ID": task_id, "Answer": f"ERROR: {e}"})

    # Enviar al servidor de scoring
    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload
    }

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        
        status = (
            f"✅ Envío realizado con éxito!\n"
            f"Usuario: {result_data.get('username')}\n"
            f"Puntuación: {result_data.get('score')}% "
            f"({result_data.get('correct_count')}/{result_data.get('total_attempted')} correctas)\n"
            f"Mensaje: {result_data.get('message')}"
        )
        return status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Error en el envío: {e}", pd.DataFrame(results_log)

# 5. INTERFAZ GRADIO
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Agente Final - AI Agents Course")
    gr.Markdown("Haz clic en el botón para que el agente resuelva el benchmark GAIA y envíe tu nota.")
    
    gr.LoginButton()
    run_button = gr.Button("🚀 Ejecutar Evaluación y Enviar Todo", variant="primary")
    
    status_output = gr.Textbox(label="Estado del Envío", lines=5)
    results_table = gr.DataFrame(label="Detalle de Respuestas")

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    demo.launch()