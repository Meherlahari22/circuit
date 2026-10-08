import os
import traceback
from typing import TypedDict, List, Optional

from flask import Flask, request, render_template_string, jsonify
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# GEMINI API KEY
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it in Render Environment Variables."
    )


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)


# ============================================================
# RESPONSE CLEANER
# ============================================================

def clean_response(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(str(item["text"]))
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts).strip()

    return str(content).strip()


# ============================================================
# CIRCUIT STATE
# ============================================================

class CircuitState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    user_input: Optional[str]
    circuit_information: Optional[str]
    analysis: Optional[str]
    verification: Optional[str]
    final_output: Optional[str]
    retry_count: int


# ============================================================
# CIRCUIT ANALYZER TOOL
# ============================================================

@tool
def analyze_circuit(user_input: str) -> str:
    """Analyze a circuit description."""

    prompt = f"""
You are an expert Electronics and Circuit Analysis Engineer.

Analyze the following circuit:

USER INPUT:
{user_input}

Provide the answer in this exact structure:

CIRCUIT TYPE:
<type of circuit>

COMPONENTS:
<list all components and values>

CONNECTIONS:
<explain how the components are connected>

GIVEN VALUES:
<list voltage, current, resistance, capacitance,
inductance, frequency, or other available values>

FORMULA:
<appropriate circuit formula>

CALCULATION:
<show the calculation step by step>

FINAL OUTPUT:
<give the calculated output clearly with units>

IMPORTANT RULES:

1. Do not invent component values.
2. Do not invent connections.
3. Use the values given by the user.
4. If information required for calculation is missing,
   write exactly:
   INFORMATION MISSING
5. Use correct electrical units.
6. Check the calculation before giving the final output.
7. Keep the explanation simple and clear.
"""

    response = llm.invoke(prompt)
    return clean_response(response)


# ============================================================
# CIRCUIT ANALYZER NODE
# ============================================================

def circuit_analyzer_node(state: CircuitState):

    user_input = state["user_input"]

    try:
        result = analyze_circuit.invoke({
            "user_input": user_input
        })

        return {
            "circuit_information": result,
            "next_step": "verification"
        }

    except Exception:
        return {
            "circuit_information": "",
            "verification": "",
            "final_output": "",
            "next_step": "manager"
        }


# ============================================================
# VERIFICATION TOOL
# ============================================================

@tool
def verify_circuit(
    user_input: str,
    circuit_information: str
) -> str:
    """Verify circuit analysis and calculation."""

    prompt = f"""
You are a senior Electronics Engineer.

USER'S CIRCUIT:
{user_input}

CIRCUIT ANALYSIS:
{circuit_information}

Verify the analysis carefully.

Check:

1. Components and their values
2. Circuit connections
3. Circuit type
4. Formula used
5. Mathematical calculation
6. Units
7. Final output

Do not introduce new values or assumptions.

Return exactly this format:

VERIFICATION:
PASS or FAIL

CHECK:
<brief explanation>

CORRECTION:
<write correction if needed, otherwise write NONE>
"""

    response = llm.invoke(prompt)
    return clean_response(response)


# ============================================================
# VERIFICATION NODE
# ============================================================

def verification_node(state: CircuitState):

    try:
        result = verify_circuit.invoke({
            "user_input": state["user_input"],
            "circuit_information": state["circuit_information"]
        })

        return {
            "verification": result,
            "next_step": "final_output"
        }

    except Exception:
        return {
            "verification": "FAIL",
            "next_step": "final_output"
        }


# ============================================================
# FINAL OUTPUT TOOL
# ============================================================

@tool
def generate_final_output(
    user_input: str,
    circuit_information: str,
    verification: str
) -> str:
    """Generate the clean final circuit analysis."""

    prompt = f"""
You are the final response agent for a Circuit Analyzer.

USER'S CIRCUIT:
{user_input}

CIRCUIT ANALYSIS:
{circuit_information}

VERIFICATION:
{verification}

Create ONLY the final answer for the user.

Use exactly this format:

========================================
        CIRCUIT ANALYSIS RESULT
========================================

CIRCUIT TYPE:
...

COMPONENTS:
...

CONNECTIONS:
...

CALCULATION:
...

FINAL OUTPUT:
...

VERIFICATION:
PASS / FAIL

========================================

IMPORTANT:

1. Do not include internal agent messages.
2. Do not include Python objects.
3. Do not include structured response metadata.
4. Do not include signatures.
5. Do not repeat the entire analysis unnecessarily.
6. Make FINAL OUTPUT very clear.
7. Include units.
8. If information is missing, clearly say:
   INFORMATION MISSING
"""

    response = llm.invoke(prompt)
    return clean_response(response)


# ============================================================
# FINAL OUTPUT NODE
# ============================================================

def final_output_node(state: CircuitState):

    try:
        result = generate_final_output.invoke({
            "user_input": state["user_input"],
            "circuit_information": state["circuit_information"],
            "verification": state["verification"]
        })

        return {
            "final_output": result,
            "next_step": "end"
        }

    except Exception:
        return {
            "final_output": "Unable to generate final circuit analysis.",
            "next_step": "end"
        }


# ============================================================
# TASK INPUT NODE
# ============================================================

def task_input_node(state: CircuitState):

    user_input = state.get("user_input")

    if not user_input:
        raise ValueError("Circuit input is missing.")

    return {
        "user_input": user_input,
        "next_step": "circuit_analyzer"
    }


# ============================================================
# MANAGER NODE
# ============================================================

def manager_node(state: CircuitState):

    next_step = state.get("next_step")

    if next_step in {
        "circuit_analyzer",
        "verification",
        "final_output"
    }:
        return {
            "next_step": next_step
        }

    return {
        "next_step": "end"
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(CircuitState)

workflow.add_node("task_input", task_input_node)
workflow.add_node("circuit_analyzer", circuit_analyzer_node)
workflow.add_node("verification", verification_node)
workflow.add_node("final_output", final_output_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "task_input")

workflow.add_conditional_edges(
    "task_input",
    lambda state: state["next_step"],
    {
        "circuit_analyzer": "circuit_analyzer"
    }
)

workflow.add_conditional_edges(
    "circuit_analyzer",
    lambda state: state["next_step"],
    {
        "verification": "verification",
        "manager": "manager"
    }
)

workflow.add_conditional_edges(
    "verification",
    lambda state: state["next_step"],
    {
        "final_output": "final_output"
    }
)

workflow.add_conditional_edges(
    "final_output",
    lambda state: state["next_step"],
    {
        "end": END
    }
)

workflow.add_conditional_edges(
    "manager",
    lambda state: state["next_step"],
    {
        "circuit_analyzer": "circuit_analyzer",
        "verification": "verification",
        "final_output": "final_output",
        "end": END
    }
)

circuit_app = workflow.compile()


# ============================================================
# WEB UI
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Circuit Analyzer Agent</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 40px 20px;
        }

        .container {
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }

        h1 {
            text-align: center;
            margin-bottom: 8px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
        }

        textarea {
            width: 100%;
            min-height: 130px;
            padding: 12px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-sizing: border-box;
            resize: vertical;
        }

        button {
            margin-top: 15px;
            width: 100%;
            padding: 13px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            background: #222;
            color: white;
            cursor: pointer;
        }

        button:hover {
            background: #444;
        }

        #result {
            margin-top: 25px;
            padding: 20px;
            background: #f7f7f7;
            border-radius: 8px;
            white-space: pre-wrap;
            font-family: Consolas, monospace;
            display: none;
        }

        #loading {
            margin-top: 15px;
            text-align: center;
            display: none;
            color: #555;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>Circuit Analyzer Agent</h1>

    <div class="subtitle">
        Analyze circuits, calculate electrical values and verify results.
    </div>

    <textarea
        id="circuitInput"
        placeholder="Example: A 10V battery is connected to a 1k ohm resistor. Find the current through the resistor."
    ></textarea>

    <button onclick="analyzeCircuit()">
        Analyze Circuit
    </button>

    <div id="loading">
        Analyzing circuit... Please wait.
    </div>

    <div id="result"></div>

</div>

<script>

async function analyzeCircuit() {

    const input = document.getElementById("circuitInput").value;
    const result = document.getElementById("result");
    const loading = document.getElementById("loading");

    if (!input.trim()) {
        alert("Please enter a circuit description.");
        return;
    }

    result.style.display = "none";
    loading.style.display = "block";

    try {

        const response = await fetch("/agent/playground/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                circuit: input
            })
        });

        const data = await response.json();

        if (data.success) {
            result.textContent = data.result;
        } else {
            result.textContent = "ERROR:\\n" + data.error;
        }

    } catch (error) {

        result.textContent =
            "ERROR:\\n" + error.message;

    }

    loading.style.display = "none";
    result.style.display = "block";
}

</script>

</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return (
        '<h2>Circuit Analyzer Agent</h2>'
        '<p>Open <a href="/agent/playground/">'
        '/agent/playground/</a> to use the application.</p>'
    )


@app.route("/agent/playground/", methods=["GET"])
def playground():
    return render_template_string(HTML_PAGE)


@app.route("/agent/playground/analyze", methods=["POST"])
def analyze():

    try:

        data = request.get_json(silent=True) or {}
        user_input = data.get("circuit", "").strip()

        if not user_input:
            return jsonify({
                "success": False,
                "error": "Circuit description is required."
            }), 400

        initial_state = {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "next_step": "circuit_analyzer",
            "user_input": user_input,
            "circuit_information": "",
            "analysis": "",
            "verification": "",
            "final_output": "",
            "retry_count": 0
        }

        result = circuit_app.invoke(initial_state)

        return jsonify({
            "success": True,
            "result": result.get(
                "final_output",
                "No final output generated."
            )
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(
        host="0.0.0.0",
        port=port
    )
