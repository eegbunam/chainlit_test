from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable, evaluate

from dotenv import load_dotenv
load_dotenv()

client = wrap_openai(OpenAI())

from prompts import LEETCODE_TEST_PROMPT_V1, LEETCODE_TEST_PROMPT_V2

@traceable
def leetcode_agent(inputs: dict) -> dict:
    messages = [
        {"role": "system", "content": LEETCODE_TEST_PROMPT_V1},
        *inputs["messages"]
    ]

    result = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )

    return {
        "message": {
            "role": "assistant",
            "content": result.choices[0].message.content
        }
    }

# The name or UUID of the LangSmith dataset to evaluate on.
data = "Leetcode Challenges"

# A string to prefix the experiment name with.
experiment_prefix = "Leetcode unit tests"

def correctness_evaluator(run, example) -> dict:
    """
    Evaluates the correctness of generated unit tests
    
    Args:
        run: Contains the run information including inputs and outputs
        example: Contains the reference example if available
    
    Returns:
        Dictionary with score (0-1) and explanation
    """
    # Extract the original LeetCode problem from inputs
    leetcode_problem = run.inputs["inputs"]["messages"][-1]["content"]
    
    # Extract the model's generated tests
    generated_tests = run.outputs["message"]["content"]
    
    # Rest of the evaluation logic remains the same
    evaluation_prompt = f"""
    Given this LeetCode problem:
    {leetcode_problem}

    Evaluate these unit tests for valid input/output pairs:
    {generated_tests}
    
    Score from 0-4:
    4 = All test cases have clear, valid input/output pairs that match the problem requirements
    3 = Most test cases are valid, but some missing/unclear outputs or don't match problem requirements
    2 = About half the test cases have valid input/output pairs
    1 = Few valid input/output pairs
    0 = No valid input/output pairs or tests don't match problem requirements
    
    Return only the number (0-4).
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a test evaluation assistant. Respond only with a number 0-4."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0
    )
    
    try:
        score = int(response.choices[0].message.content.strip())
        return {
            "key": "correctness score",
            "score": score / 4,  # Normalize to 0-1
            "explanation": f"Test correctness score: {score}/4"
        }
    except ValueError:
        return {
            "key": "correctness score",
            "score": 0,
            "explanation": "Failed to parse score"
        }

# List of evaluators to score the outputs of target task
evaluators = [
    correctness_evaluator
]

# Evaluate the target task
results = evaluate(
    leetcode_agent,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix
)