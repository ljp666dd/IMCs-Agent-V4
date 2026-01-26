from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from eval_prompt import EVALUATION_PROMPT, ELO_PROMPT
from openai import OpenAI, APIError
import json
import os
from functools import partial
import multiprocessing
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from single_agent_with_rag import _single_agent_answer_with_rag, _single_agent_answer_with_rag_cot
from multiagent import _multiagent_with_rag_cot
from autogen_core.models import ModelFamily


OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""
MODEL_NAME = ""
DATASET_PATH = "backend/evaluate/eval_rag_dataset"
EVAL_RESULT_PATH = "backend/evaluate/eval_rag_result"


def load_eval_rag_dataset(dataset_path: str) -> DatasetDict:
    """Loads the eval_rag_dataset from disk.

    Args:
        dataset_path (str): The path to the dataset.

    Returns:
        DatasetDict: The loaded dataset.
    """
    return load_from_disk(dataset_path)


def get_response_from_llm(messages: list[dict], tools: list = None, model: str = None):
    if model is None:
        raise ValueError("Model must be specified.")
    
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    try:
        if tools is None:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools
            )
        content = response.choices[0].message.content
        return content
    
    except APIError as e:
        print(e)
        return "apierror"

    except Exception as e:
        print(e)
        return "error"


def _single_model_answer(question: str, model: str):
    """Answers a question with a single model.

    Args:
        question (str): The question to answer.
        context (str): The context to answer the question in.
        model (str): The model to use.

    Returns:
        str: The answer.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]

    if model == "o1-mini" or model == "o3-mini":
        messages = [{"role": "user", "content": question}]

    answer = get_response_from_llm(messages, model=model)
    if model == "deepseek-reasoner":
        answer = answer.split("</think>")[-1].strip()
    return answer


def single_model_answer(model: str):
    eval_dataset = load_eval_rag_dataset(DATASET_PATH)
    num_threads = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_threads) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(_single_model_answer, model=model),
                    eval_dataset['question'],
                ),
                total=len(eval_dataset),
                desc=f"{model} Answering:",
            )
        )
    final_result = []
    for i, idx in enumerate(eval_dataset):
        final_result.append({"question": idx['question'], "answer": results[i], "source_doc": idx['source_doc']})
    
    os.makedirs(os.path.join(EVAL_RESULT_PATH, model), exist_ok=True)
    with open(f"{EVAL_RESULT_PATH}/{model}/single_model_answer.json", "w") as f:
        json.dump(final_result, f, indent=2)


def run_async_in_process(func, *args, **kwargs):
    return asyncio.run(func(*args, **kwargs))


def single_model_answer_with_rag(model: str):
    eval_dataset = load_eval_rag_dataset(DATASET_PATH)
    num_threads = 32 # multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_threads) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(run_async_in_process, _single_agent_answer_with_rag, model=model),
                    eval_dataset['question'],
                ),
                total=len(eval_dataset),
                desc=f"{model} Answering:",
            )
        )
    final_result = []
    for i, idx in enumerate(eval_dataset):
        final_result.append({"question": idx['question'], "answer": results[i], "source_doc": idx['source_doc']})
    
    os.makedirs(os.path.join(EVAL_RESULT_PATH, model), exist_ok=True)
    with open(f"{EVAL_RESULT_PATH}/{model}/single_model_answer_with_rag.json", "w") as f:
        json.dump(final_result, f, indent=2)


def single_model_answer_with_rag_cot(model: str):
    eval_dataset = load_eval_rag_dataset(DATASET_PATH)
    num_threads = 8 #multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_threads) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(run_async_in_process, _single_agent_answer_with_rag_cot, model=model),
                    eval_dataset['question'],
                ),
                total=len(eval_dataset),
                desc=f"{model} Answering:",
            )
        )
    final_result = []
    for i, idx in enumerate(eval_dataset):
        final_result.append({"question": idx['question'], "answer": results[i], "source_doc": idx['source_doc']})
    
    os.makedirs(os.path.join(EVAL_RESULT_PATH, model), exist_ok=True)
    with open(f"{EVAL_RESULT_PATH}/{model}/single_model_answer_with_rag_cot.json", "w") as f:
        json.dump(final_result, f, indent=2)


def multiagent_with_rag_cot(model: str):
    eval_dataset = load_eval_rag_dataset(DATASET_PATH)
    num_threads = 8 #multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_threads) as pool:
        results = list(
            tqdm(
                pool.imap(
                    partial(run_async_in_process, _multiagent_with_rag_cot, model_name=model),
                    eval_dataset['question'],
                ),
                total=len(eval_dataset),
                desc=f"{model} Answering:",
            )
        )
    final_result = []
    for i, idx in enumerate(eval_dataset):
        # if model == "deepseek-r1":
        #     results[i] = results[i].split("</think>")[-1].strip()
        final_result.append({"question": idx['question'], "answer": results[i], "topic": idx["topic"], "source_doc": idx['source_doc']})
    
    os.makedirs(os.path.join(EVAL_RESULT_PATH, model), exist_ok=True)
    with open(f"{EVAL_RESULT_PATH}/{model}/multiagent_with_rag_cot.json", "w") as f:
        json.dump(final_result, f, indent=2)


def elo_evaluation(qa_json_path_a: str, qa_json_path_b: str):
    with open(qa_json_path_a, "r") as f:
        qa_data_a = json.load(f)
    with open(qa_json_path_b, "r") as f:
        qa_data_b = json.load(f)
    
    eval_dataset = load_eval_rag_dataset(DATASET_PATH)
    args = []
    for idx, item in enumerate(eval_dataset):
        if qa_data_a[idx]['question'] == item['question'] and qa_data_b[idx]['question'] == item['question']:
            arg = {
                "instruction": item['question'],
                "response1": qa_data_a[idx]["answer"],
                "response2": qa_data_b[idx]["answer"],
                "reference_answer": item["answer"],
                "model": MODEL_NAME,
                "topic": item["topic"]
                }
            args.append(arg)

    num_threads = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_threads) as pool:
        results = list(
            tqdm(
                pool.imap(
                    _elo_evaluation,
                    args
                ),
                desc="Evaluating",
                total=len(args)
            )
        )
        a_win = 0
        b_win = 0
        tie = 0
        topic_stats = {
            "synthesis": {"a_win": 0, "b_win": 0, "tie": 0},
            "structure": {"a_win": 0, "b_win": 0, "tie": 0},
            "property": {"a_win": 0, "b_win": 0, "tie": 0},
            "application": {"a_win": 0, "b_win": 0, "tie": 0},
            "other": {"a_win": 0, "b_win": 0, "tie": 0}
        }
        detailed_results = []
        for idx, result in enumerate(results):
            if "[RESULT]" in result:
                feedback, score = result.split("[RESULT]")[0].strip(), result.split("[RESULT]")[1].strip() 
                feedback = feedback.strip()
                score = score.strip()
            else:
                feedback = result.strip()
                score = ""

            topic = args[idx]["topic"].lower()
            if "synthesis" in topic:
                topic = "synthesis"
            elif "structure" in topic:
                topic = "structure"
            elif "property" in topic:
                topic = "property"
            elif "application" in topic:
                topic = "application"
            else:
                topic = "other"

            if "A" in score:
                score = "A"
                a_win += 1
                topic_stats[topic]["a_win"] += 1
            elif "B" in score:
                score = "B"
                b_win += 1
                topic_stats[topic]["b_win"] += 1
            else:
                score = "Tie"
                tie += 1
                topic_stats[topic]["tie"] += 1

            detailed_results.append({
                "question": args[idx]["instruction"],
                "response_a": args[idx]["response1"],
                "response_b": args[idx]["response2"],
                "reference_answer": args[idx]["reference_answer"],
                "feedback": feedback,
                "winner": score,
                "topic": topic
            })
        
        total_comparisons = a_win + b_win #+ tie
        a_win_rate = a_win / total_comparisons if total_comparisons > 0 else 0
        b_win_rate = b_win / total_comparisons if total_comparisons > 0 else 0
        tie_rate = tie / total_comparisons if total_comparisons > 0 else 0
        
        summary = {
            "model_a": qa_json_path_a,
            "model_b": qa_json_path_b,
            "total_comparisons": total_comparisons,
            "model_a_wins": a_win,
            "model_b_wins": b_win,
            "ties": tie,
            "model_a_win_rate": a_win_rate,
            "model_b_win_rate": b_win_rate,
            "tie_rate": tie_rate,
            "topic_stats": topic_stats
        }
        
        print(f"Summary:")
        print(f"Total comparisons: {total_comparisons}")
        print(f"{qa_json_path_a} wins: {a_win} (Win rate: {a_win_rate:.2%})")
        print(f"{qa_json_path_b} wins: {b_win} (Win rate: {b_win_rate:.2%})")
        print(f"Ties: {tie} (Tie rate: {tie_rate:.2%})")
        print("\nTopic-wise statistics:")
        for topic, stats in topic_stats.items():
            total = stats["a_win"] + stats["b_win"] + stats["tie"]
            if total > 0:
                print(f"{topic.capitalize()}:")
                print(f"  Model A wins: {stats['a_win']} (Win rate: {stats['a_win']/total:.2%})")
                print(f"  Model B wins: {stats['b_win']} (Win rate: {stats['b_win']/total:.2%})")
                print(f"  Ties: {stats['tie']} (Tie rate: {stats['tie']/total:.2%})")
        
        # Save detailed results and summary to a JSON file
        a_name = qa_json_path_a.split("/")[-2] + "_" + qa_json_path_a.split("/")[-1].split(".")[0]
        b_name = qa_json_path_b.split("/")[-2] + "_" + qa_json_path_b.split("/")[-1].split(".")[0]
        elo_path = os.path.join(EVAL_RESULT_PATH, "elo_evaluation_results")
        os.makedirs(elo_path, exist_ok=True)
        result_file_path = f"{elo_path}/{a_name}-vs-{b_name}.json"
        with open(result_file_path, "w", encoding='utf-8') as f:
            json.dump({"summary": summary, "detailed_results": detailed_results}, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {result_file_path}")

    
def _elo_evaluation(args: dict):
    instruction = args["instruction"]
    response1 = args["response1"]
    response2 = args["response2"]
    reference_answer = args["reference_answer"]
    model = args["model"]
    messages = [
        {"role": "system", "content": "You are a fair evaluator language model."},
        {"role": "user", "content": ELO_PROMPT.format(instruction=instruction, response1=response1, response2=response2)},
    ]

    eval_result = get_response_from_llm(messages, model=model)
    return eval_result



if __name__ == "__main__":
    # single_model_answer(model="gpt-4o-mini")
    # single_model_answer(model="gemini-1.5-pro")
    # single_model_answer(model="gemini-1.5-pro")
    # single_model_answer(model="o3-mini")
    # single_model_answer(model="chatgpt-4o-latest") # chatgpt4o-latest web
    # single_model_answer(model="deepseek-reasoner")
    # single_model_answer(model="gpt-4o-2024-08-06")
    # single_model_answer_with_rag(model="gpt-4o-2024-08-06")
    # single_model_answer_with_rag_cot(model="gpt-4o-2024-08-06") # single agent
    # single_model_answer_with_rag_cot(model="deepseek-r1") # single agentl
    # multiagent_with_rag_cot(model="gpt-4o-2024-08-06") # mars(gpt-4o)
    # multiagent_with_rag_cot(model="deepseek-r1")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-r1/single_model_answer_with_rag_cot.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/single_model_answer_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-reasoner/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/single_model_answer_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/single_model_answer_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/chatgpt-4o-latest/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/single_model_answer_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/single_model_answer_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-reasoner/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-r1/multiagent_with_rag_cot.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/o3-mini/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-r1/multiagent_with_rag_cot.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-r1/multiagent_with_rag_cot.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/chatgpt-4o-latest/single_model_answer.json") # chatgpt4o
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-reasoner/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/chatgpt-4o-latest/single_model_answer.json") # chatgpt4o
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/chatgpt-4o-latest/single_model_answer.json") # chatgpt4o
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/chatgpt-4o-latest/single_model_answer.json") # chatgpt4o

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/chatgpt-4o-latest/single_model_answer.json") # chatgpt4o

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer_with_rag_cot.json") # single agent
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json") # gpt4o
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/deepseek-r1/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-mini/single_model_answer.json") # gpt4o

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer_with_rag_cot.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/o3-mini/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-reasoner/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/o3-mini/multiagent_with_rag_cot.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/o3-mini/single_model_answer.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer_with_rag_cot.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-reasoner/single_model_answer.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/o3-mini/single_model_answer.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-reasoner/single_model_answer.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/o3-mini/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-reasoner/single_model_answer.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-mini/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gemini-1.5-pro/single_model_answer.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-mini/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-mini/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer_with_rag_cot.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-mini/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/o3-mini/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-reasoner/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gemini-1.5-pro/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gemini-1.5-pro/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer_with_rag_cot.json")

    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gemini-1.5-pro/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/multiagent_with_rag_cot.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/o3-mini/single_model_answer.json")
    
    # elo_evaluation(
    #     qa_json_path_a=f"{EVAL_RESULT_PATH}/gpt-4o-2024-08-06/single_model_answer.json",
    #     qa_json_path_b=f"{EVAL_RESULT_PATH}/deepseek-reasoner/single_model_answer.json")
    
    pass
