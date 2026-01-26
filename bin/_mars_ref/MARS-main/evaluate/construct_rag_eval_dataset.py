import requests
import pandas as pd
import json
from openai import OpenAI, APIError
from tqdm import tqdm
from eval_prompt import QA_generation_prompt, question_groundedness_critique_prompt, question_relevance_critique_prompt, question_standalone_critique_prompt
import multiprocessing
from functools import partial
from datasets import Dataset, DatasetDict

# 常量
API_KEY = "dataset-OFxH5fwjOmYnfBsQkSWm8gHs"
DATASETS_NAME = ["eval-paper-new"]
N_THREADS = 32#multiprocessing.cpu_count()  # 使用所有可用的CPU核心

OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""
MODEL_NAME = ""
# Dify Datasets
DATASETS_URL = 'http://100.85.52.31:7080/v1/datasets?page=1&limit=100'
DOCUMENTS_URL = 'http://100.85.52.31:7080/v1/datasets/{}/documents'
CHUNKS_URL = 'http://100.85.52.31:7080/v1/datasets/{}/documents/{}/segments'
N_GENERATIONS = -1


def get_all_chunks(datasets_name):
    """
    获取所有知识库文档的所有块。

    Returns:
        包含所有块的列表。
    """

    headers = {'Authorization': f'Bearer {API_KEY}'}
    all_chunks = []

    # 获取数据集
    datasets_response = requests.get(DATASETS_URL, headers=headers)
    datasets = datasets_response.json()['data']

    for dataset in datasets:
        dataset_id = dataset['id']
        if dataset['name'] not in datasets_name:
            continue

        # 获取文档
        documents_response = requests.get(DOCUMENTS_URL.format(dataset_id), headers=headers)
        documents = documents_response.json()['data']

        for document in documents:
            document_id = document['id']

            # 获取块
            chunks_response = requests.get(CHUNKS_URL.format(dataset_id, document_id), headers=headers)
            chunks = chunks_response.json()['data']

            for chunk in chunks:
                if chunk['tokens'] < 150:
                    continue

                all_chunks.append({
                    'dataset_name': dataset['name'],
                    'dataset_id': dataset_id,
                    'document_id': document_id,
                    'chunk_id': chunk['id'],
                    'chunk_text': chunk['content']
                })

    return all_chunks


def get_response_from_llm(messages: list[dict], model:str, tools: list = None):
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

def qa_generator(docs_chunks: list, num_threads: int = N_THREADS):

    n_samples = len(docs_chunks) if N_GENERATIONS == -1 else N_GENERATIONS
    assert N_GENERATIONS <= len(docs_chunks), f"N_GENERATIONS MUST LOWER THAN THE LENGTH OF chunks {len(docs_chunks)}"
    print(f"Generating {n_samples} QA couples using {num_threads} threads...")

    with multiprocessing.Pool(num_threads) as pool:
        outputs = list(tqdm(pool.imap(partial(_qa_generator_single, ), docs_chunks[:n_samples]), total=n_samples))

    return outputs

def _qa_generator_single(sampled_context):
        # Generate QA couple
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": QA_generation_prompt.format(context=sampled_context['chunk_text'])}
        ]
        output_QA_couple = get_response_from_llm(messages=messages, model=MODEL_NAME)
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Topic: ")[0]
            topic = output_QA_couple.split("Topic: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            return {
                "context": sampled_context['chunk_text'],
                "question": question,
                "answer": answer,
                "topic": topic,
                "source_doc": {"dataset_id": sampled_context["dataset_id"], "document_id": sampled_context["document_id"]}
            }
        except:
            return None


def qa_critic(qas, num_threads: int = N_THREADS):

    print(f"Generating critique for each QA couple using {num_threads} threads...")
    with multiprocessing.Pool(num_threads) as pool:
        qas = list(tqdm(pool.imap(partial(_qa_critic_single, ), qas), total=len(qas)))
    return qas


def _qa_critic_single(output):
    evaluations = {
        "groundedness": get_response_from_llm(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_groundedness_critique_prompt.format(context=output['context'], question=output['question'])}],
            model=MODEL_NAME
        ),
        "relevance": get_response_from_llm(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_relevance_critique_prompt.format(question=output['question'])}],
            model=MODEL_NAME
        ),
        "standalone": get_response_from_llm(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_standalone_critique_prompt.format(question=output['question'])}],
            model=MODEL_NAME
        ),
    }
    try:
        for criterion, evaluation in evaluations.items():
            score, eval = (
                int(evaluation.split("Total rating: ")[-1].strip()),
                evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
            )
            output.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
    except Exception as e:
        pass

    return output


if __name__ == "__main__":
    chunks = get_all_chunks(DATASETS_NAME)
    qas = qa_generator(docs_chunks=chunks)
    qas = qa_critic(qas=qas)


    generated_questions = pd.DataFrame.from_dict(qas)
    # 统计groundedness_score、relevance_score和standalone_score的分布
    print(generated_questions[["groundedness_score", "relevance_score", "standalone_score"]].describe())
    generated_questions = generated_questions.loc[
        (generated_questions["groundedness_score"] >= 4)
        & (generated_questions["relevance_score"] >= 4)
        & (generated_questions["standalone_score"] >= 4)
    ]

    # 创建 Hugging Face 数据集
    dataset_dict = Dataset.from_pandas(generated_questions, split="train", preserve_index=False)

    # 保存数据集
    import os
    dir_name = os.path.dirname(__file__)
    dataset_dict.save_to_disk(os.path.join(dir_name, "eval_rag_dataset"))
    print(f"数据集已保存至本地 {dir_name}/eval_rag_dataset")

    # 如果要发布到 Hugging Face Hub，请取消注释以下行并提供您的用户名和数据集名称
    # dataset_dict.push_to_hub("your-username/your-dataset-name", private=True)
    # print("数据集已保存至 Hugging Face Hub。要发布数据集，请手动更改设置。")
