import os
import time
import json
import copy

from openai import OpenAI

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


def get_es_client():
    elastic_url = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")
    return Elasticsearch(elastic_url)

def elastic_search_text(query, course, index_name="course-questions"):
    es_client = get_es_client()
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields",
                    }
                },
                "filter": {"term": {"course": course}},
            }
        },
    }

    response = es_client.search(index=index_name, body=search_query)
    return [hit["_source"] for hit in response["hits"]["hits"]]


def elastic_search_knn(field, vector, course, index_name="course-questions"):
    es_client = get_es_client()
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {"term": {"course": course}},
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"],
    }

    es_results = es_client.search(index=index_name, body=search_query)

    return [hit["_source"] for hit in es_results["hits"]["hits"]]


def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = "\n\n".join(
        [
            f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}"
            for doc in search_results
        ]
    )
    return prompt_template.format(question=query, context=context).strip()

def get_llm_client(model_choice):
    if model_choice.startswith('ollama/'):
        ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434/v1/")
        return OpenAI(base_url=ollama_url, api_key="ollama")
    elif model_choice.startswith('openai/'):
        openai_api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
        return OpenAI(api_key=openai_api_key)
    elif model_choice.startswith('deepseek/'):
        base_llm_url = os.getenv("BASE_LLM_URL", "your-api-key-here")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
        return OpenAI(base_url=base_llm_url, api_key=deepseek_api_key)
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

def llm(prompt, model_choice):
    start_time = time.time()
    client = get_llm_client(model_choice)
    response = client.chat.completions.create(
        model=model_choice.split('/')[-1],
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    tokens = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }
    end_time = time.time()
    response_time = end_time - start_time
    
    return answer, tokens, response_time

def evaluate_relevance(question, answer, eval_llm_model_choice):
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using json code blocks, I want only a
    clean answer:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, response_time = llm(prompt, eval_llm_model_choice)

    try:
        json_eval = json.loads(evaluation)
        return json_eval['Relevance'], json_eval['Explanation'], tokens, response_time
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", tokens


def calculate_openai_cost(model_choice, tokens):
    openai_cost = 0

    if model_choice == 'openai/gpt-3.5-turbo':
        openai_cost = (tokens['prompt_tokens'] * 0.0015 + tokens['completion_tokens'] * 0.002) / 1000
    elif model_choice in ['openai/gpt-4o', 'openai/gpt-4o-mini']:
        openai_cost = (tokens['prompt_tokens'] * 0.03 + tokens['completion_tokens'] * 0.06) / 1000
    elif model_choice in ['deepseek/deepseek-chat']:
        openai_cost = (tokens['prompt_tokens'] * 0.27 + tokens['completion_tokens'] * 1.1) / 1e6
    elif model_choice in ['deepseek/deepseek-reasoner']:
        openai_cost = (tokens['prompt_tokens'] * 0.55 + tokens['completion_tokens'] * 2.19) / 1e6

    return openai_cost

def get_llm_evaluator_model_name():
    return os.getenv("EVALUATOR_LLM_MODEL_NAME", "your-api-key-here")

def get_answer(query, course, model_choice, search_type):
    if search_type == 'Vector':
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        vector = model.encode(query)
        search_results = elastic_search_knn('question_text_vector', vector, course)
    else:
        search_results = elastic_search_text(query, course)

    prompt = build_prompt(query, search_results)
    answer, tokens, response_time = llm(prompt, model_choice)
    cost = calculate_openai_cost(model_choice, tokens)
    return {
        'answer': answer,
        'response_time': response_time,
        'model_used': model_choice,
        'prompt_tokens': tokens['prompt_tokens'],
        'completion_tokens': tokens['completion_tokens'],
        'total_tokens': tokens['total_tokens'],
        'cost': cost
    }

def evaluate(query, answer):
    evaluator_llm_model_name = get_llm_evaluator_model_name()

    relevance, explanation, eval_tokens, response_time = (
        evaluate_relevance(query, answer, evaluator_llm_model_name)
    )
    cost = calculate_openai_cost(evaluator_llm_model_name, eval_tokens)
    return {
        'relevance': relevance,
        'relevance_explanation': explanation,
        'response_time': response_time,
        'model_used': evaluator_llm_model_name,
        'prompt_tokens': eval_tokens['prompt_tokens'],
        'completion_tokens': eval_tokens['completion_tokens'],
        'total_tokens': eval_tokens['total_tokens'],
        'cost': cost
    }