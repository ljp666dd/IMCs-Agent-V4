QA_generation_prompt = """
Your task is to write a factoid question and a detailed answer given a context.
Your factual question should refer to the information in the context and give a detailed, complete answer.
There are four main topics related to materials science, which are structure, synthesis, properties/properties, and application. You need to first determine which topic the context is biased towards, and your factual question must also focus on that topic.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Topic: (the topic of your factoid question, choose one of four topics: structure, synthesis, properties/properties, and application)
Answer: (your detailed answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""


question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to material science building RAG applications with the LLM.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independant this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independant from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """


# ELO_PROMPT = """###Task Description:
# An instruction (might include an Input inside it), two response to evaluate, a reference answer, and a evaluation criteria are given.
# 1. Write a detailed feedback that vote on both responses strictly based on the given evaluation criteria, not evaluating in general.
# 2. After writing a feedback, vote for a better answer between A and B. You should refer to the evaluation criteria.
# 3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{A or B}}\"
# 4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

# ###The instruction to evaluate:
# {instruction}

# ###Response A to evaluate:
# {response1}

# ###Response B to evaluate:
# {response2}

# ###Reference Answer:
# {reference_answer}

# ###Evaluation criteria:
# [Based on the reference answer, is Answer A more correct, accurate, credible, detailed and truthful, or answer B?]
# A: The response A is more correct, accurate, credible, detailed, and truthful than the response B.
# B: The response B is more correct, accurate, credible, detailed, and truthful than the response A.

# ###Feedback:"""


ELO_PROMPT = """
你需要根据以下要求，对一个问题的两个回答进行公平对比评估。
所有的对比应该首先从科学性和客观性出发，严格基于内容质量判断，而不是回答的顺序或回答格式。
有的回答包含思考过程，有的回答没有包含思考过程，但这并不影响对比的公平性。
其中，回答顺序已通过虚拟随机化处理，需严格基于内容质量判断

### 评估流程
1. 维度隔离评分（满分10,分数范围1-10）：
   <Response A>正确性：_分 | 完整性：_分 | 可信度：_分 
   <Response B>正确性：_分 | 完整性：_分 | 可信度：_分

2. 差异校验（需满足至少两项）：
   ✅ 正确性差异 ≥2分  
   ✅ 完整性差异 ≥1.5分
   ✅ 可信度差异 ≥1分

3. 最终判定条件：
   - 若三个维度均无显著差异 → 输出C
   - 若满足差异校验 → 输出优势方标识符(A/B)
   - 所有结论必须引用参考段落[§编号]验证

### 输入数据
[问题]
{instruction}

[Response A (原始顺序1)]
{response1}

[Response B (原始顺序2)]
{response2}

### 输出规范
反馈格式：Feedback: [正确性对比]...[RESULT]{{A/B/C}}"""


EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 3, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 3):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, credible, detailed, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, incredible, and/or not detailed, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, incredible, and/or not detailed, and/or not factual.
Score 3: The response is somewhat correct, accurate, credible, and/or detailed, and/or factual.
Score 4: The response is mostly correct, accurate, credible, and detailed, and factual.
Score 5: The response is completely correct, accurate, credible, and detailed, and factual.

###Feedback:"""