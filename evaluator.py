import ollama
from pydantic import BaseModel
from typing import Annotated


class Evaluator:
    """
    Evaluator with different evaluation functions.
    """

    @staticmethod
    def relevance(inputs, outputs):
        """
        Relevance: Response vs input, compare the answer to the input question
        Goal: Measure "how well does the generated response address the initial user input"
        """

        # Grade prompt
        relevance_instructions = """You are a teacher grading a quiz. 

        You will be given a QUESTION and a STUDENT ANSWER. 

        Here is the grade criteria to follow:
        (1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
        (2) Ensure the STUDENT ANSWER helps to answer the QUESTION

        Explanation:
        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
        Avoid simply stating the correct answer at the outset.

        Relevance:
        A relevance value of True means that the student's answer meets all of the criteria.
        A relevance value of False means that the student's answer does not meet all of the criteria.
        """
        answer = f"QUESTION: {inputs}\nSTUDENT ANSWER: {outputs}"
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": relevance_instructions},
                {"role": "user", "content": answer},
            ],
            format=RelevanceGrade.model_json_schema(),
        )
        grade = RelevanceGrade.model_validate_json(
            response.message.content, strict=True
        )
        return grade

    @staticmethod
    def groundedness(output, retrieved_docs):
        """
        Groundedness: Response vs retrieved docs, compare the answer to the retrieved context
        Goal: Measure "to what extent does the generated response agree with the retrieved context"
        """

        # Grade prompt
        grounded_instructions = """You are a teacher grading a quiz. 

        You will be given FACTS and a STUDENT ANSWER. 

        Here is the grade criteria to follow:
        (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
        (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

        Grounded:
        A grounded value of True means that the student's answer meets all of the criteria.
        A grounded value of False means that the student's answer does not meet all of the criteria.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""
        doc_string = "\n\n".join(doc for doc in retrieved_docs)
        answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {output}"
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": grounded_instructions},
                {"role": "user", "content": answer},
            ],
            format=GroundedGrade.model_json_schema(),
        )
        grade = GroundedGrade.model_validate_json(response.message.content, strict=True)
        return grade

    @staticmethod
    def retrieval_relevance(input, retrieved_docs):
        """
        Retrieval relevance: Retrieved docs vs input, compare the question to the retrieved context
        Goal: Measure "how relevant are my retrieved results for this query"
        """
        # Grade prompt
        retrieval_relevance_instructions = """You are a teacher grading a quiz. 

        You will be given a QUESTION and a set of FACTS provided by the student. 

        Here is the grade criteria to follow:
        (1) You goal is to identify FACTS that are completely unrelated to the QUESTION
        (2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
        (3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

        Relevance:
        A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
        A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

        Avoid simply stating the correct answer at the outset."""

        doc_string = "\n\n".join(doc for doc in retrieved_docs)
        answer = f"FACTS: {doc_string}\nQUESTION: {input}"

        # Run evaluator
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": retrieval_relevance_instructions},
                {"role": "user", "content": answer},
            ],
            format=GroundedGrade.model_json_schema(),
        )
        grade = GroundedGrade.model_validate_json(response.message.content, strict=True)
        return grade

    @staticmethod
    def correctness(outputs, answer):
        """
        Correctness: Response vs reference answer. Requires a ground truth answer to be supplied
        Goal: Measure "how correct is the RAG chain answer, relative to a ground-truth answer"
        """
        raise NotImplementedError("Correctness evaluation is not implemented")


class RelevanceGrade(BaseModel):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool, ..., "Provide the score on whether the answer addresses the question"
    ]


class GroundedGrade(BaseModel):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[
        bool, ..., "Provide the score on if the answer hallucinates from the documents"
    ]


class RetrievalRelevanceGrade(BaseModel):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[
        bool,
        ...,
        "True if the retrieved documents are relevant to the question, False otherwise",
    ]
