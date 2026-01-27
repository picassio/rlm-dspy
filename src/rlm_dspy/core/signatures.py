"""DSPy Signatures for RLM operations.

Signatures define the typed input/output contract for LLM operations.
DSPy optimizes prompts automatically based on these signatures.
"""

from typing import Literal

import dspy


class AnalyzeChunk(dspy.Signature):
    """Analyze a chunk of content to answer a query.

    Given a chunk of content and a query, extract relevant information
    that helps answer the query. Be specific and cite evidence.
    """

    query: str = dspy.InputField(desc="The question to answer")
    chunk: str = dspy.InputField(desc="A chunk of content to analyze")
    chunk_index: int = dspy.InputField(desc="Index of this chunk (for context)")
    total_chunks: int = dspy.InputField(desc="Total number of chunks")

    relevant_info: str = dspy.OutputField(desc="Information relevant to the query found in this chunk")
    confidence: Literal["high", "medium", "low", "none"] = dspy.OutputField(
        desc="Confidence that this chunk contains relevant information"
    )


class AggregateAnswers(dspy.Signature):
    """Aggregate partial answers into a final comprehensive answer.

    Given multiple partial answers from different chunks, synthesize them
    into a single coherent final answer.
    """

    query: str = dspy.InputField(desc="The original question")
    partial_answers: list[str] = dspy.InputField(desc="List of partial answers from chunks")

    final_answer: str = dspy.OutputField(desc="Comprehensive answer synthesizing all partial answers")
    sources_used: list[int] = dspy.OutputField(desc="Indices of chunks that contributed to the answer")


class DecomposeTask(dspy.Signature):
    """Decompose a complex task into simpler subtasks.

    Given a complex query and context metadata, break it down into
    smaller, manageable subtasks that can be processed independently.
    """

    query: str = dspy.InputField(desc="The complex query to decompose")
    context_size: int = dspy.InputField(desc="Total size of the context in characters")
    context_type: str = dspy.InputField(desc="Type of context (e.g., 'code', 'docs', 'mixed')")

    strategy: Literal["map_reduce", "iterative", "hierarchical", "direct"] = dspy.OutputField(
        desc="Recommended processing strategy"
    )
    chunk_size: int = dspy.OutputField(desc="Recommended chunk size in characters")
    subtasks: list[str] = dspy.OutputField(desc="List of subtasks to execute")
    reasoning: str = dspy.OutputField(desc="Explanation of the decomposition strategy")


class ExtractAnswer(dspy.Signature):
    """Extract the final answer from REPL execution results.

    Given the history of REPL interactions, extract the definitive final answer.
    """

    query: str = dspy.InputField(desc="The original query")
    execution_history: str = dspy.InputField(desc="History of REPL code executions and outputs")

    final_answer: str = dspy.OutputField(desc="The final answer extracted from the execution")
    answer_type: Literal["direct", "computed", "uncertain", "failed"] = dspy.OutputField(
        desc="How the answer was derived"
    )


class SummarizeSection(dspy.Signature):
    """Summarize a section of content while preserving key details.

    Create a concise summary that retains important information for
    answering potential queries.
    """

    section: str = dspy.InputField(desc="The section content to summarize")
    section_header: str = dspy.InputField(desc="Header or title of the section", default="")
    preserve_keywords: list[str] = dspy.InputField(desc="Keywords that should be preserved in summary", default=[])

    summary: str = dspy.OutputField(desc="Concise summary preserving key information")
    key_entities: list[str] = dspy.OutputField(desc="Important entities mentioned in the section")


class ValidateAnswer(dspy.Signature):
    """Validate an answer against the source content.

    Check if the proposed answer is supported by the evidence in the content.
    """

    query: str = dspy.InputField(desc="The original query")
    proposed_answer: str = dspy.InputField(desc="The answer to validate")
    evidence: str = dspy.InputField(desc="Source content to validate against")

    is_valid: bool = dspy.OutputField(desc="Whether the answer is supported by evidence")
    corrections: str = dspy.OutputField(desc="Suggested corrections if answer is invalid")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
