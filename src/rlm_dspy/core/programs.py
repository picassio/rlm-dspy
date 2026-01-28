"""Precompiled DSPy programs for common RLM patterns.

These programs can be optimized via DSPy's compile() and saved for reuse.
"""

from __future__ import annotations

import dspy

from .signatures import (
    AggregateAnswers,
    AnalyzeChunk,
    ValidateAnswer,
)


class RecursiveAnalyzer(dspy.Module):
    """
    A recursive analyzer that breaks down large contexts hierarchically.

    Uses DSPy's ChainOfThought for better reasoning at each step.
    """

    # Minimum chunk size to prevent infinite recursion (1KB)
    MIN_CHUNK_SIZE = 1000

    def __init__(self, max_depth: int = 3) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.analyzer = dspy.ChainOfThought(AnalyzeChunk)
        self.aggregator = dspy.ChainOfThought(AggregateAnswers)

    def forward(
        self,
        query: str,
        context: str,
        chunk_size: int = 100_000,
        depth: int = 0,
    ) -> dspy.Prediction:
        """
        Recursively analyze context to answer query.

        Args:
            query: The question to answer
            context: The context to analyze
            chunk_size: Size of chunks to process
            depth: Current recursion depth

        Returns:
            Prediction with answer and metadata
        """
        # Base case: context fits in one chunk
        if len(context) <= chunk_size or depth >= self.max_depth:
            result = self.analyzer(
                query=query,
                chunk=context,
                chunk_index=0,
                total_chunks=1,
            )
            return dspy.Prediction(
                answer=result.relevant_info,
                confidence=result.confidence,
                depth=depth,
            )

        # Recursive case: chunk and process
        chunks = self._chunk(context, chunk_size)
        partial_answers = []

        for i, chunk in enumerate(chunks):
            # Recursively analyze each chunk (halve size but respect minimum)
            next_chunk_size = max(self.MIN_CHUNK_SIZE, chunk_size // 2)
            sub_result = self.forward(
                query=query,
                context=chunk,
                chunk_size=next_chunk_size,
                depth=depth + 1,
            )
            if sub_result.confidence.lower() != "none":
                partial_answers.append(sub_result.answer)

        # Aggregate partial answers
        if not partial_answers:
            return dspy.Prediction(
                answer="No relevant information found.",
                confidence="none",
                depth=depth,
            )

        # Rolling aggregation to prevent context overflow
        # Aggregate in batches of MAX_BATCH to stay within token limits
        MAX_BATCH = 5
        while len(partial_answers) > MAX_BATCH:
            # Aggregate in batches, then aggregate the aggregations
            batched_answers = []
            for i in range(0, len(partial_answers), MAX_BATCH):
                batch = partial_answers[i:i + MAX_BATCH]
                if len(batch) == 1:
                    batched_answers.append(batch[0])
                else:
                    batch_result = self.aggregator(
                        query=query,
                        partial_answers=batch,
                    )
                    batched_answers.append(batch_result.final_answer)
            partial_answers = batched_answers

        aggregated = self.aggregator(
            query=query,
            partial_answers=partial_answers,
        )

        return dspy.Prediction(
            answer=aggregated.final_answer,
            confidence="high" if len(partial_answers) > 1 else "medium",
            depth=depth,
            sources=aggregated.sources_used,
        )

    def _chunk(self, text: str, size: int, overlap: int = 500) -> list[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to split
            size: Target chunk size in characters
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
        """
        # Validate inputs to prevent infinite loops / OOM
        if size <= 0:
            size = 100_000  # Default chunk size
        # Ensure overlap is at most half the chunk size to guarantee meaningful progress
        overlap = min(overlap, size // 2)
        # Minimum step size to prevent excessive chunk creation
        min_step = max(size // 4, 1000)  # At least 25% progress or 1000 chars

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            step = max(min_step, size - overlap)  # Ensure meaningful progress
            start += step
            if end >= len(text):
                break
        return chunks


class ChunkedProcessor(dspy.Module):
    """
    Process large contexts by chunking and parallel analysis.

    Simpler than RecursiveAnalyzer - just chunks once and aggregates.
    Good for medium-sized contexts where one level of chunking suffices.
    """

    def __init__(self) -> None:
        super().__init__()
        self.analyzer = dspy.ChainOfThought(AnalyzeChunk)
        self.aggregator = dspy.ChainOfThought(AggregateAnswers)

    def forward(
        self,
        query: str,
        context: str,
        chunk_size: int = 100_000,
    ) -> dspy.Prediction:
        """
        Chunk context, analyze each chunk, aggregate results.

        Args:
            query: The question to answer
            context: The context to analyze
            chunk_size: Size of chunks

        Returns:
            Prediction with answer
        """
        # Split into chunks with safety checks
        chunks = []
        start = 0
        overlap = 500
        # Ensure chunk_size is valid and overlap doesn't cause infinite loop
        if chunk_size <= 0:
            chunk_size = 100_000
        overlap = min(overlap, chunk_size - 1)  # Ensure overlap < chunk_size

        while start < len(context):
            end = min(start + chunk_size, len(context))
            chunks.append(context[start:end])
            step = max(1, chunk_size - overlap)  # Ensure we make progress
            start += step
            if end >= len(context):
                break

        # Analyze each chunk
        partial_answers = []
        chunk_metadata = []

        for i, chunk in enumerate(chunks):
            result = self.analyzer(
                query=query,
                chunk=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
            )
            chunk_metadata.append(
                {
                    "index": i,
                    "confidence": result.confidence,
                }
            )
            if result.confidence.lower() != "none":
                partial_answers.append(result.relevant_info)

        # Aggregate
        if not partial_answers:
            return dspy.Prediction(
                answer="No relevant information found in the provided context.",
                chunks_analyzed=len(chunks),
                chunks_with_info=0,
                chunk_metadata=chunk_metadata,  # Include for debugging
            )

        aggregated = self.aggregator(
            query=query,
            partial_answers=partial_answers,
        )

        return dspy.Prediction(
            answer=aggregated.final_answer,
            chunks_analyzed=len(chunks),
            chunks_with_info=len(partial_answers),
            sources=aggregated.sources_used,
            chunk_metadata=chunk_metadata,  # Include for debugging
        )


class MapReduceProcessor(dspy.Module):
    """
    Classic map-reduce pattern for large-scale analysis.

    Map: Apply analysis to each chunk independently
    Reduce: Aggregate all results into final answer

    Supports custom map and reduce functions.
    """

    def __init__(
        self,
        map_signature: type[dspy.Signature] = AnalyzeChunk,
        reduce_signature: type[dspy.Signature] = AggregateAnswers,
    ) -> None:
        super().__init__()
        self.mapper = dspy.ChainOfThought(map_signature)
        self.reducer = dspy.ChainOfThought(reduce_signature)

    def forward(
        self,
        query: str,
        chunks: list[str],
    ) -> dspy.Prediction:
        """
        Map analysis over chunks, then reduce to final answer.

        Args:
            query: The question to answer
            chunks: Pre-chunked context

        Returns:
            Prediction with final answer
        """
        # Map phase
        mapped_results = []
        for i, chunk in enumerate(chunks):
            result = self.mapper(
                query=query,
                chunk=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
            )
            mapped_results.append(result)

        # Filter relevant results
        partial_answers = [
            r.relevant_info for r in mapped_results 
            if hasattr(r, "confidence") and r.confidence.lower() != "none"
        ]

        if not partial_answers:
            return dspy.Prediction(
                answer="No relevant information found.",
                map_count=len(chunks),
                reduce_input_count=0,
            )

        # Reduce phase
        reduced = self.reducer(
            query=query,
            partial_answers=partial_answers,
        )

        return dspy.Prediction(
            answer=reduced.final_answer,
            map_count=len(chunks),
            reduce_input_count=len(partial_answers),
            sources=reduced.sources_used if hasattr(reduced, "sources_used") else [],
        )


class ValidatedAnalyzer(dspy.Module):
    """
    Analyzer that validates its answers against evidence.

    Two-phase process:
    1. Generate answer from context
    2. Validate answer is supported by evidence
    """

    def __init__(self) -> None:
        super().__init__()
        self.analyzer = ChunkedProcessor()
        self.validator = dspy.ChainOfThought(ValidateAnswer)

    def forward(
        self,
        query: str,
        context: str,
        chunk_size: int = 100_000,
    ) -> dspy.Prediction:
        """
        Analyze and validate the answer.

        Args:
            query: The question
            context: The context
            chunk_size: Chunk size for analysis

        Returns:
            Prediction with validated answer
        """
        # Phase 1: Generate answer
        analysis = self.analyzer(
            query=query,
            context=context,
            chunk_size=chunk_size,
        )

        # Phase 2: Validate
        # Use a representative sample of context for validation
        # Limit to avoid context window overflow on large codebases
        max_evidence_chars = min(chunk_size * 2, 200_000)
        if len(context) > max_evidence_chars:
            # Take beginning + end to capture both setup and conclusions
            half = max_evidence_chars // 2
            evidence = (
                context[:half] 
                + f"\n\n... [{len(context) - max_evidence_chars:,} chars omitted] ...\n\n" 
                + context[-half:]
            )
        else:
            evidence = context
        
        validation = self.validator(
            query=query,
            proposed_answer=analysis.answer,
            evidence=evidence,
        )

        # Return validated or corrected answer
        # Normalize is_valid to boolean (DSPy may return string "True"/"False")
        is_valid = validation.is_valid
        if isinstance(is_valid, str):
            is_valid = is_valid.lower() in ("true", "yes", "1")
        
        if is_valid:
            return dspy.Prediction(
                answer=analysis.answer,
                validated=True,
                confidence=validation.confidence,
            )
        else:
            # Apply corrections if invalid
            return dspy.Prediction(
                answer=validation.corrections if validation.corrections else analysis.answer,
                validated=False,
                original_answer=analysis.answer,
                confidence=validation.confidence,
            )
