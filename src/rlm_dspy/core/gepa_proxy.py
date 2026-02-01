"""
Lightweight proxy module for fast GEPA optimization.

Instead of running the full RLM interpreter loop (10-50 LLM calls per eval),
this proxy uses a simple dspy.Predict that shares the same signature/instructions.

After GEPA evolves the proxy's instructions, they can be transferred back to RLM.
"""

from __future__ import annotations

import logging
from typing import Any

import dspy

logger = logging.getLogger(__name__)


class RLMProxy(dspy.Module):
    """
    Lightweight proxy for RLM that uses dspy.Predict instead of full interpreter loop.
    
    This runs in 1 LLM call instead of 10-50, making GEPA ~50x faster.
    The optimized instructions can then be transferred to the real RLM.
    
    Example:
        # Create proxy from RLM
        proxy = RLMProxy.from_rlm(rlm)
        
        # Run GEPA on proxy (fast!)
        optimized_proxy = gepa.compile(proxy, trainset)
        
        # Transfer instructions back to RLM
        instructions = extract_proxy_instructions(optimized_proxy)
        apply_instructions_to_rlm(rlm, instructions)
    """
    
    def __init__(
        self,
        signature: type[dspy.Signature] | str = "context, query -> answer",
        instructions: str | None = None,
    ):
        super().__init__()
        
        # Build signature with instructions
        if isinstance(signature, str):
            base_sig = dspy.Signature(signature)
        else:
            base_sig = signature
        
        if instructions:
            # Create signature class with docstring (instructions)
            class ProxySignature(base_sig):
                pass
            ProxySignature.__doc__ = instructions
            self.signature = ProxySignature
        else:
            self.signature = base_sig
        
        # Single predict (1 LLM call vs RLM's 10-50)
        self.predict = dspy.Predict(self.signature)
    
    def forward(self, context: str = "", query: str = "") -> dspy.Prediction:
        """Run a single prediction (no interpreter loop)."""
        return self.predict(context=context, query=query)
    
    @classmethod
    def from_rlm(cls, rlm: Any) -> "RLMProxy":
        """Create a proxy from an existing RLM instance.
        
        Extracts the signature and instructions from the RLM to create
        a lightweight equivalent for optimization.
        """
        # Get signature from RLM - prefer the outer signature which has correct instructions
        # DSPy's RLM has signature at module level, not in internal predictors
        signature = "context, query -> answer"  # Default
        instructions = None
        
        # Try to get signature from RLM module
        if hasattr(rlm, "signature"):
            sig = rlm.signature
            signature = sig
            # Get instructions from signature
            if hasattr(sig, "__doc__") and sig.__doc__:
                # Filter out Python's built-in str docstring which sometimes leaks
                doc = sig.__doc__
                if doc and not doc.startswith("str(object"):
                    instructions = doc
            if hasattr(sig, "instructions") and sig.instructions:
                inst = sig.instructions
                if inst and not inst.startswith("str(object"):
                    instructions = inst
        
        # Fallback: try _signature attribute (our custom wrapper)
        if instructions is None:
            if hasattr(rlm, "_signature"):
                sig = rlm._signature
                if hasattr(sig, "__doc__") and sig.__doc__:
                    doc = sig.__doc__
                    if doc and not doc.startswith("str(object"):
                        instructions = doc
        
        return cls(signature=signature, instructions=instructions)
    
    def get_instructions(self) -> str | None:
        """Get the current instructions (signature docstring)."""
        sig = self.predict.signature if hasattr(self.predict, 'signature') else self.signature
        return getattr(sig, '__doc__', None) or getattr(sig, 'instructions', None)


def extract_proxy_instructions(proxy: RLMProxy) -> dict[str, str]:
    """Extract evolved instructions from an optimized proxy.
    
    Returns dict mapping predictor name to instruction text.
    """
    instructions = {}
    
    def is_valid_instruction(text: str | None) -> bool:
        """Check if text is a valid instruction (not Python's str docstring)."""
        if not text:
            return False
        # Filter out Python's built-in str docstring which sometimes leaks through
        if text.startswith("str(object"):
            return False
        # Must have reasonable length
        if len(text) < 20:
            return False
        return True
    
    # Get from predict signature
    if hasattr(proxy, 'predict'):
        pred = proxy.predict
        if hasattr(pred, 'signature'):
            sig = pred.signature
            # Try instructions attribute first (DSPy's preferred location)
            doc = getattr(sig, 'instructions', None)
            if not is_valid_instruction(doc):
                doc = getattr(sig, '__doc__', None)
            
            if is_valid_instruction(doc):
                instructions['generate_action'] = doc
                instructions['predict'] = doc
    
    # Also check extended_signature
    if hasattr(proxy, 'predict') and hasattr(proxy.predict, 'extended_signature'):
        ext_sig = proxy.predict.extended_signature
        doc = getattr(ext_sig, 'instructions', None)
        if not is_valid_instruction(doc):
            doc = getattr(ext_sig, '__doc__', None)
        
        if is_valid_instruction(doc):
            instructions['predict_extended'] = doc
    
    return instructions


def create_proxy_metric(base_metric=None):
    """
    Create a GEPA-compatible metric for proxy optimization.
    
    Uses multi-dimensional scoring inspired by QMD:
    - Completeness (30): Does the answer address the query?
    - Accuracy (30): Does it match expected answer?
    - Specificity (20): Does it reference specific code/files?
    - Format (20): Is it well-structured?
    
    Returns dspy.Prediction with score and feedback.
    """
    from .rlm_reward import score_answer_detailed
    
    def proxy_metric(
        gold: dspy.Example,
        pred: dspy.Prediction | None,
        trace=None,
        pred_name=None, 
        pred_trace=None,
    ) -> dspy.Prediction:
        """Score proxy prediction using multi-dimensional reward."""
        if pred is None:
            return dspy.Prediction(score=0.0, feedback="No prediction generated")
        
        answer = getattr(pred, 'answer', '') or ''
        query = getattr(gold, 'query', '') or ''
        expected = getattr(gold, 'answer', '') or getattr(gold, 'expected_answer', '') or ''
        
        if not answer:
            return dspy.Prediction(score=0.0, feedback="Empty answer")
        
        # Use multi-dimensional scoring
        result = score_answer_detailed(query, answer, expected)
        
        # Build feedback from dimensions and deductions
        feedback_parts = [
            f"comp={result['completeness']}/30",
            f"acc={result['accuracy']}/30",
            f"spec={result['specificity']}/20",
            f"fmt={result['format']}/20",
        ]
        if result['deductions']:
            feedback_parts.append(f"issues: {', '.join(result['deductions'][:2])}")
        
        feedback = f"{result['rating']} ({result['percentage']:.0f}%): {' '.join(feedback_parts)}"
        
        return dspy.Prediction(score=result['score'], feedback=feedback[:200])
    
    return proxy_metric


def run_fast_gepa(
    rlm,
    trainset: list[dspy.Example],
    config=None,
    reflection_lm=None,
) -> tuple[dict[str, str], Any]:
    """
    Run fast GEPA optimization using proxy.
    
    1. Create lightweight proxy from RLM
    2. Run GEPA on proxy (fast - 1 LLM call per eval)
    3. Extract evolved instructions
    4. Return instructions to apply to RLM
    
    Args:
        rlm: The RLM instance (used to create proxy)
        trainset: Training examples
        config: GEPAConfig (optional)
        reflection_lm: LM for GEPA reflection
    
    Returns:
        (instructions_dict, gepa_result)
    """
    from .gepa_optimizer import GEPAConfig, GEPAOptimizer
    
    # Create proxy from RLM
    proxy = RLMProxy.from_rlm(rlm._rlm if hasattr(rlm, '_rlm') else rlm)
    logger.info("Created RLM proxy for fast GEPA")
    
    # Configure GEPA
    if config is None:
        config = GEPAConfig(auto="light", num_threads=2)
    
    # Create optimizer with proxy-specific metric
    optimizer = GEPAOptimizer(
        config=config,
        metric=create_proxy_metric(),
        reflection_lm=reflection_lm,
    )
    
    # Run GEPA on proxy
    lm = rlm._lm if hasattr(rlm, '_lm') else dspy.settings.lm
    dspy.configure(lm=lm)
    
    optimized_proxy, result = optimizer.optimize(
        program=proxy,
        trainset=trainset,
        lm=lm,
    )
    
    # Extract evolved instructions
    instructions = extract_proxy_instructions(optimized_proxy)
    logger.info("Extracted %d evolved instructions from proxy", len(instructions))
    
    return instructions, result
