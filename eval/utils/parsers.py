import logging
import re
from typing import Iterable, Optional
from lm_eval.tasks.hendrycks_math.utils import remove_boxed, last_boxed_only_string

LOGGER = logging.getLogger(__name__)

_THINK_TAG_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_BOXED_PATTERN = re.compile(r"\\(?:boxed|fbox)\s*\{", re.DOTALL)
_GSM8K_ANSWER_PATTERN = re.compile(r"####\s*(?P<answer>[^\n\r]+)")
_LATEX_TEXT_COMMAND_PATTERN = re.compile(
    r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|operatorname|mbox)\s*\{",
    re.DOTALL,
)
_MATH_DELIMITERS = (
    (r"\\(", r"\\)"),
    (r"\\[", r"\\]"),
    ("$", "$"),
)
_NATURAL_LANGUAGE_PATTERNS = (
    re.compile(
        r"(?:^|\n|\r)\s*(?:therefore|thus|hence|so)?\s*(?:the\s+)?final\s+answer\s*(?:is|=|:)?\s*(?P<answer>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n|\r)\s*answer\s*(?:is|=|:)?\s*(?P<answer>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n|\r)\s*we\s+(?:get|obtain|find)\s*(?P<answer>.+)",
        re.IGNORECASE,
    ),
)
_NUMERIC_FALLBACK_PATTERN = re.compile(
    r"(?P<number>[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?)"
)


def _remove_think_tags(output: str) -> str:
    """Remove reasoning enclosed in <think> tags."""
    if not output:
        return ""

    sanitized = _THINK_TAG_PATTERN.sub(" ", output)
    sanitized = re.sub(r"<think\b[^>]*>.*$", " ", sanitized, flags=re.IGNORECASE | re.DOTALL)
    return sanitized


def _extract_balanced_braced_content(text: str, opening_brace_index: int) -> Optional[str]:
    """Extract the content inside a balanced brace expression."""
    if opening_brace_index < 0 or opening_brace_index >= len(text) or text[opening_brace_index] != "{":
        return None

    depth = 0
    start_index = opening_brace_index + 1

    for index in range(opening_brace_index, len(text)):
        char = text[index]
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start_index:index]

    return None


def _strip_matching_outer_braces(value: str) -> str:
    """Remove one layer of matching outer braces when the entire string is wrapped."""
    candidate = value.strip()
    if len(candidate) < 2 or candidate[0] != "{" or candidate[-1] != "}":
        return candidate

    depth = 0
    for index, char in enumerate(candidate):
        if char == "{" and (index == 0 or candidate[index - 1] != "\\"):
            depth += 1
        elif char == "}" and (index == 0 or candidate[index - 1] != "\\"):
            depth -= 1
            if depth == 0 and index != len(candidate) - 1:
                return candidate

    return candidate[1:-1].strip()


def _unwrap_latex_text_commands(value: str) -> str:
    """Flatten common LaTeX text-style commands while preserving their inner content."""
    result = value
    while True:
        match = _LATEX_TEXT_COMMAND_PATTERN.search(result)
        if match is None:
            break

        brace_index = match.end() - 1
        content = _extract_balanced_braced_content(result, brace_index)
        if content is None:
            break

        command_span_start = match.start()
        command_span_end = brace_index + len(content) + 2
        result = f"{result[:command_span_start]}{content}{result[command_span_end:]}"

    return result


def _strip_math_delimiters(value: str) -> str:
    r"""Remove common outer math delimiters such as $, \( \), and \[ \]."""
    candidate = value.strip()
    changed = True

    while changed:
        changed = False
        for left, right in _MATH_DELIMITERS:
            if candidate.startswith(left) and candidate.endswith(right) and len(candidate) >= len(left) + len(right):
                candidate = candidate[len(left) : len(candidate) - len(right)].strip()
                changed = True

    return candidate


def _cleanup_extracted_answer(answer: str) -> str:
    """Apply lightweight normalization to an extracted answer string."""
    cleaned = (answer or "").strip()
    if not cleaned:
        return ""

    cleaned = _remove_think_tags(cleaned)
    cleaned = _strip_math_delimiters(cleaned)
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = _unwrap_latex_text_commands(cleaned)
    cleaned = _strip_matching_outer_braces(cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.rstrip(" .;,:")
    return cleaned.strip()


def _find_last_match(patterns: Iterable[re.Pattern], text: str) -> str:
    """Return the answer group from the last matching pattern, if any."""
    last_answer = ""
    for pattern in patterns:
        for match in pattern.finditer(text):
            candidate = _cleanup_extracted_answer(match.group("answer"))
            if candidate:
                last_answer = candidate
    return last_answer


def _extract_boxed_answer(output: str) -> str:
    """Extract the last valid LaTeX boxed answer from the output."""
    last_answer = ""
    for match in _BOXED_PATTERN.finditer(output or ""):
        opening_brace_index = match.end() - 1
        content = _extract_balanced_braced_content(output, opening_brace_index)
        if content:
            cleaned = _cleanup_extracted_answer(content)
            if cleaned:
                last_answer = cleaned
    return last_answer


def extract_math_answer(output: str) -> str:
    """Extract math answer using standard lm_eval first, then fallback heuristics."""
    if not isinstance(output, str): 
        return ""
        
    # PRIORITY 1: Canonical Hendrycks Math extraction (Safest approach)
    try:
        ans = remove_boxed(last_boxed_only_string(output))
        if ans: 
            return ans
    except Exception:
        pass

    # PRIORITY 2: Strip <think> tags to avoid extracting intermediate reasoning steps
    output_no_think = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
    if not output_no_think:
        output_no_think = output

    # Heuristic: Catch extremely short, bare outputs (e.g., just "Evelyn" or "65")
    if len(output_no_think) > 0 and len(output_no_think) <= 15:
        return output_no_think.strip('.,; ')

    # PRIORITY 3: Fallback Regex (For DPO/SFT models that ignore \boxed formatting)
    fallback_patterns = [
        r'####\s*\$?([^\n\.\$]+)',
        r'(?i)the final answer(?:\s+is)?\s*:?\s*\$?([^\n\.\$]+)',
        r'(?i)the answer is\s*:?\s*\$?([^\n\.\$]+)',
        r'(?i)therefore,?\s*(?:we have|it is)?\s*\$?([^\n\.\$]+)(?:\s+is the answer|\s+is correct)?',
        
        # --- HEURISTIC CATCH-ALLS ---
        r'(?i)there are\s*\$?([^\s\n\.\$]+)\s*[a-zA-Z]', # Matches formats like "there are 9 elements"
        r'=\s*\$?([^\s\n\.\$]+)\.?\s*$',                 # Matches equations at the very end; \s*$ handles hidden whitespaces
        r'(?i)is\s*\$?([^\s\n\.\$]+)\.?\s*$'             # Matches concise conclusions like "is 4/9."; \s*$ handles hidden whitespaces
    ]
    
    for pattern in fallback_patterns:
        match = re.search(pattern, output_no_think)
        if match:
            ans = match.group(1).strip()
            # Clean up trailing punctuation
            ans = ans.rstrip('.,;')
            # Remove \text{} formatting if present
            ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
            return ans
            
    return ""


def extract_gsm8k_answer(output: str) -> str:
    """Extract a GSM8K answer using the canonical #### marker and robust fallbacks."""
    try:
        last_marker_answer = ""
        for match in _GSM8K_ANSWER_PATTERN.finditer(output or ""):
            candidate = _cleanup_extracted_answer(match.group("answer"))
            if candidate:
                last_marker_answer = candidate
        if last_marker_answer:
            return last_marker_answer

        sanitized_output = _remove_think_tags(output or "")
        fallback_answer = _find_last_match(_NATURAL_LANGUAGE_PATTERNS, sanitized_output)
        if fallback_answer:
            return fallback_answer

        numeric_candidates = [
            _cleanup_extracted_answer(match.group("number"))
            for match in _NUMERIC_FALLBACK_PATTERN.finditer(sanitized_output)
        ]
        numeric_candidates = [candidate for candidate in numeric_candidates if candidate]
        if numeric_candidates:
            return numeric_candidates[-1]

        return ""
    except Exception as exc:
        LOGGER.exception("Failed to extract GSM8K answer: %s", exc)
        return ""
