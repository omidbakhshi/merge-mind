import openai
import logging
import json
import tiktoken
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, cast
from dataclasses import dataclass
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from src.circuit_breaker import get_openai_circuit_breaker, CircuitBreakerOpenException

logger = logging.getLogger(__name__)


@dataclass
class CodeReviewResult:
    """Result of a code review analysis"""

    file_path: str
    severity: str  # critical, major, minor, suggestion
    line_number: Optional[int]
    message: str
    code_snippet: Optional[str]
    suggestion: Optional[str]
    confidence: float  # 0.0 to 1.0


@dataclass
class AnalysisContext:
    """Context information for code analysis"""

    project_name: str
    merge_request_title: str
    merge_request_description: str
    target_branch: str
    similar_code_examples: List[Dict[str, Any]]  # From vector store
    project_patterns: List[str]  # Learned patterns from the project
    recent_reviews: List[str]  # Recent review comments for consistency


class OpenAIAnalyzer:
    """Analyzes code changes using OpenAI API with learned context"""

    # Token limits for different models
    MODEL_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16384,
        "gpt-5": 150000,
    }

    # Review focus areas
    REVIEW_ASPECTS = [
        "code_quality",
        "bugs",
        "security",
        "performance",
        "maintainability",
        "best_practices",
        "documentation",
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        vector_store=None,
        max_tokens: int = 2000,
    ):
        """Initialize the analyzer

        Args:
            api_key: OpenAI API key
            model: Model to use for analysis
            vector_store: Vector store instance for retrieving similar code
            max_tokens: Maximum tokens for response
        """
        self.api_key = api_key
        self.model = model
        self.vector_store = vector_store
        self.max_tokens = max_tokens

        openai.api_key = api_key
        self.encoding = tiktoken.encoding_for_model(model)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.review_cache: Dict[str, List[CodeReviewResult]] = {}

    def update_model(self, new_model: str) -> None:
        """Update the model used for analysis

        Args:
            new_model: New OpenAI model to use
        """
        logger.info(f"Updating OpenAI model from {self.model} to {new_model}")
        self.model = new_model
        self.encoding = tiktoken.encoding_for_model(new_model)
        logger.info(f"OpenAI model updated successfully to {new_model}")

    def update_api_key(self, new_api_key: str) -> None:
        """Update the OpenAI API key

        Args:
            new_api_key: New OpenAI API key
        """
        logger.info("Updating OpenAI API key")
        self.api_key = new_api_key
        openai.api_key = new_api_key
        logger.info("OpenAI API key updated successfully")

        # Performance optimizations
        self._embedding_cache: Dict[str, List[float]] = {}
        self._embedding_cache_max_size = 500
        self._concurrent_requests_semaphore = asyncio.Semaphore(3)  # Limit concurrent OpenAI calls

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with caching and batching"""
        async with self._concurrent_requests_semaphore:
            # Check cache first
            cached_embeddings = []
            texts_to_embed = []
            cache_indices = []

            for i, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in self._embedding_cache:
                    cached_embeddings.append((i, self._embedding_cache[text_hash]))
                else:
                    texts_to_embed.append(text)
                    cache_indices.append(i)

            # Get embeddings for uncached texts
            if texts_to_embed:
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: openai.Embedding.create(
                            input=texts_to_embed,
                            model="text-embedding-ada-002"
                        )
                    )

                    # Cache new embeddings
                    for text, embedding_data in zip(texts_to_embed, response['data']):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        embedding = embedding_data['embedding']

                        if len(self._embedding_cache) < self._embedding_cache_max_size:
                            self._embedding_cache[text_hash] = embedding
                        else:
                            # Remove oldest (FIFO)
                            oldest_key = next(iter(self._embedding_cache))
                            del self._embedding_cache[oldest_key]
                            self._embedding_cache[text_hash] = embedding

                        cached_embeddings.append((cache_indices[len(cached_embeddings) - len(texts_to_embed)], embedding))

                except Exception as e:
                    logger.error(f"Failed to get batch embeddings: {e}")
                    raise

            # Sort by original order and return
            cached_embeddings.sort(key=lambda x: x[0])
            return [emb for _, emb in cached_embeddings]

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

    def _get_cache_key(self, file_path: str, diff: str) -> str:
        """Generate cache key for a review"""
        content = f"{file_path}:{diff}"
        return hashlib.md5(content.encode()).hexdigest()

    async def analyze_file_diff(
        self,
        file_path: str,
        diff: str,
        context: AnalysisContext,
        language: Optional[str] = None,
    ) -> List[CodeReviewResult]:
        """Analyze a single file diff

        Args:
            file_path: Path to the file
            diff: Git diff content
            context: Analysis context with project information
            language: Programming language
        """
        start_time = datetime.now()
        logger.info(f"Starting analysis of {file_path} ({len(diff)} chars)")

        # Check cache
        cache_key = self._get_cache_key(file_path, diff)
        if cache_key in self.review_cache:
            logger.info(f"Using cached review for {file_path}")
            return self.review_cache[cache_key]

        # Retrieve similar code from vector store if available
        similar_code = []
        if self.vector_store:
            logger.debug(f"Retrieving similar code for {file_path}")
            similar_code = await self._get_similar_code(diff, context.project_name)
            logger.debug(f"Found {len(similar_code)} similar code examples")

        # Build the review prompt
        logger.debug(f"Building review prompt for {file_path}")
        prompt = self._build_review_prompt(
            file_path=file_path,
            diff=diff,
            context=context,
            similar_code=similar_code,
            language=language,
        )

        # Check token limits and truncate if necessary
        prompt_tokens = self.count_tokens(prompt)
        available_tokens = (
            self.MODEL_LIMITS.get(self.model, 8192) - self.max_tokens - 500
        )

        if prompt_tokens > available_tokens:
            logger.warning(f"Prompt too long ({prompt_tokens} tokens, available: {available_tokens}), truncating diff")
            diff = self._truncate_to_token_limit(diff, available_tokens - 2000)
            prompt = self._build_review_prompt(
                file_path=file_path,
                diff=diff,
                context=context,
                similar_code=similar_code[:2],  # Limit similar examples
                language=language,
            )
            logger.debug(f"Diff truncated, new prompt tokens: {self.count_tokens(prompt)}")

        # Call OpenAI API
        try:
            logger.debug(f"Calling OpenAI API for {file_path} with model {self.model}")
            response = await self._call_openai(prompt)
            results = self._parse_review_response(response, file_path)

            # Cache the results
            self.review_cache[cache_key] = results

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Completed analysis of {file_path} in {duration:.2f}s: {len(results)} issues found")

            return results

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to analyze {file_path} after {duration:.2f}s: {e}", exc_info=True)
            return []

    def _build_review_prompt(
        self,
        file_path: str,
        diff: str,
        context: AnalysisContext,
        similar_code: List[Dict],
        language: Optional[str],
    ) -> str:
        """Build a comprehensive review prompt"""

        prompt = f"""You are an expert code reviewer for the {context.project_name} project.

Project Context:
- Merge Request: {context.merge_request_title}
- Description: {context.merge_request_description}
- Target Branch: {context.target_branch}

File: {file_path}
Language: {language or 'Unknown'}

"""

        # Add similar code examples if available
        if similar_code:
            prompt += "\nSimilar code patterns from this project:\n"
            for idx, example in enumerate(similar_code[:3], 1):
                prompt += f"\nExample {idx} ({example.get('file', 'unknown')}):\n"
                prompt += f"```\n{example.get('code', '')[:500]}\n```\n"

        # Add project patterns if available
        if context.project_patterns:
            prompt += "\nProject coding patterns to follow:\n"
            for pattern in context.project_patterns[:5]:
                prompt += f"- {pattern}\n"

        prompt += f"""

Code Diff to Review:
```diff
{diff}
```

Please provide a detailed code review focusing on:
1. Critical issues (bugs, security vulnerabilities)
2. Major issues (performance problems, design flaws)
3. Minor issues (code style, naming conventions)
4. Suggestions for improvement

For each issue found, provide a JSON response with the following structure:
{{
    "reviews": [
        {{
            "severity": "critical|major|minor|suggestion",
            "line_number": null or line number,
            "message": "Clear description of the issue",
            "code_snippet": "The problematic code if applicable",
            "suggestion": "How to fix it",
            "confidence": 0.0 to 1.0
        }}
    ],
    "summary": "Overall assessment of the changes"
}}

Be specific and actionable. Reference line numbers where possible.
Consider the existing codebase patterns and maintain consistency.
Only report actual issues, not stylistic preferences unless they violate project standards.
"""

        return prompt

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with circuit breaker protection"""
        circuit_breaker = get_openai_circuit_breaker()

        try:
            logger.debug("Calling OpenAI API through circuit breaker")

            # Use circuit breaker to protect the API call
            response = await circuit_breaker.call(
                self._call_openai_direct,
                prompt
            )

            return response

        except CircuitBreakerOpenException as e:
            logger.error(f"OpenAI circuit breaker is OPEN: {e}")
            raise Exception("OpenAI service is temporarily unavailable. Please try again later.")

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    async def _call_openai_direct(self, prompt: str) -> str:
        """Direct OpenAI API call with retry logic"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert code reviewer providing detailed, actionable feedback.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=self.max_tokens,
                        temperature=0.3,  # Lower temperature for more consistent reviews
                        response_format={"type": "json_object"},  # Force JSON response
                    ),
                )

                return response.choices[0].message.content

            except openai.error.RateLimitError as _e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == max_retries - 1:
                    raise
                continue

        # This should never be reached, but mypy needs it
        raise RuntimeError("Failed to get response from OpenAI after all retries")

    def _parse_review_response(
        self, response: str, file_path: str
    ) -> List[CodeReviewResult]:
        """Parse OpenAI response into structured review results"""
        results = []

        try:
            # Parse JSON response
            data = json.loads(response)
            reviews = data.get("reviews", [])

            for review in reviews:
                # Extract line number from message if not provided
                line_number = review.get("line_number")
                if not line_number:
                    line_match = re.search(
                        r"line (\d+)", review.get("message", ""), re.IGNORECASE
                    )
                    if line_match:
                        line_number = int(line_match.group(1))

                result = CodeReviewResult(
                    file_path=file_path,
                    severity=review.get("severity", "suggestion"),
                    line_number=line_number,
                    message=review.get("message", ""),
                    code_snippet=review.get("code_snippet"),
                    suggestion=review.get("suggestion"),
                    confidence=float(review.get("confidence", 0.7)),
                )

                results.append(result)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")

            # Fallback: Try to extract insights from non-JSON response
            if "critical" in response.lower() or "bug" in response.lower():
                results.append(
                    CodeReviewResult(
                        file_path=file_path,
                        severity="major",
                        line_number=None,
                        message="Review found potential issues. Please review the diff carefully.",
                        code_snippet=None,
                        suggestion=response[:500],
                        confidence=0.5,
                    )
                )

        return results

    async def _get_similar_code(self, diff: str, project_name: str) -> List[Dict]:
        """Retrieve similar code from vector store"""
        if not self.vector_store:
            return []

        try:
            # Extract meaningful code chunks from diff
            code_chunks = self._extract_code_chunks(diff)

            similar_examples = []
            for chunk in code_chunks[:3]:  # Limit to top 3 chunks
                results = await self.vector_store.search_similar(
                    query=chunk, collection_name=f"gitlab_code_{project_name}", limit=2
                )
                similar_examples.extend(results)

            return similar_examples

        except Exception as e:
            logger.error(f"Failed to get similar code: {e}")
            return []

    def _extract_code_chunks(self, diff: str) -> List[str]:
        """Extract meaningful code chunks from diff"""
        chunks = []
        current_chunk = []

        for line in diff.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                # Added line
                current_chunk.append(line[1:])
            elif line.startswith("@@"):
                # New hunk, save current chunk
                if len(current_chunk) > 0:
                    chunks.append("\n".join(current_chunk))
                current_chunk = []

        # Save last chunk
        if len(current_chunk) > 0:
            chunks.append("\n".join(current_chunk))

        return chunks

    async def analyze_merge_request(
        self,
        file_diffs: List[Tuple[str, str, Optional[str]]],
        context: AnalysisContext,
        batch_size: int = 5,
    ) -> Dict[str, Any]:
        """Analyze an entire merge request

        Args:
            file_diffs: List of (file_path, diff, language) tuples
            context: Analysis context
            batch_size: Number of files to analyze concurrently
        """
        all_results = []

        # Process files in batches
        for i in range(0, len(file_diffs), batch_size):
            batch = file_diffs[i : i + batch_size]

            tasks = [
                self.analyze_file_diff(file_path, diff, context, language)
                for file_path, diff, language in batch
            ]

            batch_results = await asyncio.gather(*tasks)
            all_results.extend([item for sublist in batch_results for item in sublist])

        # Generate summary
        summary = self._generate_review_summary(all_results, len(file_diffs))

        return {"results": all_results, "summary": summary}

    def _generate_review_summary(
        self, results: List[CodeReviewResult], total_files: int
    ) -> Dict[str, Any]:
        """Generate a summary of the review"""
        summary = {
            "statistics": {
                "files_reviewed": total_files,
                "total_issues": len(results),
                "critical": 0,
                "major": 0,
                "minor": 0,
                "suggestions": 0,
            },
            "findings": {"critical": [], "major": [], "minor": [], "suggestions": []},
            "status": "approved",
        }

        statistics = cast(Dict[str, int], summary["statistics"])
        findings = cast(Dict[str, List[str]], summary["findings"])

        for result in results:
            severity = result.severity.lower()
            if severity in statistics:
                statistics[severity] += 1
            else:
                statistics["suggestions"] += 1

            finding = f"{result.file_path}: {result.message}"
            if result.line_number:
                finding = f"{result.file_path}:{result.line_number} - {result.message}"

            if severity in findings:
                findings[severity].append(finding)

        # Determine overall status
        if statistics["critical"] > 0:
            summary["status"] = "needs_work"
            summary[
                "overall_assessment"
            ] = "Critical issues found that must be addressed before merging."
        elif statistics["major"] > 2:
            summary["status"] = "needs_work"
            summary[
                "overall_assessment"
            ] = "Multiple major issues found. Please review and address them."
        elif statistics["major"] > 0:
            summary["status"] = "conditional"
            summary[
                "overall_assessment"
            ] = "Some issues found but can be merged with caution."
        else:
            summary[
                "overall_assessment"
            ] = "Code looks good overall with only minor suggestions."

        return summary
