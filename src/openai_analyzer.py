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
        """Build a comprehensive, framework-aware review prompt"""

        # Detect framework/technology
        framework = self._detect_framework(file_path, diff, language)

        prompt = f"""You are an expert code reviewer specializing in {framework or language or 'general'} development for the {context.project_name} project.

    Project Context:
    - Merge Request: {context.merge_request_title}
    - Description: {context.merge_request_description}
    - Target Branch: {context.target_branch}
    - File: {file_path}
    - Language: {language or 'Unknown'}
    - Framework: {framework or 'Unknown'}

    """

        # Add similar code examples if available
        if similar_code:
            prompt += "\n📚 Similar code patterns from this project:\n"
            for idx, example in enumerate(similar_code[:3], 1):
                prompt += f"\nExample {idx} ({example.get('file', 'unknown')}):\n"
                prompt += f"```\n{example.get('code', '')[:500]}\n```\n"

        # Add project patterns if available
        if context.project_patterns:
            prompt += "\n🎯 Project coding patterns to follow:\n"
            for pattern in context.project_patterns[:5]:
                prompt += f"- {pattern}\n"

        prompt += f"""

    📝 Code Diff to Review:
    ```diff
    {diff}
    ```

    """

        # Add framework-specific review guidelines
        prompt += self._get_framework_specific_guidelines(framework, language)

        prompt += """

    📋 Review Focus Areas:

    **1. Critical Issues (MUST FIX before merge)**
       - Security vulnerabilities (SQL injection, XSS, CSRF, authentication bypass)
       - Data integrity issues (race conditions, data loss, corruption)
       - Breaking changes (API contract violations, backward compatibility)
       - Memory leaks or resource exhaustion
       - Critical performance bottlenecks

    **2. Major Issues (SHOULD FIX)**
       - Logic errors and bugs
       - Significant performance problems (N+1 queries, inefficient algorithms)
       - Poor error handling (unhandled exceptions, silent failures)
       - Design flaws (tight coupling, violation of SOLID principles)
       - Missing validation or sanitization
       - Incorrect async/await usage

    **3. Minor Issues (CONSIDER FIXING)**
       - Code style violations (inconsistent formatting, naming conventions)
       - Missing type hints/annotations
       - Code duplication (DRY violations)
       - Overly complex code (high cyclomatic complexity)
       - Missing or inadequate comments for complex logic
       - Inefficient but functional code

    **4. Suggestions (NICE TO HAVE)**
       - Refactoring opportunities
       - Modern language features that could be used
       - Better abstractions or patterns
       - Additional test coverage areas
       - Documentation improvements
       - Accessibility improvements (for frontend)

    🎯 **Response Format (JSON only):**
    ```json
    {
        "reviews": [
            {
                "severity": "critical|major|minor|suggestion",
                "line_number": <line_number or null>,
                "message": "Clear, actionable description of the issue",
                "code_snippet": "The problematic code excerpt",
                "suggestion": "Specific fix or improvement with code example",
                "confidence": 0.0-1.0,
                "reasoning": "Why this is an issue (optional, for complex cases)"
            }
        ],
        "summary": "Overall assessment focusing on: 1) Most critical concerns, 2) Code quality level, 3) Recommendation (approve/needs work)"
    }
    ```

    ⚠️ **Important Guidelines:**
    - Be specific and actionable - provide exact fixes, not vague advice
    - Reference line numbers when possible
    - Consider the existing codebase patterns shown above
    - Prioritize real bugs and security issues over style preferences
    - Only report actual issues - don't nitpick if code follows project standards
    - Provide code examples in suggestions when helpful
    - Consider performance implications at scale
    - Think about maintainability and future developers
    - For frontend: consider UX, accessibility, and bundle size
    - For backend: consider scalability, security, and data integrity

    Start your analysis now:"""

        return prompt

    def _detect_framework(self, file_path: str, diff: str, language: Optional[str]) -> Optional[str]:
        """Detect framework from file path, diff content, and language"""

        file_lower = file_path.lower()
        diff_lower = diff.lower()

        # PHP/Laravel detection
        if language == 'php' or file_path.endswith('.php'):
            if any(keyword in diff_lower for keyword in ['eloquent', 'illuminate\\', 'artisan', 'facade', 'app(']):
                return 'Laravel'
            if any(keyword in diff_lower for keyword in ['symfony\\', 'doctrine\\', 'container']):
                return 'Symfony'
            return 'PHP'

        # JavaScript/TypeScript framework detection
        if language in ['javascript', 'typescript'] or file_path.endswith(('.js', '.ts', '.jsx', '.tsx', '.vue')):
            if 'nuxt' in file_lower or 'pages/' in file_lower or 'layouts/' in file_lower:
                return 'Nuxt.js'
            if 'next' in file_lower or any(
                    keyword in diff_lower for keyword in ['usenext', 'getserversideprops', 'getstaticprops']):
                return 'Next.js'
            if '.vue' in file_path or 'vue' in diff_lower:
                return 'Vue.js'
            if any(keyword in diff_lower for keyword in ['react', 'usestate', 'useeffect', 'jsx']):
                return 'React'
            return 'JavaScript/TypeScript'

        # Python framework detection
        if language == 'python' or file_path.endswith('.py'):
            if any(keyword in diff_lower for keyword in ['django.', 'from django', 'models.model']):
                return 'Django'
            if any(keyword in diff_lower for keyword in ['flask', '@app.route', 'from flask']):
                return 'Flask'
            if any(keyword in diff_lower for keyword in ['fastapi', '@app.get', '@app.post']):
                return 'FastAPI'
            return 'Python'

        return None

    def _get_framework_specific_guidelines(self, framework: Optional[str], language: Optional[str]) -> str:
        """Get framework-specific review guidelines"""

        if framework == 'Laravel':
            return """
    🔧 **Laravel-Specific Review Checklist:**

    **Security:**
    - ✓ Using Eloquent ORM or query builder (not raw queries)
    - ✓ Input validation with Form Requests or validate()
    - ✓ CSRF protection on forms (@csrf directive)
    - ✓ Mass assignment protection ($fillable/$guarded)
    - ✓ Authorization checks (gates, policies, middleware)
    - ✓ SQL injection prevention (parameterized queries)
    - ✓ XSS prevention (blade {{ }} escaping)

    **Best Practices:**
    - ✓ Following PSR standards (PSR-12 for code style)
    - ✓ Proper use of service containers and dependency injection
    - ✓ Route model binding when applicable
    - ✓ Eloquent relationships properly defined
    - ✓ Avoiding N+1 queries (use eager loading with())
    - ✓ Using collections efficiently (avoid loading all records)
    - ✓ Proper transaction handling for data integrity
    - ✓ Queue long-running tasks (don't block HTTP requests)
    - ✓ Using Laravel helpers (old(), request(), auth(), etc.)
    - ✓ Proper error handling (try-catch, custom exceptions)

    **Performance:**
    - ✓ Database indexing on queried columns
    - ✓ Caching frequently accessed data (cache() facade)
    - ✓ Query optimization (select specific columns, chunking)
    - ✓ Avoiding memory leaks (large collections, file handles)
    - ✓ Using database transactions appropriately

    **Code Organization:**
    - ✓ Controllers are thin (business logic in services/actions)
    - ✓ Models contain only data logic
    - ✓ Proper use of Form Requests for validation
    - ✓ Resources/Transformers for API responses
    - ✓ Jobs for background processing
    - ✓ Events and Listeners for decoupled logic

    **Common Laravel Pitfalls to Check:**
    - ⚠️ Not using DB transactions for multi-step operations
    - ⚠️ Raw SQL without parameter binding
    - ⚠️ Using select * instead of specific columns
    - ⚠️ Not eager loading relationships (N+1 problem)
    - ⚠️ Missing validation on user input
    - ⚠️ Storing sensitive data in plain text
    - ⚠️ Not using rate limiting on sensitive endpoints
    - ⚠️ Forgetting to check authorization before actions
    """

        elif framework == 'Nuxt.js':
            return """
    🔧 **Nuxt.js-Specific Review Checklist:**

    **Security:**
    - ✓ Input sanitization (especially for v-html)
    - ✓ CSRF protection on API calls
    - ✓ Secure cookie configuration (httpOnly, secure, sameSite)
    - ✓ Environment variables not exposed to client
    - ✓ API keys and secrets in server-side only
    - ✓ Content Security Policy (CSP) headers configured
    - ✓ XSS prevention (avoid v-html with user input)

    **Performance & SEO:**
    - ✓ Proper use of asyncData vs fetch vs created
    - ✓ Static generation (SSG) where possible
    - ✓ Dynamic imports for heavy components
    - ✓ Image optimization (nuxt/image module)
    - ✓ Meta tags for SEO (useHead, useSeoMeta)
    - ✓ Lazy loading components and images
    - ✓ Avoiding unnecessary re-renders
    - ✓ Using computed properties instead of methods in templates
    - ✓ Proper error handling (error.vue, try-catch)

    **Nuxt 3 Best Practices:**
    - ✓ Using Composition API (setup, ref, reactive)
    - ✓ Auto-imports (no manual component imports needed)
    - ✓ Proper use of composables (useState, useFetch, useAsyncData)
    - ✓ Server routes in /server/api for backend logic
    - ✓ Middleware for route protection
    - ✓ Proper TypeScript usage (type safety)
    - ✓ Using Pinia for state management (not Vuex)

    **Common Nuxt Pitfalls:**
    - ⚠️ Accessing window/document in SSR context
    - ⚠️ Not handling async data errors
    - ⚠️ Memory leaks from event listeners not cleaned up
    - ⚠️ Large bundle sizes (check imports)
    - ⚠️ Using fetch in components without error handling
    - ⚠️ Not using key attribute in v-for loops
    - ⚠️ Mutating props directly
    - ⚠️ Missing loading states for async operations
    - ⚠️ Not optimizing images (large file sizes)

    **Accessibility (a11y):**
    - ✓ Semantic HTML elements
    - ✓ ARIA labels where needed
    - ✓ Keyboard navigation support
    - ✓ Focus management
    - ✓ Alt text for images
    - ✓ Color contrast ratios
    """

        elif framework == 'Vue.js':
            return """
    🔧 **Vue.js-Specific Review Checklist:**

    **Best Practices:**
    - ✓ Proper component naming (PascalCase)
    - ✓ Props validation with types
    - ✓ Emitting events instead of mutating props
    - ✓ Using computed properties for derived state
    - ✓ Proper lifecycle hook usage
    - ✓ Key attribute in v-for
    - ✓ Avoiding v-if with v-for on same element
    - ✓ Using $emit for child-to-parent communication

    **Performance:**
    - ✓ Lazy loading components
    - ✓ Using v-show vs v-if appropriately
    - ✓ Avoiding unnecessary watchers
    - ✓ Functional components for simple presentational components
    - ✓ Virtual scrolling for large lists
    """

        elif framework == 'React':
            return """
    🔧 **React-Specific Review Checklist:**

    **Best Practices:**
    - ✓ Proper hook usage (rules of hooks)
    - ✓ Dependency arrays in useEffect
    - ✓ Key prop in lists
    - ✓ Memoization (useMemo, useCallback) where appropriate
    - ✓ Error boundaries for error handling
    - ✓ Avoiding prop drilling (Context, composition)
    - ✓ Proper TypeScript types for props

    **Performance:**
    - ✓ React.memo for expensive components
    - ✓ Code splitting with lazy and Suspense
    - ✓ Avoiding unnecessary re-renders
    """

        elif framework in ['PHP', 'Symfony']:
            return """
    🔧 **PHP-Specific Review Checklist:**

    **Security:**
    - ✓ Parameterized queries (no string concatenation in SQL)
    - ✓ Input validation and sanitization
    - ✓ Password hashing (password_hash/password_verify)
    - ✓ CSRF protection
    - ✓ XSS prevention (htmlspecialchars, ENT_QUOTES)

    **Best Practices:**
    - ✓ PSR standards compliance (PSR-1, PSR-12)
    - ✓ Type declarations (strict_types=1)
    - ✓ Proper error handling (exceptions, not error suppression)
    - ✓ Dependency injection
    - ✓ Using modern PHP features (null coalescing, spread operator)
    """

        elif framework == 'Python':
            return """
    🔧 **Python-Specific Review Checklist:**

    **Best Practices:**
    - ✓ PEP 8 compliance (formatting, naming)
    - ✓ Type hints for function parameters and returns
    - ✓ Docstrings for classes and functions
    - ✓ Context managers for resource handling (with)
    - ✓ List comprehensions instead of loops (where readable)
    - ✓ Using enumerate and zip appropriately
    - ✓ Exception handling (specific exceptions, not bare except)
    """

        else:
            return """
    🔧 **General Code Review Checklist:**

    **Security:**
    - ✓ Input validation and sanitization
    - ✓ Proper authentication and authorization
    - ✓ Secure data storage and transmission
    - ✓ No hardcoded secrets or credentials

    **Best Practices:**
    - ✓ Clear, descriptive naming
    - ✓ Single Responsibility Principle
    - ✓ DRY (Don't Repeat Yourself)
    - ✓ Proper error handling
    - ✓ Comprehensive comments for complex logic
    - ✓ Efficient algorithms and data structures
    """

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
