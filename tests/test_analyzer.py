"""
Test suite for OpenAI analyzer
Location: tests/test_analyzer.py
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from src.openai_analyzer import OpenAIAnalyzer, CodeReviewResult, AnalysisContext


class TestOpenAIAnalyzer:
    """Test OpenAI analyzer functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        with patch('src.openai_analyzer.openai'):
            return OpenAIAnalyzer(
                api_key="test_key",
                model="gpt-4-turbo-preview",
                max_tokens=2000
            )

    def test_token_counting(self, analyzer):
        """Test token counting functionality"""
        text = "This is a test string for counting tokens."
        token_count = analyzer.count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)

    def test_cache_key_generation(self, analyzer):
        """Test cache key generation"""
        file_path = "test.py"
        diff = "def test():\n    pass"

        key1 = analyzer._get_cache_key(file_path, diff)
        key2 = analyzer._get_cache_key(file_path, diff)
        key3 = analyzer._get_cache_key("other.py", diff)

        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key

    @pytest.mark.asyncio
    async def test_analyze_file_diff(self, analyzer):
        """Test file diff analysis"""
        # Mock OpenAI response
        mock_response = json.dumps({
            "reviews": [
                {
                    "severity": "minor",
                    "line_number": 10,
                    "message": "Consider using more descriptive variable names",
                    "suggestion": "Use 'user_count' instead of 'uc'",
                    "confidence": 0.8
                }
            ],
            "summary": "Code looks good with minor suggestions"
        })

        with patch.object(analyzer, '_call_openai',
                         new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            context = AnalysisContext(
                project_name="test_project",
                merge_request_title="Add feature",
                merge_request_description="New feature",
                target_branch="main",
                similar_code_examples=[],
                project_patterns=[],
                recent_reviews=[]
            )

            results = await analyzer.analyze_file_diff(
                "test.py",
                "+def test():\n+    uc = 0\n+    return uc",
                context,
                "python"
            )

            assert len(results) == 1
            assert results[0].severity == "minor"
            assert results[0].line_number == 10
            assert "variable names" in results[0].message

    def test_parse_review_response(self, analyzer):
        """Test parsing of OpenAI responses"""
        response = json.dumps({
            "reviews": [
                {
                    "severity": "critical",
                    "line_number": 5,
                    "message": "SQL injection vulnerability",
                    "suggestion": "Use parameterized queries",
                    "confidence": 0.95
                }
            ]
        })

        results = analyzer._parse_review_response(response, "db.py")

        assert len(results) == 1
        assert results[0].severity == "critical"
        assert results[0].file_path == "db.py"
        assert "SQL injection" in results[0].message

    def test_extract_code_chunks(self, analyzer):
        """Test code chunk extraction from diff"""
        diff = """
@@ -1,5 +1,8 @@
 def existing():
     pass

+def new_function():
+    x = 1
+    y = 2
+    return x + y
+
@@ -10,3 +13,6 @@
 class Existing:
     pass

+class NewClass:
+    def __init__(self):
+        self.value = 0
"""

        chunks = analyzer._extract_code_chunks(diff)

        assert len(chunks) > 0
        assert "new_function" in chunks[0]
        assert any("NewClass" in chunk for chunk in chunks)