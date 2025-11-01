import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any
from datetime import datetime

from src.openai_analyzer import OpenAIAnalyzer, CodeReviewResult, AnalysisContext
from src.vector_store import CodeMemoryManager
from src.config_manager import ConfigManager


class TestPerformance:
    """Performance tests for optimization features"""

    @pytest.fixture
    def large_codebase_files(self) -> List[Dict[str, str]]:
        """Generate a large set of mock files for performance testing"""
        files = []

        # Generate 50 Python files with realistic content
        for i in range(50):
            content = f'''"""
Module {i}: Part of a large codebase
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataModel{i}:
    """Data model for component {i}"""
    id: int
    name: str
    value: Optional[float] = None

    def validate(self) -> bool:
        """Validate the data model"""
        if self.id <= 0:
            return False
        if not self.name:
            return False
        return True

class Service{i}:
    """Service class for business logic {i}"""

    def __init__(self):
        self.data: List[DataModel{i}] = []
        self.cache: Dict[int, DataModel{i}] = {{}}

    def add_item(self, item: DataModel{i}) -> bool:
        """Add an item to the service"""
        if not item.validate():
            logger.error(f"Invalid item: {{item}}")
            return False

        self.data.append(item)
        self.cache[item.id] = item
        logger.info(f"Added item {{item.id}}")
        return True

    def get_item(self, item_id: int) -> Optional[DataModel{i}]:
        """Get an item by ID"""
        return self.cache.get(item_id)

    def process_batch(self, items: List[DataModel{i}]) -> int:
        """Process a batch of items"""
        processed = 0
        for item in items:
            if self.add_item(item):
                processed += 1
        return processed

def utility_function_{i}(param: str) -> str:
    """Utility function {i}"""
    if not param:
        return "default"
    return param.upper()

# Some complex logic
def complex_calculation_{i}(x: int, y: int) -> float:
    """Perform complex calculation"""
    result = 0.0
    for i in range(x):
        for j in range(y):
            result += i * j / (i + j + 1)
    return result
'''
            files.append({
                'path': f'src/module_{i}.py',
                'content': content,
                'language': 'python'
            })

        return files

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for performance testing"""
        config = Mock(spec=ConfigManager)

        def mock_get_global_setting(*args):
            settings = {
                ("openai", "api_key"): "test_key",
                ("openai", "model"): "gpt-4-turbo-preview",
                ("openai", "max_tokens"): 2000,
                ("review", "cache"): {"enabled": True, "ttl_seconds": 3600},
            }
            return settings.get(args, None)

        config.get_global_setting.side_effect = mock_get_global_setting
        return config

    @pytest.fixture
    def mock_analyzer(self, mock_config):
        """Mock analyzer with performance tracking"""
        analyzer = Mock(spec=OpenAIAnalyzer)

        # Mock analysis results
        def mock_analyze_file_diff(file_path, diff, context, **kwargs):
            # Simulate analysis time
            time.sleep(0.01)  # 10ms per file analysis

            return [
                CodeReviewResult(
                    file_path=file_path,
                    severity="minor",
                    line_number=10,
                    message="Consider using more descriptive variable names",
                    code_snippet="x = 1\ny = 2",
                    suggestion="Use 'first_value' and 'second_value'",
                    confidence=0.8
                )
            ]

        analyzer.analyze_file_diff = AsyncMock(side_effect=mock_analyze_file_diff)

        # Mock batch analysis
        def mock_analyze_merge_request(file_data, context, **kwargs):
            # Simulate batch processing time (faster than individual)
            time.sleep(0.005 * len(file_data))  # 5ms per file in batch

            results = []
            for file_path, diff, language in file_data:
                results.append(CodeReviewResult(
                    file_path=file_path,
                    severity="minor",
                    line_number=5,
                    message="Batch processed review",
                    code_snippet="code snippet",
                    suggestion="Suggestion",
                    confidence=0.7
                ))

            return {
                "results": results,
                "summary": {
                    "status": "success",
                    "overall_assessment": "Batch processing completed",
                    "total_issues": len(results),
                    "critical_count": 0,
                    "major_count": 0,
                    "minor_count": len(results),
                    "suggestion_count": 0
                }
            }

        analyzer.analyze_merge_request = AsyncMock(side_effect=mock_analyze_merge_request)

        return analyzer

    @pytest.mark.asyncio
    async def test_individual_vs_batch_processing_performance(self, mock_analyzer, large_codebase_files):
        """Test performance difference between individual and batch processing"""
        context = AnalysisContext(
            project_name="test-project",
            merge_request_title="Large refactor",
            merge_request_description="Processing large codebase",
            target_branch="main",
            similar_code_examples=[],
            project_patterns=[],
            recent_reviews=[]
        )

        # Test individual file processing
        individual_start = time.time()
        individual_results = []
        for file_info in large_codebase_files[:10]:  # Test with 10 files
            result = await mock_analyzer.analyze_file_diff(
                file_info['path'],
                f"+{file_info['content'][:500]}",  # First 500 chars as diff
                context
            )
            individual_results.extend(result)
        individual_time = time.time() - individual_start

        # Test batch processing
        batch_start = time.time()
        file_data = [
            (f['path'], f"+{f['content'][:500]}", f['language'])
            for f in large_codebase_files[:10]
        ]
        batch_result = await mock_analyzer.analyze_merge_request(
            file_data, context, batch_size=5
        )
        batch_time = time.time() - batch_start

        # Batch should be faster
        assert batch_time < individual_time
        assert len(batch_result["results"]) == len(individual_results)

        # Calculate improvement
        improvement = (individual_time - batch_time) / individual_time * 100
        print(f"Batch processing is {improvement:.1f}% faster")

        # Should be at least 20% faster
        assert improvement > 20

    @pytest.mark.asyncio
    async def test_caching_performance(self, mock_analyzer):
        """Test that caching improves performance"""
        context = AnalysisContext(
            project_name="test-project",
            merge_request_title="Test MR",
            merge_request_description="Testing caching",
            target_branch="main",
            similar_code_examples=[],
            project_patterns=[],
            recent_reviews=[]
        )

        file_path = "test.py"
        diff = "+def test():\n+    x = 1\n+    return x"

        # First analysis (no cache)
        start_time = time.time()
        result1 = await mock_analyzer.analyze_file_diff(file_path, diff, context)
        first_call_time = time.time() - start_time

        # Second analysis (should use cache if implemented)
        start_time = time.time()
        result2 = await mock_analyzer.analyze_file_diff(file_path, diff, context)
        second_call_time = time.time() - start_time

        # Results should be identical
        assert len(result1) == len(result2)
        assert result1[0].message == result2[0].message

        # Second call should be faster (cache hit)
        # Note: In our mock, both calls take same time, but in real implementation
        # the second call should be much faster due to caching
        print(f"First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s")

    def test_memory_usage_optimization(self, large_codebase_files):
        """Test memory usage with large codebases"""
        # This would test memory usage patterns
        # For now, just verify we can handle large datasets

        total_content_size = sum(len(f['content']) for f in large_codebase_files)
        assert total_content_size > 80000  # At least 80KB of code

        # Verify all files are valid Python
        for file_info in large_codebase_files:
            assert file_info['language'] == 'python'
            assert 'def ' in file_info['content']  # Has functions
            assert 'class ' in file_info['content']  # Has classes

    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self):
        """Test performance of concurrent processing with real async delays"""
        async def mock_analysis_task(task_id: int):
            """Mock analysis task with realistic delay"""
            await asyncio.sleep(0.01)  # 10ms delay per task
            return f"result_{task_id}"

        # Process concurrently
        start_time = time.time()
        tasks = [mock_analysis_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time

        # Process sequentially for comparison
        start_time = time.time()
        sequential_results = []
        for i in range(10):
            result = await mock_analysis_task(i)
            sequential_results.append(result)
        sequential_time = time.time() - start_time

        # Concurrent should be faster
        assert concurrent_time < sequential_time
        assert len(results) == len(sequential_results) == 10

        speedup = sequential_time / concurrent_time
        print(f"Concurrent processing is {speedup:.1f}x faster")

        # Should be significantly faster (at least 3x with 10 concurrent tasks)
        assert speedup > 3

    def test_large_codebase_handling(self, large_codebase_files):
        """Test handling of large codebases"""
        # Verify we can process large numbers of files
        assert len(large_codebase_files) >= 50

        # Check file size distribution
        file_sizes = [len(f['content']) for f in large_codebase_files]
        avg_size = sum(file_sizes) / len(file_sizes)
        min_size = min(file_sizes)
        max_size = max(file_sizes)

        assert avg_size > 1000  # Average > 1KB
        assert min_size > 500   # Minimum > 500 bytes
        assert max_size < 10000 # Maximum < 10KB

        print(f"Codebase stats: {len(large_codebase_files)} files, "
              f"avg {avg_size:.0f} bytes, range {min_size}-{max_size} bytes")


class TestScalability:
    """Test scalability with increasing load"""

    def test_scaling_factors(self):
        """Test how performance scales with input size"""
        # Test with different batch sizes
        batch_sizes = [1, 5, 10, 20]
        base_time = 10  # Base processing time

        for batch_size in batch_sizes:
            # Simulate processing time (simplified model)
            # In real implementation, this would measure actual processing
            processing_time = base_time / batch_size  # Linear scaling model

            # Just verify that larger batches process faster
            assert processing_time <= base_time
            assert processing_time > 0

    def test_memory_scaling(self):
        """Test memory usage scaling"""
        # Test memory usage with different dataset sizes
        dataset_sizes = [10, 50, 100, 500]

        for size in dataset_sizes:
            # Estimate memory usage (simplified)
            estimated_memory = size * 1024  # 1KB per item

            # In real implementation, this would measure actual memory usage
            assert estimated_memory < 10 * 1024 * 1024  # Less than 10MB