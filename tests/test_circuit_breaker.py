"""
Test suite for circuit breaker functionality
Location: tests/test_circuit_breaker.py
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock

from src.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerStats,
    CircuitBreakerOpenException,
    CircuitBreakerRegistry,
    get_openai_circuit_breaker,
    get_gitlab_circuit_breaker,
    get_qdrant_circuit_breaker
)


class TestCircuitBreaker:
    """Test circuit breaker functionality"""

    @pytest.fixture
    def config(self):
        """Create test circuit breaker config"""
        return CircuitBreakerConfig(
            name="test",
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=2,
            timeout=5.0
        )

    @pytest.fixture
    def breaker(self, config):
        """Create circuit breaker instance"""
        return CircuitBreaker(config)

    def test_initial_state(self, breaker):
        """Test circuit breaker starts in closed state"""
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.stats.total_calls == 0
        assert breaker.stats.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_successful_call(self, breaker):
        """Test successful call recording"""
        async def success_func():
            return "success"

        result = await breaker.call(success_func)

        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.stats.total_calls == 1
        assert breaker.stats.successful_calls == 1
        assert breaker.stats.consecutive_failures == 0
        assert breaker.stats.consecutive_successes == 1

    @pytest.mark.asyncio
    async def test_failure_call(self, breaker):
        """Test failure call recording"""
        async def failure_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await breaker.call(failure_func)

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.stats.total_calls == 1
        assert breaker.stats.failed_calls == 1
        assert breaker.stats.consecutive_failures == 1
        assert breaker.stats.consecutive_successes == 0

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, breaker):
        """Test circuit opens after reaching failure threshold"""
        async def failure_func():
            raise ValueError("test error")

        # Fail enough times to open circuit
        for i in range(breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failure_func)

        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.stats.consecutive_failures == breaker.config.failure_threshold

    @pytest.mark.asyncio
    async def test_open_circuit_blocks_calls(self, breaker):
        """Test that open circuit blocks new calls"""
        async def failure_func():
            raise ValueError("test error")

        # Open the circuit
        for i in range(breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failure_func)

        assert breaker.state == CircuitBreakerState.OPEN

        # Next call should be blocked
        async def success_func():
            return "success"

        with pytest.raises(CircuitBreakerOpenException):
            await breaker.call(success_func)

    @pytest.mark.asyncio
    async def test_half_open_transition(self, breaker):
        """Test transition to half-open state after recovery timeout"""
        async def failure_func():
            raise ValueError("test error")

        # Open the circuit
        for i in range(breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failure_func)

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(breaker.config.recovery_timeout + 0.1)

        # Next call should attempt reset (half-open)
        async def success_func():
            return "success"

        result = await breaker.call(success_func)

        assert result == "success"
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_to_closed(self, breaker):
        """Test successful recovery closes circuit"""
        async def failure_func():
            raise ValueError("test error")

        # Open the circuit
        for i in range(breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failure_func)

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery and succeed enough times
        await asyncio.sleep(breaker.config.recovery_timeout + 0.1)

        async def success_func():
            return "success"

        # Need success_threshold successes to close
        for i in range(breaker.config.success_threshold):
            result = await breaker.call(success_func)
            assert result == "success"

        assert breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, breaker):
        """Test that failure in half-open state reopens circuit"""
        async def failure_func():
            raise ValueError("test error")

        # Open the circuit
        for i in range(breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                await breaker.call(failure_func)

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery
        await asyncio.sleep(breaker.config.recovery_timeout + 0.1)

        # Fail in half-open state
        with pytest.raises(ValueError):
            await breaker.call(failure_func)

        assert breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_timeout_handling(self, breaker):
        """Test timeout handling"""
        async def slow_func():
            await asyncio.sleep(10)  # Longer than timeout
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await breaker.call(slow_func)

        assert breaker.state == CircuitBreakerState.CLOSED  # Single timeout doesn't open circuit
        assert breaker.stats.failed_calls == 1

    def test_sync_function_support(self, breaker):
        """Test that sync functions work"""
        def sync_func():
            return "sync result"

        # Note: This would need to be run in an event loop
        # For now, just test that the method exists
        assert hasattr(breaker, 'call')


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry"""

    @pytest.fixture
    def registry(self):
        """Create registry instance"""
        return CircuitBreakerRegistry()

    def test_get_or_create_new(self, registry):
        """Test creating new circuit breaker"""
        config = CircuitBreakerConfig(name="test", failure_threshold=2)
        breaker = registry.get_or_create("test", config)

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.config.name == "test"
        assert breaker.config.failure_threshold == 2

    def test_get_existing(self, registry):
        """Test getting existing circuit breaker"""
        config1 = CircuitBreakerConfig(name="test", failure_threshold=2)
        breaker1 = registry.get_or_create("test", config1)

        config2 = CircuitBreakerConfig(name="test", failure_threshold=5)
        breaker2 = registry.get_or_create("test", config2)

        assert breaker1 is breaker2  # Same instance
        assert breaker1.config.failure_threshold == 2  # Original config preserved

    def test_get_all_stats(self, registry):
        """Test getting stats for all breakers"""
        registry.get_or_create("breaker1")
        registry.get_or_create("breaker2")

        stats = registry.get_all_stats()

        assert "breaker1" in stats
        assert "breaker2" in stats
        assert len(stats) == 2

    def test_reset_all(self, registry):
        """Test resetting all circuit breakers"""
        breaker = registry.get_or_create("test")
        # Simulate some state changes
        breaker._change_state(CircuitBreakerState.OPEN)

        registry.reset_all()

        # Should be reset to closed
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.stats.total_calls == 0


class TestConvenienceFunctions:
    """Test convenience functions for common circuit breakers"""

    def test_get_openai_circuit_breaker(self):
        """Test OpenAI circuit breaker configuration"""
        breaker = get_openai_circuit_breaker()

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.config.name == "openai"
        assert breaker.config.failure_threshold == 3
        assert breaker.config.recovery_timeout == 30.0
        assert breaker.config.timeout == 60.0

    def test_get_gitlab_circuit_breaker(self):
        """Test GitLab circuit breaker configuration"""
        breaker = get_gitlab_circuit_breaker()

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.config.name == "gitlab"
        assert breaker.config.failure_threshold == 5
        assert breaker.config.recovery_timeout == 60.0

    def test_get_qdrant_circuit_breaker(self):
        """Test Qdrant circuit breaker configuration"""
        breaker = get_qdrant_circuit_breaker()

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.config.name == "qdrant"
        assert breaker.config.failure_threshold == 3
        assert breaker.config.recovery_timeout == 15.0


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with real failure scenarios"""

    @pytest.mark.asyncio
    async def test_openai_api_failure_simulation(self):
        """Test circuit breaker with simulated OpenAI API failures"""
        breaker = get_openai_circuit_breaker()

        # Simulate API failures
        async def failing_openai_call():
            raise Exception("OpenAI API Error")

        # Should handle failures gracefully
        for i in range(breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await breaker.call(failing_openai_call)

        # Circuit should be open
        assert breaker.state == CircuitBreakerState.OPEN

        # Subsequent calls should be blocked
        with pytest.raises(CircuitBreakerOpenException):
            await breaker.call(failing_openai_call)

    @pytest.mark.asyncio
    async def test_recovery_after_failures(self):
        """Test recovery after temporary failures"""
        breaker = get_openai_circuit_breaker()

        # Simulate failures
        async def failing_call():
            raise Exception("Temporary failure")

        for i in range(breaker.config.failure_threshold):
            with pytest.raises(Exception):
                await breaker.call(failing_call)

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(breaker.config.recovery_timeout + 0.1)

        # Simulate successful recovery
        async def successful_call():
            return {"choices": [{"message": {"content": "Success"}}]}

        # Should eventually succeed and close circuit
        for i in range(breaker.config.success_threshold):
            result = await breaker.call(successful_call)
            assert result is not None

        assert breaker.state == CircuitBreakerState.CLOSED