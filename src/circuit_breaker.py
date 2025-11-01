import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    expected_exception: tuple = (Exception,)  # Exceptions that count as failures
    success_threshold: int = 3  # Successes needed to close circuit in half-open state
    timeout: float = 30.0  # Request timeout
    name: str = "default"


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """Circuit breaker implementation"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = threading.Lock()
        self._last_state_change = time.time()

        logger.info(f"Initialized circuit breaker '{config.name}' with failure_threshold={config.failure_threshold}, recovery_timeout={config.recovery_timeout}")

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.state != CircuitBreakerState.OPEN:
            return False

        time_since_failure = time.time() - (self.stats.last_failure_time or 0)
        return time_since_failure >= self.config.recovery_timeout

    def _record_success(self):
        """Record a successful call"""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.consecutive_failures = 0
            self.stats.consecutive_successes += 1
            self.stats.last_success_time = time.time()

            # Transition from half-open to closed
            if self.state == CircuitBreakerState.HALF_OPEN and self.stats.consecutive_successes >= self.config.success_threshold:
                self._change_state(CircuitBreakerState.CLOSED)
                logger.info(f"Circuit breaker '{self.config.name}' closed after {self.stats.consecutive_successes} consecutive successes")

    def _record_failure(self, exception: Exception):
        """Record a failed call"""
        with self._lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0
            self.stats.last_failure_time = time.time()

            # Transition to open if threshold reached
            if self.state == CircuitBreakerState.CLOSED and self.stats.consecutive_failures >= self.config.failure_threshold:
                self._change_state(CircuitBreakerState.OPEN)
                logger.warning(f"Circuit breaker '{self.config.name}' opened after {self.stats.consecutive_failures} consecutive failures")

            # Transition back to open from half-open
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._change_state(CircuitBreakerState.OPEN)
                logger.warning(f"Circuit breaker '{self.config.name}' reopened after failure in half-open state")

    def _change_state(self, new_state: CircuitBreakerState):
        """Change circuit breaker state"""
        old_state = self.state
        self.state = new_state
        self.stats.state_changes += 1
        self._last_state_change = time.time()

        logger.info(f"Circuit breaker '{self.config.name}' state changed: {old_state.value} -> {new_state.value}")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function call through the circuit breaker"""
        # Check if circuit should be reset
        if self._should_attempt_reset():
            with self._lock:
                self._change_state(CircuitBreakerState.HALF_OPEN)
                logger.info(f"Circuit breaker '{self.config.name}' attempting reset")

        # Check if call is allowed
        if self.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenException(f"Circuit breaker '{self.config.name}' is OPEN")

        try:
            # Execute the function with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args, **kwargs),
                    timeout=self.config.timeout
                )

            self._record_success()
            return result

        except self.config.expected_exception as e:
            self._record_failure(e)
            raise
        except asyncio.TimeoutError as e:
            logger.warning(f"Call to '{self.config.name}' timed out after {self.config.timeout}s")
            self._record_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as circuit breaker failures
            logger.error(f"Unexpected error in circuit breaker '{self.config.name}': {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            return {
                "name": self.config.name,
                "state": self.state.value,
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "success_rate": (self.stats.successful_calls / self.stats.total_calls) if self.stats.total_calls > 0 else 0,
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "last_failure_time": self.stats.last_failure_time,
                "last_success_time": self.stats.last_success_time,
                "state_changes": self.stats.state_changes,
                "time_since_last_state_change": time.time() - self._last_state_change,
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""

    def __init__(self) -> None:
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if config is None:
            config = CircuitBreakerConfig(name=name)

        with self._lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(config)
            return self.breakers[name]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        with self._lock:
            return {name: breaker.get_stats() for name, breaker in self.breakers.items()}

    def reset_all(self):
        """Reset all circuit breakers to closed state"""
        with self._lock:
            for breaker in self.breakers.values():
                breaker._change_state(CircuitBreakerState.CLOSED)
                breaker.stats = CircuitBreakerStats()
            logger.info("Reset all circuit breakers")


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


# Convenience functions for common use cases
def get_openai_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for OpenAI API calls"""
    return circuit_breaker_registry.get_or_create(
        "openai",
        CircuitBreakerConfig(
            name="openai",
            failure_threshold=3,  # Open after 3 failures
            recovery_timeout=30.0,  # Wait 30 seconds before retry
            timeout=60.0,  # 60 second timeout for API calls
        )
    )


def get_gitlab_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for GitLab API calls"""
    return circuit_breaker_registry.get_or_create(
        "gitlab",
        CircuitBreakerConfig(
            name="gitlab",
            failure_threshold=5,  # Open after 5 failures
            recovery_timeout=60.0,  # Wait 1 minute before retry
            timeout=30.0,  # 30 second timeout for API calls
        )
    )


def get_qdrant_circuit_breaker() -> CircuitBreaker:
    """Get circuit breaker for Qdrant vector store calls"""
    return circuit_breaker_registry.get_or_create(
        "qdrant",
        CircuitBreakerConfig(
            name="qdrant",
            failure_threshold=3,  # Open after 3 failures
            recovery_timeout=15.0,  # Wait 15 seconds before retry
            timeout=10.0,  # 10 second timeout for vector operations
        )
    )