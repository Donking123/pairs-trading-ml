# Event Bus

**Package**: `asian_adr.event_bus`

## Responsibilities

- Decouple all components via pub/sub
- Guarantee ordered delivery within a topic
- Support both in-process (`MemoryBus`) and out-of-process (Kafka) modes
- Provide backpressure signalling to publishers on critical topics

**Inputs**: events published by any component
**Outputs**: events delivered to all subscribers of the corresponding topic

## Module Structure

```
event_bus/
├── base.py             # Protocol / interface definition
├── memory_bus.py       # Synchronous in-memory bus (backtest + unit tests)
├── asyncio_bus.py      # Single-process asyncio.Queue (live prototype)
└── kafka_bus.py        # Production Kafka implementation
```

## Interface

```python
class AbstractEventBus(Protocol):
    async def publish(self, topic: str, event: BaseEvent) -> None: ...
    async def subscribe(self, topic: str, handler: Callable) -> None: ...
    async def subscribe_many(self, topics: list[str], handler: Callable) -> None: ...
```

## Topic Map

| Topic | Producers | Consumers |
|-------|-----------|-----------|
| `market-data` | Market Data Handler | H&S Engine, Position Engine, Sequencer |
| `fx-rates` | FX Rate Feed | H&S Engine, Position Engine |
| `signals` | H&S Engine | Risk Engine |
| `risk-decisions` | Risk Engine | Asian Execution Sequencer |
| `orders` | Asian Execution Sequencer | Broker Gateway, Monitoring |
| `fills` | Broker Gateway | Sequencer, Position Engine, ROCE/RUCE Calculator |
| `positions` | Position Engine | Risk Engine, Dashboard |
| `pair-registry` | Research Engine | H&S Engine, Risk Engine |
| `alerts` | Risk, Feed, Sequencer | Monitoring, Dashboard |

## Backpressure (asyncio.Queue)

```python
class AsyncioBus:
    def __init__(self, max_queue_size: int = 10_000):
        self._queues: dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=max_queue_size)
        )

    async def publish(self, topic: str, event: BaseEvent):
        try:
            self._queues[topic].put_nowait(event)
        except asyncio.QueueFull:
            metrics.event_bus_dropped_total.labels(topic=topic).inc()
            logger.warning("bus_queue_full", topic=topic)
            if topic in {"orders", "fills", "risk-decisions"}:
                await self._queues[topic].put(event)   # block on critical topics
```

## Implementations

| Implementation | Use Case | Notes |
|----------------|----------|-------|
| `MemoryBus` | Backtest + unit tests | Synchronous; deterministic; no async overhead; `flush()` after each event |
| `AsyncioBus` | Live single-process prototype (Phase 1–2) | asyncio.Queue per topic; backpressure on critical topics |
| `KafkaBus` | Production (Phase 3+) | Full durability + replay; each component becomes its own process |
