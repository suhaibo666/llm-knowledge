"""Single-process asyncio fixture; assumes calls use one event loop."""
import asyncio


class Cache:
    def __init__(self, value):
        self.version = 0
        self.value = value
        self.lock = asyncio.Lock()

    async def refresh(self, loader):
        baseline = self.version
        proposal = await loader()
        async with self.lock:
            if self.version != baseline:
                return False
            self.value = proposal
            self.version += 1
            return True

    def read(self):
        return self.version, self.value
