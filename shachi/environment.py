class Environment:
    async def reset(self):
        raise NotImplementedError

    async def step(self, actions):
        raise NotImplementedError