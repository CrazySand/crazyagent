from __future__ import annotations

from .utils import CS

from abc import ABC, abstractmethod
import json

from tabulate import tabulate
from typeguard import typechecked

# When using list[str] as a type annotation, typeguard behaves as follows:
# For example, print(x([dict(), 1, 'abc']))
# The list must contain at least one element of type str.
# typeguard performs a "most permissive match" check; if it finds at least one str, it considers this a "partially matching" case.

MAXCOLWIDTH = 100

class Message(ABC):

    @abstractmethod
    def __iter__(self):
        pass

@typechecked
class SystemMessage(Message):

    def __init__(self, content: str):
        self.role = 'system'
        self.content = content

    def __iter__(self):
        yield 'role', self.role
        yield 'content', self.content

    def format(self, **kwargs) -> SystemMessage:
        self.content = self.content.format(**kwargs)
        return self

@typechecked
class HumanMessage(Message):

    def __init__(self, content: str):
        self.role = 'user'
        self.content = content

    def __iter__(self):
        yield 'role', self.role
        yield 'content', self.content

@typechecked
class AIMessage(Message):

    def __init__(self, content: str):
        self.role = 'assistant'
        self.content = content

    def __iter__(self):
        yield 'role', self.role
        yield 'content', self.content

@typechecked
class AICallToolMessage(Message):

    def __init__(self, tool_call_id: str, tool_name: str, tool_args: str):
        self.role = 'assistant'
        self.content = None
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.tool_args = tool_args

    def __iter__(self):
        yield 'role', self.role
        yield 'content', self.content
        yield 'tool_calls', [
            {
                'id': self.tool_call_id,
                'type': 'function',
                'function': {
                    'name': self.tool_name,
                    'arguments': self.tool_args,
                },
                'type': 'function',
                'index': 0
            }
        ]

@typechecked
class ToolMessage(Message):

    def __init__(self, content: str, tool_call_id: str):
        self.role = 'tool'
        self.content = content
        self.tool_call_id = tool_call_id
    
    def __iter__(self):
        yield 'role', self.role
        yield 'content', self.content
        yield 'tool_call_id', self.tool_call_id

class Memory:

    def __init__(self, max_turns: int = 5):
        self._messages: list[Message] = []
        self._system_message: SystemMessage = None
        self.max_turns = max_turns

    @property
    def system_message(self) -> SystemMessage:
        return self._system_message

    @system_message.setter
    def system_message(self, system_message: SystemMessage) -> None:
        if not isinstance(system_message, SystemMessage):
            raise ValueError('System message must be an instance of the SystemMessage class')
        self._system_message = system_message

    def update(self, *args) -> None:
        for m in args:
            if not isinstance(m, Message):
                raise ValueError('Message must be an instance of the Message class')
            if isinstance(m, SystemMessage):
                raise ValueError('Please set the system message using the system_message property')
        for m in args:
            self._messages.append(m)

    def pop(self) -> Message:
        return self._messages.pop()

    def __iter__(self):
        """Return messages limited by max_turns, for use as the 'messages' parameter in the OpenAI module."""
        messages = self._messages
        if len(messages) > self.max_turns * 2:
            messages = messages[-self.max_turns*2:]
        if self._system_message:
            messages = [self._system_message] + messages
        yield from [dict(m) for m in messages]

    def __str__(self):
        """Tabular display of all chat messages"""
        r = [['Role', 'Content']]
        if self._system_message:
            r.append([CS.red('system'), CS.red(self._system_message.content)])
        for m in self._messages:
            if isinstance(m, HumanMessage):
                role = CS.purple('user')
                content = CS.purple(m.content)
            elif isinstance(m, AIMessage):
                role = CS.blue('assistant')
                content = CS.blue(m.content)
            elif isinstance(m, AICallToolMessage):
                role = CS.yellow('assistant')
                try:
                    tool_args = json.loads(m.tool_args)
                    formatted_args = ', '.join(
                        f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                        for k, v in tool_args.items()
                    )
                    content = CS.yellow(f"{m.tool_name}({formatted_args})")
                except:
                    content = CS.yellow(f"{m.tool_name}(???)")
            elif isinstance(m, ToolMessage):
                role = CS.green('tool')
                content = CS.green(m.content)

            r.append([role, content])
        return tabulate(r, headers='firstrow', tablefmt='grid', maxcolwidths=[None, MAXCOLWIDTH],disable_numparse=True)

__all__ = [
    'Memory',
    'SystemMessage',
    'HumanMessage',
    'AIMessage',
    'AICallToolMessage',
    'ToolMessage'
]