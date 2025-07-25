from .memory import *
from ._response import Response

from typing import Literal
from collections import defaultdict
import json

from openai import OpenAI, AsyncOpenAI

class Chat:

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def stream(
        self,
        user_prompt: str = None, 
        temperature: float | None = None,
        memory: Memory = None, 
        tools: list[callable] = []
    ):
        temperature = self.check_temperature(temperature)
        memory, tool_map, tools_definition = self.prepare(
            user_prompt=user_prompt,
            memory=memory,
            tools=tools
        )

        resp = Response()       
        assistant_response: str = ''
        while True:
            chat_completion_stream = self._client.chat.completions.create(
                model=self.model,
                messages=list(memory),
                tools=tools_definition if tools_definition else None,
                stream=True,
                temperature=temperature
            )

            tools_to_call = defaultdict(dict)
            now_tool_call_id: str = None

            for chunk in chat_completion_stream:
                # print(chunk)
                choice = chunk.choices[0]  
                finish_reason: Literal['stop', 'tool_calls', None] = choice.finish_reason
                content: str | None = choice.delta.content  
                if content == '' and finish_reason not in ['stop', 'tool_calls']: continue

                # Normal conversation termination
                if finish_reason == 'stop':
                    memory.update(AIMessage(content=assistant_response))
                    resp.stop_usage = self.get_stream_usage_when_done(chunk)
                    yield resp
                    return
                # Tool call termination
                elif finish_reason == 'tool_calls':
                    for k, v in dict(tools_to_call).items():
                        tool_call_id: str = k
                        tool_name: str = v['tool_name']
                        tool_args: str = v['tool_args']
                        tool_args_dict: dict = json.loads(v['tool_args'])

                        tool_response: str = self.get_tool_response(
                            tool_map=tool_map,
                            tool_name=tool_name,
                            tool_args=tool_args_dict
                        )
                        memory.update(
                            AICallToolMessage(tool_call_id, tool_name, tool_args), 
                            ToolMessage(tool_response, tool_call_id)
                        )
                        resp.add_tool_call_info(
                            name=tool_name, 
                            args=tool_args, 
                            response=tool_response, 
                            usage=self.get_stream_usage_when_done(chunk)
                        )
                        # This restricts the model to calling only one tool at a time, which has proven to be correct.
                        # The most stable pattern is: tool call -> chat -> tool call -> chat.
                        # If multiple tools are called at once, and a tool's arguments depend on the output of a previous tool, it will fail.
                        break  
                
                # Handle tool calls in non-termination cases
                if (tool_calls := choice.delta.tool_calls) is not None:
                    tool_call = tool_calls[0]
                    # func_name is a string the first time, and None in subsequent occurrences
                    func_name = tool_call.function.name
                    # func_args is an empty string the first time, and a non-empty string in subsequent occurrences
                    func_args = tool_call.function.arguments
                    if tool_call.id and tool_call.id not in tools_to_call:
                        now_tool_call_id = tool_call.id
                        tools_to_call[now_tool_call_id]['tool_name'] = func_name

                    if not tools_to_call[now_tool_call_id].get('tool_args'):
                        tools_to_call[now_tool_call_id]['tool_args'] = ''
                    if func_args is not None:
                        tools_to_call[now_tool_call_id]['tool_args'] += func_args
                    continue
                # Handle content in non-termination cases
                if content is None: continue
                else:
                    assistant_response += content
                    yield Response(content=content)

    def invoke(
        self,
        user_prompt: str,
        temperature: float | None = None,
        memory: Memory = None,
        tools: list[callable] = []
    ):
        temperature = self.check_temperature(temperature)
        memory, tool_map, tools_definition = self.prepare(
            user_prompt=user_prompt,
            memory=memory,
            tools=tools
        )
        resp = Response()
        while True:
            chat_completion = self._client.chat.completions.create(
                model=self.model,
                messages=list(memory),
                tools=tools_definition if tools_definition else None,
                temperature=temperature
            )

            choice = chat_completion.choices[0]
            finish_reason: Literal['stop', 'tool_calls'] = choice.finish_reason
            content: str = choice.message.content
            usage = {
                'input_tokens': chat_completion.usage.prompt_tokens,
                'output_tokens': chat_completion.usage.completion_tokens,
                'total_tokens': chat_completion.usage.total_tokens
            }

            if finish_reason == 'stop':
                memory.update(AIMessage(content))
                resp.content = content
                resp.stop_usage = usage
                return resp
            elif finish_reason == 'tool_calls':
                tool_call = choice.message.tool_calls[0]

                tool_call_id: str = tool_call.id
                tool_name: str = tool_call.function.name
                tool_args: str = tool_call.function.arguments
                tool_args_dict: dict = json.loads(tool_args)

                tool_response = self.get_tool_response(
                    tool_map=tool_map,
                    tool_name=tool_name,
                    tool_args=tool_args_dict
                )

                memory.update(
                    AICallToolMessage(tool_call_id, tool_name, tool_args),
                    ToolMessage(content=tool_response, tool_call_id=tool_call_id)
                )
                resp.add_tool_call_info(
                    name=tool_name,
                    args=tool_args,
                    response=tool_response,
                    usage=usage
                )

    async def astream(
        self,
        user_prompt: str,
        temperature: float | None = None,
        memory: Memory = None,
        tools: list[callable] = []
    ):
        temperature = self.check_temperature(temperature)
        memory, tool_map, tools_definition = self.prepare(
            user_prompt=user_prompt,
            memory=memory,
            tools=tools
        )

        resp = Response()
        assistant_response: str = ''
        while True:
            chat_completion_stream = await self._async_client.chat.completions.create(
                model=self.model,
                messages=list(memory),
                tools=tools_definition if tools_definition else None,
                stream=True,
                temperature=temperature
            )

            tools_to_call = defaultdict(dict)
            now_tool_call_id: str = None

            async for chunk in chat_completion_stream:
                choice = chunk.choices[0]  
                finish_reason: Literal['stop', 'tool_calls', None] = choice.finish_reason
                content: str | None = choice.delta.content
                if content == '' and finish_reason not in ['stop', 'tool_calls']: continue

                # Normal conversation termination
                if finish_reason == 'stop':
                    memory.update(AIMessage(content=assistant_response))
                    resp.stop_usage = self.get_stream_usage_when_done(chunk)
                    yield resp
                    return
                # Tool call termination
                elif finish_reason == 'tool_calls':
                    for k, v in dict(tools_to_call).items():
                        tool_call_id: str = k
                        tool_name: str = v['tool_name']
                        tool_args: str = v['tool_args']
                        tool_args_dict: dict = json.loads(v['tool_args'])

                        tool_response: str = await self.get_async_tool_response(
                            tool_map=tool_map,
                            tool_name=tool_name,
                            tool_args=tool_args_dict
                        )
                        memory.update(AICallToolMessage(tool_call_id, tool_name, tool_args), ToolMessage(tool_response, tool_call_id))
                        resp.add_tool_call_info(
                            name=tool_name, 
                            args=tool_args, 
                            response=tool_response, 
                            usage=self.get_stream_usage_when_done(chunk)
                        )
                        # This restricts the model to calling only one tool at a time, which has proven to be correct.
                        # The most stable pattern is: tool call -> chat -> tool call -> chat.
                        # If multiple tools are called at once, and a tool's arguments depend on the output of a previous tool, it will fail.
                        break  
                
                # Handle tool calls in non-termination cases
                if (tool_calls := choice.delta.tool_calls) is not None:
                    tool_call = tool_calls[0]
                    # func_name is a string the first time, and None in subsequent occurrences
                    func_name = tool_call.function.name
                    # func_args is an empty string the first time, and a non-empty string in subsequent occurrences
                    func_args = tool_call.function.arguments
                    if tool_call.id and tool_call.id not in tools_to_call:
                        now_tool_call_id = tool_call.id
                        tools_to_call[now_tool_call_id]['tool_name'] = func_name

                    if not tools_to_call[now_tool_call_id].get('tool_args'):
                        tools_to_call[now_tool_call_id]['tool_args'] = ''
                    if func_args is not None:
                        tools_to_call[now_tool_call_id]['tool_args'] += func_args
                    continue
                # Handle content in non-termination cases
                if content is None: continue
                else:
                    assistant_response += content
                    yield Response(content=content)

    async def ainvoke(
        self,
        user_prompt: str,
        temperature: float | None = None,
        memory: Memory = None,
        tools: list[callable] = []
    ):
        temperature = self.check_temperature(temperature)
        memory, tool_map, tools_definition = self.prepare(
            user_prompt=user_prompt,
            memory=memory,
            tools=tools
        )
        resp = Response()
        while True:
            chat_completion = await self._async_client.chat.completions.create(
                model=self.model,
                messages=list(memory),
                tools=tools_definition if tools_definition else None,
                temperature=temperature
            )

            choice = chat_completion.choices[0]
            finish_reason: Literal['stop', 'tool_calls'] = choice.finish_reason
            content: str = choice.message.content
            usage = {
                'input_tokens': chat_completion.usage.prompt_tokens,
                'output_tokens': chat_completion.usage.completion_tokens,
                'total_tokens': chat_completion.usage.total_tokens
            }

            if finish_reason == 'stop':
                memory.update(AIMessage(content))
                resp.content = content
                resp.stop_usage = usage
                return resp
            elif finish_reason == 'tool_calls':
                tool_call = choice.message.tool_calls[0]

                tool_call_id: str = tool_call.id
                tool_name: str = tool_call.function.name
                tool_args: str = tool_call.function.arguments
                tool_args_dict: dict = json.loads(tool_args)

                tool_response = await self.get_async_tool_response(
                    tool_map=tool_map,
                    tool_name=tool_name,
                    tool_args=tool_args_dict
                )

                memory.update(
                    AICallToolMessage(tool_call_id, tool_name, tool_args),
                    ToolMessage(content=tool_response, tool_call_id=tool_call_id)
                )
                resp.add_tool_call_info(
                    name=tool_name,
                    args=tool_args,
                    response=tool_response,
                    usage=usage
                )

    def check_tools(self, tools: list[callable]) -> tuple[dict[str, callable], list[dict]]:
        tool_map = {tool.__name__: tool for tool in tools}
        tools_definition = []
        for tool in tools:
            if not hasattr(tool, '_tool_definition'):
                raise ValueError("Tool functions must use the @crazy_tool decorator")
            tools_definition.append(tool._tool_definition)
        return tool_map, tools_definition

    def get_tool_response(
        self, 
        tool_map: dict[str, callable], 
        tool_name: str, 
        tool_args: dict
    ) -> dict:
        tool_response = tool_map[tool_name](**tool_args)
        return tool_response
    
    async def get_async_tool_response(
        self,
        tool_map: dict[str, callable],
        tool_name: str,
        tool_args: dict
    ) -> dict:
        tool = tool_map[tool_name]
        if tool._is_async:
            tool_response = await tool(**tool_args)
        else:
            tool_response = tool(**tool_args)
        return tool_response

    def prepare(
        self,
        user_prompt: str = None, 
        memory: Memory = None, 
        tools: list[callable] = []   
    ) -> tuple[Memory, dict, list]:
        if (not user_prompt is None) and not isinstance(user_prompt, str):
            raise ValueError('user_prompt must be a string or None')
        if memory:
            if not isinstance(memory, Memory):
                raise ValueError("memory must be a Memory object")
        else:
            memory = Memory()
        if user_prompt is not None:
            memory.update(HumanMessage(content=user_prompt))

        tool_map, tools_definition = self.check_tools(tools)
        return memory, tool_map, tools_definition
    
    def get_stream_usage_when_done(self, chunk) -> dict:
        # The APIs of kimi and deepseek only differ in the stream method: kimi's usage is in choice, while deepseek's usage is in chunk.
        match self.name:
            case 'deepseek' | 'openai':
                usage = dict(chunk.usage)
            case 'kimi':
                usage: dict = chunk.choices[0].usage
            case 'ollama'|'qwen':
                # ollama does not return usage information in the stream response.
                return {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0
                }
        return {
            'input_tokens': usage['prompt_tokens'],
            'output_tokens': usage['completion_tokens'],
            'total_tokens': usage['total_tokens']
        }

    def check_temperature(self, temperature: float | None) -> float:
        """Check and validate the temperature setting."""
        match self.name:
            case 'openai':
                temperature_range = (0, 2)
                default_temperature = 1.0
            case 'deepseek':
                temperature_range = (0, 1.5)
                default_temperature = 1.0
            case 'kimi':
                temperature_range = (0, 1.0)
                default_temperature = 0.3
            case 'ollama':
                if temperature is None:
                    return 0.0
            case 'qwen':
                temperature_range = (0, 2)
                default_temperature = 1.0

        if temperature is None:
            return default_temperature
        if not isinstance(temperature, float):
            raise ValueError('temperature must be a float')
        if not (temperature_range[0] <= temperature <= temperature_range[1]):
            raise ValueError(f'temperature must be in range {temperature_range}')
             
class CloseAI(Chat):

    def __init__(
        self,
        api_key: str,
        model: str = 'gpt-4o-mini',
        base_url: str = 'https://api.openai.com/v1'
    ):
        super().__init__(api_key, base_url, model)
        self.name = 'openai'

class Deepseek(Chat):

    def __init__(
        self,
        api_key: str,
        model: str = 'deepseek-chat',
        base_url: str = 'https://api.deepseek.com',
    ):
        super().__init__(api_key, base_url, model)
        self.name = 'deepseek'

class Moonshot(Chat):

    def __init__(
        self, 
        api_key: str, 
        model = 'moonshot-v1-8k', 
        base_url = 'https://api.moonshot.cn/v1'
    ):
        super().__init__(api_key, base_url, model)
        self.name = 'kimi'

class Ollama(Chat):

    def __init__(
        self,
        model: str,
        base_url: str = 'http://localhost:11434/v1/',
        api_key: str = 'ollama'
    ):
        super().__init__(api_key, base_url, model)
        self.name = 'ollama'

class Qwen(Chat):

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    ):
        super().__init__(api_key, base_url, model)
        self.name = 'qwen'
