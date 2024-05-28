from openai import OpenAI
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessage,
    ChatCompletionToolMessageParam,
)

from functions import get_invoices


class AiAgent:

    system_prompt = """
You are in charge of answering financial questions.
"""

    available_functions: list[ChatCompletionToolParam] = [
        {
            "type": "function",
            "function": {
                "name": "get_invoices",
                "description": "Get all invoices for a given user.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]

    def __init__(self):
        self.openai = OpenAI(api_key="sk-proj-xxxxxxxxxx")

    def start_chat(self, question: str):
        answer: str | ChatCompletionMessage | None = None

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        while not isinstance(answer, str):
            answer = self.chat_assistant(messages, self.available_functions)
            if isinstance(answer, ChatCompletionMessage) and answer.tool_calls:
                messages.append(answer)  # type: ignore
                result = []
                for tool_call in answer.tool_calls:
                    if tool_call.function.name == "get_invoices":
                        invoices = str(get_invoices())
                        tool_result: ChatCompletionToolMessageParam = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": invoices,
                        }

                        result.append(tool_result)
                messages.extend(result)

        return answer

    def chat_assistant(
        self,
        messages: list[ChatCompletionMessageParam],
        descriptions: list[ChatCompletionToolParam],
    ) -> str | ChatCompletionMessage:
        response = self.openai.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=messages,
            tools=descriptions,
            temperature=0,
        )

        assistant_answer = response.choices[0].message

        if assistant_answer.tool_calls and len(assistant_answer.tool_calls) > 0:
            return assistant_answer
        elif assistant_answer.content:
            return assistant_answer.content
        return assistant_answer
