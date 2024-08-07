from typing import cast

import llama_cpp.llama_types as llama_types
from llama_cpp import llama

from llama_cpp.llama_chat_format import FunctionaryV25ChatHandler, functionary_new_chat_handler
from unittest.mock import Mock

from llama_cpp.llama_tokenizer import LlamaHFTokenizer


class LlamaMock:
    tokenizer_ = LlamaHFTokenizer.from_pretrained('meetkai/functionary-small-v2.5-GGUF')
    create_completion = Mock()
    verbose = True


def test_functionary_v25_generate_text_only():
    llama_mock = LlamaMock()

    llama_mock.create_completion.return_value = llama_types.CreateCompletionResponse(
        id='cmpl-0963a3b2-e0d5-4dac-948f-74fdc6875140',
        object='text_completion',
        created=1723029767,
        model='functionary-small-v2.5.Q4_0.gguf',
        choices=[
            llama_types.CompletionChoice(
                text='The capital of France is Paris.',
                index=0,
                logprobs=None,
                finish_reason='stop',
            ),
        ],
        usage=llama_types.CompletionUsage(
            prompt_tokens=82,
            completion_tokens=7,
            total_tokens=89,
        ),
    )

    response: llama_types.CreateChatCompletionResponse = functionary_new_chat_handler(
        llama=cast(llama.Llama, llama_mock),
        messages=[
            llama_types.ChatCompletionRequestUserMessage(
                role='user',
                content='what is the capital of France'
            ),
        ],
        stream=False
    )

    # Ensure the llama_mock has been called
    llama_mock.create_completion.assert_called_once()

    assert len(response.get('choices')) == 1

    message: llama_types.ChatCompletionResponseMessage = response.get('choices')[0].get('message')
    assert message.get('content') == 'The capital of France is Paris.'
    assert message.get('role') == 'assistant'


def test_functionary_v25_generate_tools():
    llama_mock = LlamaMock()

    llama_mock.create_completion.side_effect = [
        # Should use tool or text
        llama_types.CreateCompletionResponse(
            id='cmpl-f25ed8fd-85bd-49e6-b3a6-cc827774764f',
            object='text_completion',
            created=1723032052,
            model='/models/functionary-small-v2.5.Q4_0.gguf',
            choices=[
                llama_types.CompletionChoice(
                    text=FunctionaryV25ChatHandler.tool_call_token,
                    index=0,
                    logprobs=None,
                    finish_reason='length',
                ),
            ],
            usage=llama_types.CompletionUsage(
                prompt_tokens=121,
                completion_tokens=1,
                total_tokens=122,
            ),
        ),
        # which tool to uses
        llama_types.CreateCompletionResponse(
            id='cmpl-397f36e4-06ff-4f3e-8a00-f5069d2f3cac',
            object='text_completion',
            created=1723032055,
            model='/models/functionary-small-v2.5.Q4_0.gguf',
            choices=[
                llama_types.CompletionChoice(
                    text='get_current_weather',
                    index=0,
                    logprobs=None,
                    finish_reason='stop',
                ),
            ],
            usage=llama_types.CompletionUsage(
                prompt_tokens=122,
                completion_tokens=4,
                total_tokens=126,
            ),
        ),
        # determine arguments
        llama_types.CreateCompletionResponse(
            id='cmpl-702420e2-ceac-4658-9872-941b2a7f88dd',
            object='text_completion',
            created=1723032055,
            model='/models/functionary-small-v2.5.Q4_0.gguf',
            choices=[
                llama_types.CompletionChoice(
                    text='{"location": "Hanoi, Vietnam"}',
                    index=0,
                    logprobs=None,
                    finish_reason='stop',
                ),
            ],
            usage=llama_types.CompletionUsage(
                prompt_tokens=126,
                completion_tokens=9,
                total_tokens=135,
            ),
        ),
        # should continue
        llama_types.CreateCompletionResponse(
            id= 'cmpl-d17805ca-8f84-466c-8440-8c04587ed5d2',
            object= 'text_completion',
            created=1723032057,
            model= '/models/functionary-small-v2.5.Q4_0.gguf',
            choices=[
                llama_types.CompletionChoice(
                    text='',
                    index=0,
                    logprobs=None,
                    finish_reason='stop',
                ),
            ],
            usage=llama_types.CompletionUsage(
                prompt_tokens=135,
                completion_tokens=0,
                total_tokens=135,
            ),
        ),
    ]

    response: llama_types.CreateChatCompletionResponse = functionary_new_chat_handler(
        llama=cast(llama.Llama, llama_mock),
        messages=[
            llama_types.ChatCompletionRequestUserMessage(
                role='user',
                content="what's the weather like in Hanoi?",
            ),
        ],
        stream=False,
        tools=[
            llama_types.ChatCompletionTool(
                type='function',
                function=llama_types.ChatCompletionToolFunction(
                    name='get_current_weather',
                    description='Get the current weather',
                    parameters={
                        'type': 'object',
                        'properties': {
                            'location': {
                                'type': 'string',
                                'description': 'The city and state, e.g., San Francisco, CA'
                            },
                        },
                        'required': ['location']
                    }
                )
            )
        ]
    )

    # Ensure the llama_mock has been called
    llama_mock.create_completion.assert_called()

    assert len(response.get('choices')) == 1

    # should be calling tools
    assert response.get('choices')[0].get('finish_reason') == 'tool_calls'

    message: llama_types.ChatCompletionResponseMessage = response.get('choices')[0].get('message')

    assert message.get('role') == 'assistant'

    tool_calls: llama_types.ChatCompletionMessageToolCalls = message.get('tool_calls')
    assert len(tool_calls) == 1

    assert tool_calls[0].get('type') == 'function'
    assert tool_calls[0].get('function').get('name') == 'get_current_weather'
    assert tool_calls[0].get('function').get('arguments') == '{"location": "Hanoi, Vietnam"}'


def test_functionary_v25_assemble_messages():
    llama_mock = LlamaMock()

    llama_mock.create_completion.return_value = llama_types.CreateCompletionResponse(
        id='cmpl-6ac62660-6010-4eee-aa82-0d763f7101de',
        object='text_completion',
        created=1723035815,
        model='functionary-small-v2.5.Q4_0.gguf',
        choices=[
            llama_types.CompletionChoice(
                text='The current weather in Hanoi, Vietnam is 37°C.',
                index=0,
                logprobs=None,
                finish_reason='stop',
            ),
        ],
        usage=llama_types.CompletionUsage(
            prompt_tokens=112,
            completion_tokens=18,
            total_tokens=130,
        ),
    )

    response: llama_types.CreateChatCompletionResponse = functionary_new_chat_handler(
        llama=cast(llama.Llama, llama_mock),
        messages=[
            llama_types.ChatCompletionRequestUserMessage(
                role='user',
                content="what's the weather like in Hanoi?",
            ),
            llama_types.ChatCompletionRequestAssistantMessage(
                role='assistant',
                content=None,
                tool_calls=[
                    llama_types.ChatCompletionMessageToolCall(
                        id='call_EPXRajVAJ0TlvzeZ80v4uiYn',
                        type='function',
                        function=llama_types.ChatCompletionMessageToolCallFunction(
                            name='get_current_weather',
                            arguments='{"location": "Hanoi, Vietnam"}'
                        ),
                    ),
                ],
            ),
            llama_types.ChatCompletionRequestToolMessage(
                role='tool',
                content='{"location": "Hanoi, Vietnam", "temperature": "unknown"}',
                name='get_current_weather',
                tool_call_id='call_EPXRajVAJ0TlvzeZ80v4uiYn',
            ),
        ],
        stream=False,
    )

    # Ensure the llama_mock has been called
    llama_mock.create_completion.assert_called_once()

    assert len(response.get('choices')) == 1

    message: llama_types.ChatCompletionResponseMessage = response.get('choices')[0].get('message')
    assert message.get('content') == 'The current weather in Hanoi, Vietnam is 37°C.'
    assert message.get('role') == 'assistant'