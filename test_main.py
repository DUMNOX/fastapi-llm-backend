"""main.py 的自动化测试。

测试按 Arrange / Act / Assert 组织。所有模型调用都会被 mock，不会访问 DeepSeek，
也不会消耗 Token。模块注释说明每组测试负责保护哪一种行为。
"""

# ==================== 测试依赖与公共对象 ====================

import json
from types import SimpleNamespace

import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAIError,
)

import main


# TestClient 在进程内调用 FastAPI，不需要先启动 uvicorn。
test_client = TestClient(main.app)

# OpenAI SDK 的异常需要携带 httpx 请求或响应对象。
# example.invalid 是保留域名，只用于构造对象，不会发起网络请求。
fake_request = httpx2.Request(
    "post",
    "https://example.invalid",
)
fake_auth_response = httpx2.Response(
    401,
    request=fake_request,
)

# ==================== 基础接口测试 ====================

def test_health_returns_ok():
    """健康检查应返回固定状态。"""

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_greet_returns_expected_message():
    """问候接口应读取请求体并返回规定字段。"""

    response = test_client.post(
        "/greet",
        json={"name": "Codex"},
    )

    assert response.status_code == 200
    assert response.json() == {"greeting": "Hello, Codex",}

# ==================== 普通聊天成功与输入校验测试 ====================

def test_chat_returns_mocked_completion(monkeypatch):
    """普通聊天应发送正确参数，并转换回答与 Token usage。"""

    # Arrange：异步 fake 保持与真实 SDK 方法相同的 await 调用方式。
    async def fake_create(**kwargs):
        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["messages"] == [
            {
                "role": "system",
                "content": main.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "  什么是 FastAPI  ",
            },
        ]
    
        # SimpleNamespace 用最小对象模拟 SDK 的属性访问结构。
        return SimpleNamespace(
                choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="这是模拟回答。",
                    ),
                ),
            ],
            model="deepseek-chat",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )
    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
        )

    response = test_client.post(
        "/chat",
        json={
            "message": "  什么是 FastAPI  ",
            "history": [],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "这是模拟回答。",
        "model": "deepseek-chat",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

def test_chat_rejects_empty_message():
    """空消息应在进入接口前被 Pydantic 拒绝。"""

    response = test_client.post(
        "/chat",
        json={
            "message": "",
            "history": [],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == [
        "body",
        "message",
    ]

def test_chat_sends_history_and_allows_missing_usage(monkeypatch):
    """历史消息必须保持顺序，缺少 usage 时接口仍应成功。"""

    async def fake_create(**kwargs):
        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["messages"] == [
            {
                "role": "system",
                "content": main.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "我正在学习 FastAPI",
            },
            {
                "role": "assistant",
                "content": "很好，我们继续。",
            },
            {
                "role" : "user",
                "content": "我刚才在学什么？",
            }
        ]

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="你刚才在学习 FastAPI。",
                    ),
                ),
            ],
            model="deepseek-chat",
            usage=None,
        )
    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )


    response = test_client.post(
        "/chat",
        json={
            "message": "我刚才在学什么？",
            "history": [
                {
                    "role": "user",
                    "content": "我正在学习 FastAPI",
                },
                {
                    "role": "assistant",
                    "content": "很好，我们继续。",
                },
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "你刚才在学习 FastAPI。",
        "model": "deepseek-chat",
        "usage": None,
    }

# ==================== 普通聊天异常映射测试 ====================

# 参数化让同一套 Act/Assert 覆盖四种 SDK 异常，减少重复测试代码。
@pytest.mark.parametrize(
    (
        "api_error",
        "expected_status_code",
        "expected_detail",
    ),
    [
        (
            APITimeoutError(request=fake_request),
            504,
            "LLM service timed out",
        ),
        (
            APIConnectionError(request=fake_request),
            502,
            "LLM service connection error",
        ),
        (
            AuthenticationError(
                "bad key",
                response=fake_auth_response,
                body=None,
            ),
            502,
            "LLM service authentication failed",
        ),
        (
            OpenAIError("request failed"),
            502,
            "LLM service request failed",
        ),
    ],
)
def test_chat_maps_openai_errors_to_http_errors(
    monkeypatch,
    api_error,
    expected_status_code,
    expected_detail,
):
    """SDK 异常应转换为稳定的 HTTP 状态码和错误信息。"""

    async def fake_create(**kwargs):
        raise api_error

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    response = test_client.post(
        "/chat",
        json={
            "message": "测试异常处理",
            "history": [],
        },
    )

    assert response.status_code == expected_status_code
    assert response.json() == {
        "detail": expected_detail,
    }

# ==================== 结构化聊天成功与观测字段测试 ====================

def test_chat_structured_returns_valid_answer(
    monkeypatch,
    caplog,
):
    """合法 JSON 应通过 Schema,并返回模型、版本和固定耗时。"""

    # Arrange: 准备一个假的模型调用
    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"summary":"Git 用于版本管理",'
                            '"key_points":["保存代码快照","查看修改历史"],'
                            '"next_action":"继续练习 Git"}'
                        ),
                    ),
                ),
            ],
            model="deepseek-chat",
            usage=None,
        )

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    # 只替换 main 直接导入的 perf_counter，不修改共享 time 模块。
    # 两个时间点相差 0.125 秒，因此预期耗时固定为 125 毫秒。
    fake_times = iter([100.0, 100.125])
    monkeypatch.setattr(
        main,
        "perf_counter",
        fake_times.__next__,
    )

    # Arrange：只捕获 main 模块产生的 INFO 日志。
    caplog.set_level(
        "INFO",
        logger=main.logger.name,
    )

    #Act: 模拟客户端调用结构化接口
    response = test_client.post(
        "/chat/structured",
        json={
            "message": "Git是什么?",
            "history": [],
        },
    )

    #Assert: 检查HTTP状态结构化内容
    assert response.status_code == 200
    assert response.json() == {
        "answer":{
            "summary": "Git 用于版本管理",
            "key_points":[
                "保存代码快照",
                "查看修改历史",
            ],
            "next_action": "继续练习 Git",
        },
        "requested_model": "deepseek-chat",
        "model": "deepseek-chat",
        "prompt_version": "structured-v2",
        "llm_latency_ms": 125.0,
        "attempts": 1,
        "usage": None,
    }
    # Assert：筛选 main 模块写出的 JSON 日志。
    llm_events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == main.logger.name
    ]

    assert llm_events[-1] == {
        "event": "structured_output_succeeded",
        "requested_model": "deepseek-chat",
        "response_model": "deepseek-chat",
        "prompt_version": "structured-v2",
        "llm_latency_ms": 125.0,
        "attempts": 1,
        "usage": None,
        "error_code": None,
    }

def test_chat_structured_retries_once_and_succeeds(
    monkeypatch,
):
    """第一次输出非法 JSON 时，应修复重试并在第二次成功。"""

    # Arrange: 依次提供一次错误回答和一次正确回答，
    fake_answers = iter(
        [
            "这不是合法 JSON",
            (
                '{"summary":"第二次成功",'
                '"key_points":["已重新生成"],'
                '"next_action":"继续学习"}'
            ),
        ]
    )
    captured_messages = []

    async def fake_create(**kwargs):
        captured_messages.append(kwargs["messages"])

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=next(fake_answers),
                    ),
                ),
            ],
            model="deepseek-chat",
            usage=next(fake_usages),
        )

    first_usage = SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                )
    
    second_usage = SimpleNamespace(
                    prompt_tokens=20,
                    completion_tokens=7,
                    total_tokens=27,
                )
    
    fake_usages = iter([first_usage, second_usage])

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    #固定整个两次调用流程的起止时间为250毫秒。
    fake_times = iter([100.0, 100.25])      
    monkeypatch.setattr(
        main,
        "perf_counter",
        fake_times.__next__,
    )

    #ACT
    response = test_client.post(
        "/chat/structured",
        json={
            "message": "请返回结构化回答",
            "history": [],
        },
    )

    #Assert: 第二次成功，因此总尝试次数是2
    assert response.status_code == 200
    assert response.json()["answer"]["summary"] == "第二次成功"
    assert response.json()["attempts"] == 2
    assert response.json()["llm_latency_ms"] == 250.0
    print("实际返回：", response.json())
    assert response.json()["usage"]["prompt_tokens"] == 30
    assert response.json()["usage"]["completion_tokens"] == 12
    assert response.json()["usage"]["total_tokens"] == 42

    #模型SDK 确实被调用两次。
    assert len(captured_messages) == 2

    # 第一次使用原始消息，第二次额外带有格式修复 Prompt。
    assert (
        captured_messages[1][-1]["content"]
        == main.STRUCTURED_RETRY_PROMPT
    )






# ==================== 结构化输出失败测试 ====================

def test_chat_structured_rejects_non_json(
    monkeypatch,
    caplog,
):
    """普通文本无法通过 json.loads，应映射为 502。"""

    #Arrange: 假模型返回普通文字，不是JSON
    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="这不是合法的 JSON",
                    ),
                ),
            ],
            model="deepseek-chat",
            usage=None,
        )
    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create
    )

    caplog.set_level(
        "INFO",
        logger=main.logger.name
    )

    #Act: 调用结构化接口
    response = test_client.post(
        "/chat/structured",
        json={
            "message": "请使用结构化回答",
            "history": [],
        },
    )

    #Assert: 上游模型格式错误，应返回502
    assert response.status_code == 502
    assert response.json() == {
        "detail": {
            "code": "invalid_json",
            "message": "LLM returned invalid JSON",
        },
    }
    # Assert：失败日志只记录分类和元数据，不保存原始回答。
    llm_events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == main.logger.name
    ]
    failed_event = llm_events[-1]

    assert failed_event["event"] == "structured_output_failed"
    assert failed_event["error_code"] == "invalid_json"
    assert "raw_answer" not in failed_event
    assert "message" not in failed_event

def test_chat_structured_rejects_missing_field(monkeypatch):
    """合法 JSON 缺少字段时，应由 Pydantic 拒绝并映射为 502。"""

    #Arrange: JSON 语法格式正确，但缺少必填字段 next_action
    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"answer": "这是回答", '
                            '"key_points": ["重点一", "重点二"]}'
                        ),
                    ),
                ),
            ],
            model="deepseek-chat",
            usage=None,
        )

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    #Act: 调用结构化接口输出
    response = test_client.post(
        "/chat/structured",
        json={
            "message": "请返回结构化回答",
            "history": [],
        },
    )

    #Assert: JSON能解析，但不符合Pydantic Schema
    assert response.status_code == 502
    assert response.json() == {
        "detail": {
            "code": "schema_validation_failed",
            "message": "LLM JSON does not match the required schema",
        },
    }

# ==================== 结构化模型请求参数测试 ====================

def test_chat_structured_sends_prompt_and_json_mode(monkeypatch):
    """结构化接口必须传入模型配置、JSON mode 和两段 Prompt。"""

    #Arrange: 用字典记录接口传给模型的参数
    captured_kwargs = {}

    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"summary": "这是回答", '
                            '"key_points": ["重点一", "重点二"], '
                            '"next_action": "继续学习"}'
                        ),
                    ),
                ),
            ],

            model="deepseek-chat",
            usage=None,
        )

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    #Act
    response = test_client.post(
        "/chat/structured",
        json={
            'message': "请返回结构化回答",
            "history": [],
        },
    )

    #Assert: 接口调用成功
    assert response.status_code == 200

    #Assert 要求模型使用JSON模式
    assert captured_kwargs["response_format"] == {
        "type": "json_object",
    }
    assert captured_kwargs["model"] == main.LLM_MODEL

    # 首次成功时不应该添加修复 Prompt。
    assert len(captured_kwargs["messages"]) == 2
    assert all(
        message["content"] != main.STRUCTURED_RETRY_PROMPT
        for message in captured_kwargs["messages"]
    )

    #Assert: 第一条消息使用结构化输出Prompt
    system_message = captured_kwargs["messages"][0]

    assert system_message["role"] == "system"
    assert main.SYSTEM_PROMPT in system_message["content"]
    assert main.STRUCTURED_OUTPUT_PROMPT in system_message["content"]


def test_echo_returns_expected_message():
    """回显接口应读取请求体并返回规定字段。"""

    response = test_client.post(
        "/echo",
        json={"text": "  hellow  "},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "  hellow  ", "length" :10}

def test_echo_returns_failed_expected_message():
    """回显接口应读取请求体并返回规定字段。"""

    response = test_client.post(
        "/echo",
        json={"text": "    "},
    )

    assert response.status_code == 422











def test_chat_structured_preserves_usage_when_retry_usage_is_none(
    monkeypatch,
):
    """第一次输出非法 JSON 时，应修复重试并在第二次成功。"""

    # Arrange: 依次提供一次错误回答和一次正确回答，
    fake_answers = iter(
        [
            "这不是合法 JSON",
            (
                '{"summary":"第二次成功",'
                '"key_points":["已重新生成"],'
                '"next_action":"继续学习"}'
            ),
        ]
    )
    captured_messages = []

    async def fake_create(**kwargs):
        captured_messages.append(kwargs["messages"])

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=next(fake_answers),
                    ),
                ),
            ],
            model="deepseek-chat",
            usage=next(fake_usages),
        )

    first_usage = SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                )
    
    second_usage = None  # 第二次调用返回 None，模拟模型没有返回 usage 的情况。
    
    fake_usages = iter([first_usage, second_usage])

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    #固定整个两次调用流程的起止时间为250毫秒。
    fake_times = iter([100.0, 100.25])      
    monkeypatch.setattr(
        main,
        "perf_counter",
        fake_times.__next__,
    )

    #ACT
    response = test_client.post(
        "/chat/structured",
        json={
            "message": "请返回结构化回答",
            "history": [],
        },
    )

    #Assert: 第二次成功，因此总尝试次数是2
    assert response.status_code == 200
    assert response.json()["answer"]["summary"] == "第二次成功"
    assert response.json()["attempts"] == 2
    assert response.json()["llm_latency_ms"] == 250.0
    print("实际返回：", response.json())
    assert response.json()["usage"]["prompt_tokens"] == 10
    assert response.json()["usage"]["completion_tokens"] == 5
    assert response.json()["usage"]["total_tokens"] == 15

    #模型SDK 确实被调用两次。
    assert len(captured_messages) == 2

    # 第一次使用原始消息，第二次额外带有格式修复 Prompt。
    assert (
        captured_messages[1][-1]["content"]
        == main.STRUCTURED_RETRY_PROMPT
    )



def test_chat_rejects_whitespace_before_calling_llm(monkeypatch):
    #如果错误的进入这个llm调用，这个fake就会让测试失败
    async def fake_create(**kwargs):
        raise AssertionError("空白信息不调用 LLM")

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    #发送纯空白信息
    response = test_client.post(
        "/chat",
        json={"message": "  ", "history": []},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "message"]

def test_chat_structured_rejects_whitespace_before_calling_llm(monkeypatch):
    async def fake_create(**kwargs):
        raise AssertionError("空白信息不调用 LLM")

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    #发送纯空白信息
    response = test_client.post(
        "/chat/structured",
        json={"message": "  ", "history": []},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "message"]



def test_chat_stream_returns_mocked_stream(monkeypatch):
    """流式聊天应返回迭代器,并返回正确的回答"""
    captured_calls = []  # 放在fake外面，保存本次测试的调用记录


    async def fake_create(**kwargs):
        captured_calls.append(kwargs)  # 保存调用参数
        assert kwargs["stream"] is True

        model = kwargs["model"]
        messages = kwargs["messages"]
        assert kwargs["stream"] is True
        # 模拟流式回答，创建一个异步生成器
        async def fake_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="你")
                    )
                    ]
                )
            yield SimpleNamespace(
              choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="")
                    )
                    ]
                )
            yield SimpleNamespace(
             choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None)
                    )
                    ]
                )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="好")
                    )
                    ]
                )
        return fake_stream()

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    response = test_client.post(
        "/chat/stream",
        json={"message": "打个招呼", "history": []},
)
    assert len(captured_calls) == 1

    assert captured_calls[0]["model"] == main.LLM_MODEL
    assert captured_calls[0]["messages"] == [
        {"role": "system", "content": main.SYSTEM_PROMPT},
        {"role": "user", "content": "打个招呼"},]
    assert response.status_code == 200
    assert response.text == "你好"
    assert response.headers["content-type"].split(";")[0] == "text/plain"

    
def test_chat_stream_times_out(monkeypatch):
    """流式聊天如果超时,应返回504"""

    async def fake_create(**kwargs):
        raise APITimeoutError(request=fake_request)

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create
    )

    response = test_client.post(
        "/chat/stream",
        json={"message": "打个招呼", "history": []},
    )

    assert response.status_code == 504





def test_select_history_keeps_latest_messages_within_budget():
    history = [
        main.ChatMessage(role="user",content="甲乙"),
        main.ChatMessage(role="assistant",content="丙丁戊"),
        main.ChatMessage(role="user", content="己庚"),
    ]

    before = [message.model_dump() for message in history]

    result = main.select_history_with_budget(history,5)
    assert [message.content for message in result] == [
        "丙丁戊",
        "己庚",
    ]

    result = main.select_history_with_budget(history,4)
    assert [message.content for message in result] == [
        "己庚",
    ]

    result = main.select_history_with_budget(history,1)
    assert [message.content for message in result] == []

    result = main.select_history_with_budget(history,0)
    assert [message.content for message in result] == []

    result = main.select_history_with_budget(history=[],max_characters=5)
    assert [message.content for message in result] == []

    with pytest.raises(ValueError):
        main.select_history_with_budget(history, -1)

    assert [message.model_dump() for message in history] == before



def test_new_chat_returns_mocked_completion(monkeypatch):
    """SDK只收到预算内的最新历史，system和当前消息保持原样。"""
    history = [
        main.ChatMessage(role="user", content="甲" * 1500),
        main.ChatMessage(role="assistant", content="乙" * 1500),
        main.ChatMessage(role="user", content="丙" * 1500),
    ]
    # 输入历史共4500字符；预算4000，只能保留最新两条，共3000字符。
    async def fake_create(**kwargs):
        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["messages"] == [
            {
                "role": "system",
                "content": main.SYSTEM_PROMPT,
            },
            {
                "role": "assistant",
                "content": "乙" * 1500,
            },
            {
                "role": "user",
                "content": "丙" * 1500,
            },
            {
                "role": "user",
                "content": "  什么是 FastAPI  ",
            },
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="这是模拟回答。",
                    ),
                ),
            ],
            model="deepseek-chat",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )

    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        fake_create,
    )

    response = test_client.post(
        "/chat",
        json={
            "message": "  什么是 FastAPI  ",
            "history": [message.model_dump() for message in history],
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "这是模拟回答。"
        
             
                    






