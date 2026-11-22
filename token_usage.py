from pydantic import BaseModel


class TokenUsage(BaseModel):
    """记录模型输入、输出与总 Token，后续用于成本和实验对比。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

def merge_token_usage(
    total_usage: TokenUsage | None,
    current_usage,
) -> TokenUsage | None:
    """把一次模型调用的 usage 累加到本次请求的总 usage.

    第一次调用时 total_usage 是 None, 直接创建 Tokenusage。
    发生重试时,将第二次调用的 Token 加到第一次结果上。
    如果模型没有返回 usage。则保留已有统计。
    """

    if current_usage is None:
        return total_usage

    if total_usage is None:
        return TokenUsage(
            prompt_tokens=current_usage.prompt_tokens,
            completion_tokens=current_usage.completion_tokens,
            total_tokens=current_usage.total_tokens,
        )

    return TokenUsage(
        prompt_tokens=(
            total_usage.prompt_tokens
            + current_usage.prompt_tokens
        ),
        completion_tokens=(
            total_usage.completion_tokens
            + current_usage.completion_tokens
        ),
        total_tokens=(
            total_usage.total_tokens
            + current_usage.total_tokens
        ),
    )