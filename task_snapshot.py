"""固定教学快照；不是实时工程状态，日期保持原始快照日期。"""

LEARNING_SNAPSHOT = {
    "as_of": "2026-10-22",
    "data_kind": "learning_snapshot",
    "projects": {
        "agent": {
            "tasks": [
                {
                    "id": "agent-stream-ok",
                    "title": "流式成功路径测试",
                    "status": "done",
                },
                {
                    "id": "agent-tool-query",
                    "title": "项目任务查询工具",
                    "status": "pending",
                },
            ]
        },
        "bishe": {
            "tasks": [
                {
                    "id": "bishe-response-id",
                    "title": "合成模型响应ID检查",
                    "status": "done",
                }
            ]
        },
    },
}
