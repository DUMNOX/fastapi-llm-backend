from copy import deepcopy

import pytest

from project_tasks import select_project_tasks
from project_tool import get_project_tasks


@pytest.fixture
def snapshot():
    # 教学快照/规划样例，不是自动读取的实时工程状态。
    return {
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


def test_select_agent(snapshot):

    result = select_project_tasks(snapshot,"agent")

    expected = {
    "project": "agent",
    "as_of": "2026-10-22",
    "data_kind": "learning_snapshot",
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
    }
    assert result == expected


def test_unknown_project(snapshot):

    with  pytest.raises(ValueError):
        select_project_tasks(snapshot,"www")


def test_empty_tasks(snapshot):
   snapshot["projects"]["agent"]["tasks"] = []

   result = select_project_tasks(snapshot,"agent")

   assert result["tasks"] == []




def test_snapshot_unchanged(snapshot):
    before = deepcopy(snapshot)
    result = select_project_tasks(snapshot,"agent")
    assert snapshot == before

def test_select_bishe(snapshot):
    result = select_project_tasks(snapshot,"bishe")
    expected = {
        "as_of": "2026-10-22",
        "data_kind":"learning_snapshot",
        "project": "bishe",
        "tasks": [
            {
                "id": "bishe-response-id",
                "title": "合成模型响应ID检查",
                "status": "done",
            }
        ]
    }
    assert result == expected

def test_get_project_tasks():
    result = get_project_tasks("agent")
    assert  result["project"] == "agent"
    assert  result["tasks"][0]["id"] == "agent-stream-ok"
    
