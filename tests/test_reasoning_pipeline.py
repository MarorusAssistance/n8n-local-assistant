from __future__ import annotations

from app.reasoning.pipeline import run_reasoning_pipeline
from app.reasoning.types import (
    CheckerIssue,
    CheckerResult,
    ContextDocChunk,
    ContextPack,
    ContextPackBudget,
    NodeCard,
    PlanSpec,
    PlanStep,
    RouterConstraints,
    RouterOutput,
)


def _router() -> RouterOutput:
    return RouterOutput(
        intent="create",
        goal="demo",
        constraints=RouterConstraints(),
        missing_info=[],
        complexity_score=1,
    )


def _context() -> ContextPack:
    return ContextPack(
        nodeCards=[
            NodeCard(type="n8n-nodes-base.webhook"),
            NodeCard(type="n8n-nodes-base.googleSheets"),
        ],
        docChunks=[
            ContextDocChunk(id="doc-1", text="doc", source="docs"),
        ],
        budget=ContextPackBudget(
            maxNodeCards=10,
            maxDocChunks=6,
            maxContextTokens=2500,
            estimatedTokens=120,
        ),
    )


def test_pipeline_runs_second_iteration_when_checker_has_errors(monkeypatch) -> None:
    calls = {
        "context_pack": 0,
        "plan_workflow": 0,
        "revise_plan": 0,
        "check_plan": 0,
    }

    def fake_route_prompt(*args, **kwargs):
        _ = args, kwargs
        return _router()

    def fake_context_pack(*args, **kwargs):
        _ = args, kwargs
        calls["context_pack"] += 1
        return _context()

    def fake_plan_workflow(*args, **kwargs):
        _ = args, kwargs
        calls["plan_workflow"] += 1
        return PlanSpec(
            summary="bad plan",
            steps=[
                PlanStep(
                    id="s1",
                    nodeType="n8n-nodes-base.googleSheets",
                    purpose="Write row",
                    inputs=["json"],
                    outputs=["sheet"],
                )
            ],
            dataFlowNotes=[],
            questionsForUser=[],
        )

    def fake_revise_plan(*args, **kwargs):
        _ = args, kwargs
        calls["revise_plan"] += 1
        return PlanSpec(
            summary="fixed plan",
            steps=[
                PlanStep(
                    id="s1",
                    nodeType="n8n-nodes-base.webhook",
                    purpose="Trigger",
                    inputs=[],
                    outputs=["json"],
                ),
                PlanStep(
                    id="s2",
                    nodeType="n8n-nodes-base.googleSheets",
                    purpose="Write row",
                    inputs=["json"],
                    outputs=["sheet"],
                ),
            ],
            dataFlowNotes=[],
            questionsForUser=[],
        )

    def fake_check_plan(*args, **kwargs):
        _ = args, kwargs
        calls["check_plan"] += 1
        if calls["check_plan"] == 1:
            return CheckerResult(
                ok=False,
                issues=[
                    CheckerIssue(
                        severity="error",
                        code="missing_trigger_step",
                        msg="missing trigger",
                    )
                ],
            )
        return CheckerResult(ok=True, issues=[])

    monkeypatch.setattr("app.reasoning.pipeline.route_prompt", fake_route_prompt)
    monkeypatch.setattr("app.reasoning.pipeline.build_context_pack", fake_context_pack)
    monkeypatch.setattr("app.reasoning.pipeline.plan_workflow", fake_plan_workflow)
    monkeypatch.setattr("app.reasoning.pipeline.revise_plan", fake_revise_plan)
    monkeypatch.setattr("app.reasoning.pipeline.check_plan", fake_check_plan)

    result = run_reasoning_pipeline("Crea workflow")

    assert result.second_iteration_used is True
    assert result.attempts == 2
    assert calls["context_pack"] == 1
    assert calls["plan_workflow"] == 1
    assert calls["revise_plan"] == 1
    assert calls["check_plan"] == 2
