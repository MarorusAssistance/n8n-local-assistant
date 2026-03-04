from __future__ import annotations

from app.reasoning import planner, router
from app.reasoning.types import (
    ContextDocChunk,
    ContextPack,
    ContextPackBudget,
    PlanSpec,
    PlanStep,
    RouterConstraints,
    RouterOutput,
)
from app.workflow import node_analyzer
from app.workflow.planner import PlanScope
from app.workflow.workflow_summary import NodeSummary


def _router_output() -> RouterOutput:
    return RouterOutput(
        intent="create",
        goal="build flow",
        constraints=RouterConstraints(),
        missing_info=[],
        complexity_score=1,
    )


def _context_pack() -> ContextPack:
    return ContextPack(
        nodeCards=[],
        docChunks=[ContextDocChunk(id="d1", text="doc", source="docs")],
        budget=ContextPackBudget(
            maxNodeCards=10,
            maxDocChunks=6,
            maxContextTokens=2500,
            estimatedTokens=100,
        ),
    )


def test_router_uses_legacy_fallback_when_structured_output_fails(monkeypatch) -> None:
    monkeypatch.setattr(router.settings, "ROUTER_USE_LLM", True, raising=False)
    monkeypatch.setattr(
        router,
        "_route_with_structured_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        router,
        "_route_with_legacy_json",
        lambda *args, **kwargs: RouterOutput(
            intent="fix",
            goal="legacy",
            constraints=RouterConstraints(),
            missing_info=[],
            complexity_score=2,
        ),
    )

    output = router.route_prompt("Arregla este workflow")
    assert output.intent == "fix"
    assert output.goal == "legacy"


def test_planner_uses_legacy_fallback_when_structured_output_fails(monkeypatch) -> None:
    context = ContextPack(
        nodeCards=[
            {
                "type": "n8n-nodes-base.webhook",
                "displayName": "Webhook",
            }
        ],
        docChunks=[ContextDocChunk(id="d1", text="doc", source="docs")],
        budget=ContextPackBudget(
            maxNodeCards=10,
            maxDocChunks=6,
            maxContextTokens=2500,
            estimatedTokens=100,
        ),
    )

    monkeypatch.setattr(
        planner,
        "_run_planner_structured_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        planner,
        "_run_planner_legacy_call",
        lambda *args, **kwargs: PlanSpec(
            summary="legacy",
            steps=[
                PlanStep(
                    id="s1",
                    nodeType="n8n-nodes-base.webhook",
                    purpose="trigger",
                    inputs=[],
                    outputs=["json"],
                )
            ],
            dataFlowNotes=[],
            questionsForUser=[],
        ),
    )

    plan = planner.plan_workflow(
        user_prompt="Crea un workflow",
        router_output=_router_output(),
        context_pack=context,
        model=None,
    )
    assert plan.summary == "legacy"
    assert len(plan.steps) == 1
    assert plan.steps[0].nodeType == "n8n-nodes-base.webhook"


def test_node_analyzer_uses_legacy_fallback_when_structured_output_fails(monkeypatch) -> None:
    node = NodeSummary(
        node_id="1",
        name="HTTP Request",
        node_type="n8n-nodes-base.httpRequest",
        short_type="httpRequest",
        disabled=False,
        has_credentials=True,
        has_expressions=False,
        parameter_keys=(),
        parameters_preview="",
        position=(0, 0),
    )

    monkeypatch.setattr(
        node_analyzer,
        "_run_structured_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        node_analyzer,
        "_run_legacy_json",
        lambda *args, **kwargs: node_analyzer.NodeAnalyzerOutput(
            risk_level="medium",
            findings=["legacy finding"],
            recommendations=["legacy rec"],
            confidence=0.7,
        ),
    )

    finding = node_analyzer.analyze_node(
        question="Que falla?",
        node=node,
        micro_context_text="ctx",
        docs_chunks=[{"text": "doc"}],
        scope=PlanScope.node_specific,
    )

    assert finding.risk_level == "medium"
    assert finding.findings == ["legacy finding"]
    assert finding.recommendations == ["legacy rec"]
