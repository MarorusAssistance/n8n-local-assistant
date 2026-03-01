from __future__ import annotations

from app.reasoning.checker import check_plan
from app.reasoning.types import (
    ContextDocChunk,
    ContextPack,
    ContextPackBudget,
    NodeCard,
    PlanSpec,
    PlanStep,
    RouterConstraints,
    RouterOutput,
)


def _router_create() -> RouterOutput:
    return RouterOutput(
        intent="create",
        goal="demo",
        constraints=RouterConstraints(),
        missing_info=[],
        complexity_score=1,
    )


def _context_pack(node_cards: list[NodeCard]) -> ContextPack:
    return ContextPack(
        nodeCards=node_cards,
        docChunks=[
            ContextDocChunk(
                id="doc-1",
                text="doc",
                source="docs",
            )
        ],
        budget=ContextPackBudget(
            maxNodeCards=10,
            maxDocChunks=6,
            maxContextTokens=2500,
            estimatedTokens=100,
        ),
    )


def test_checker_errors_when_create_has_no_trigger() -> None:
    context = _context_pack(
        [
            NodeCard(type="n8n-nodes-base.googleSheets"),
        ]
    )
    plan = PlanSpec(
        summary="plan",
        steps=[
            PlanStep(
                id="s1",
                nodeType="n8n-nodes-base.googleSheets",
                purpose="Write rows",
                inputs=["json"],
                outputs=["sheet"],
            )
        ],
        dataFlowNotes=[],
        questionsForUser=[],
    )

    result = check_plan(plan, _router_create(), context)
    codes = [issue.code for issue in result.issues]
    assert "missing_trigger_step" in codes
    assert not result.ok


def test_checker_errors_when_required_credentials_missing() -> None:
    context = _context_pack(
        [
            NodeCard(
                type="n8n-nodes-base.googleSheets",
                requiredCredentials=["googleSheetsOAuth2Api"],
            ),
            NodeCard(type="n8n-nodes-base.webhook"),
        ]
    )
    plan = PlanSpec(
        summary="plan",
        steps=[
            PlanStep(
                id="s1",
                nodeType="n8n-nodes-base.webhook",
                purpose="Trigger workflow",
                inputs=[],
                outputs=["json"],
            ),
            PlanStep(
                id="s2",
                nodeType="n8n-nodes-base.googleSheets",
                purpose="Write row",
                inputs=["json"],
                outputs=["sheet"],
                credentialsNeeded=[],
            ),
        ],
        dataFlowNotes=[],
        questionsForUser=[],
    )

    result = check_plan(plan, _router_create(), context)
    codes = [issue.code for issue in result.issues]
    assert "missing_credentials" in codes
    assert not result.ok


def test_checker_errors_when_node_type_is_not_in_node_cards() -> None:
    context = _context_pack([NodeCard(type="n8n-nodes-base.webhook")])
    plan = PlanSpec(
        summary="plan",
        steps=[
            PlanStep(
                id="s1",
                nodeType="n8n-nodes-base.unknownCustom",
                purpose="Trigger workflow",
                inputs=[],
                outputs=["json"],
            )
        ],
        dataFlowNotes=[],
        questionsForUser=[],
    )

    result = check_plan(plan, _router_create(), context)
    codes = [issue.code for issue in result.issues]
    assert "unknown_node_type" in codes
    assert not result.ok


def test_checker_warns_on_branching_without_if_or_switch() -> None:
    context = _context_pack(
        [
            NodeCard(type="n8n-nodes-base.webhook"),
            NodeCard(type="n8n-nodes-base.googleSheets"),
        ]
    )
    plan = PlanSpec(
        summary="plan",
        steps=[
            PlanStep(
                id="s1",
                nodeType="n8n-nodes-base.webhook",
                purpose="Trigger workflow",
                inputs=[],
                outputs=["json"],
            ),
            PlanStep(
                id="s2",
                nodeType="n8n-nodes-base.googleSheets",
                purpose="Route by if condition and write row",
                inputs=["json"],
                outputs=["sheet"],
            ),
        ],
        dataFlowNotes=[],
        questionsForUser=[],
    )

    result = check_plan(plan, _router_create(), context)
    warning_codes = [issue.code for issue in result.issues if issue.severity == "warn"]
    assert "branching_without_if_switch" in warning_codes
