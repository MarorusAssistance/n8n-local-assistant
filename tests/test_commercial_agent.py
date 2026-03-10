from __future__ import annotations

import pytest

from app.features.reasoning.multi_agent_contracts import AgentStage, BusinessContextSummary
from app.graphs.nodes import commercial_agent as commercial


def _candidate(
    *,
    title: str,
    business_problem: str,
    desired_outcome: str,
    expected_value: str,
    feasibility: str,
    value_score: float,
    clarity_score: float,
    actionability_score: float,
) -> commercial.CommercialUseCaseCandidate:
    return commercial.CommercialUseCaseCandidate(
        title=title,
        business_problem=business_problem,
        desired_outcome=desired_outcome,
        expected_value=expected_value,
        feasibility=feasibility,
        value_score=value_score,
        clarity_score=clarity_score,
        actionability_score=actionability_score,
    )


def _output(*candidates: commercial.CommercialUseCaseCandidate) -> commercial.CommercialDiscoveryOutput:
    return commercial.CommercialDiscoveryOutput(
        business_context_summary=BusinessContextSummary(
            process_scope="Operations and customer support",
            pain_points=["manual handoffs"],
            desired_outcomes=["faster response"],
            constraints=[],
        ),
        candidates=list(candidates),
        notes=[],
    )


@pytest.mark.parametrize(
    ("candidates", "selected_title"),
    [
        (
            [
                _candidate(
                    title="Automate invoice dispute triage",
                    business_problem="Finance team manually triages invoice disputes with long delays and missed SLA.",
                    desired_outcome="Route invoice disputes automatically to the right owner with SLA tracking.",
                    expected_value="Reduce financial penalties and improve cashflow reliability.",
                    feasibility="medium: existing process map is available.",
                    value_score=4.8,
                    clarity_score=4.4,
                    actionability_score=4.2,
                ),
                _candidate(
                    title="Auto-send weekly report",
                    business_problem="Managers manually export and email weekly KPI reports.",
                    desired_outcome="Send weekly KPI digest automatically.",
                    expected_value="Save admin time.",
                    feasibility="high: easy setup.",
                    value_score=2.9,
                    clarity_score=4.0,
                    actionability_score=4.3,
                ),
                _candidate(
                    title="Lead routing optimization",
                    business_problem="New leads are assigned late and sales follow-up is inconsistent.",
                    desired_outcome="Assign leads instantly based on territory and urgency.",
                    expected_value="Increase conversion and protect pipeline revenue.",
                    feasibility="medium: clear routing rules exist.",
                    value_score=4.6,
                    clarity_score=4.1,
                    actionability_score=4.0,
                ),
            ],
            "Automate invoice dispute triage",
        ),
        (
            [
                _candidate(
                    title="Onboarding checklist automation",
                    business_problem="HR tracks onboarding manually across spreadsheets and tasks are missed.",
                    desired_outcome="Standardize onboarding tasks and reminders automatically.",
                    expected_value="Reduce onboarding delays and productivity loss.",
                    feasibility="medium: process defined.",
                    value_score=4.2,
                    clarity_score=4.2,
                    actionability_score=4.1,
                ),
                _candidate(
                    title="Slack birthday reminders",
                    business_problem="Team forgets birthdays and celebrations.",
                    desired_outcome="Post birthday reminders.",
                    expected_value="Improve team morale slightly.",
                    feasibility="high: simple.",
                    value_score=1.8,
                    clarity_score=4.4,
                    actionability_score=4.5,
                ),
                _candidate(
                    title="Support escalation automation",
                    business_problem="Critical support tickets are escalated too late and breach SLA.",
                    desired_outcome="Escalate high-priority tickets automatically within minutes.",
                    expected_value="Reduce churn risk and SLA penalties.",
                    feasibility="medium: severity metadata already exists.",
                    value_score=4.7,
                    clarity_score=4.0,
                    actionability_score=3.9,
                ),
            ],
            "Support escalation automation",
        ),
        (
            [
                _candidate(
                    title="Collections follow-up automation",
                    business_problem="Accounts receivable follow-ups are manual and inconsistent.",
                    desired_outcome="Trigger follow-up sequences by overdue bucket automatically.",
                    expected_value="Accelerate collections and improve cash position.",
                    feasibility="medium: receivables data is accessible.",
                    value_score=4.7,
                    clarity_score=4.1,
                    actionability_score=4.1,
                ),
                _candidate(
                    title="Meeting note formatting",
                    business_problem="Operations team reformats meeting notes manually.",
                    desired_outcome="Auto-format meeting notes.",
                    expected_value="Minor time savings.",
                    feasibility="high: very easy.",
                    value_score=1.9,
                    clarity_score=4.0,
                    actionability_score=3.8,
                ),
                _candidate(
                    title="Client renewal risk alerts",
                    business_problem="Renewal risks are detected late and account teams react too slowly.",
                    desired_outcome="Flag at-risk renewals early and trigger follow-up actions.",
                    expected_value="Protect recurring revenue and reduce churn.",
                    feasibility="low: needs signal calibration.",
                    value_score=4.6,
                    clarity_score=3.9,
                    actionability_score=3.7,
                ),
            ],
            "Collections follow-up automation",
        ),
    ],
)
def test_multiple_use_cases_select_one_primary(
    candidates: list[commercial.CommercialUseCaseCandidate],
    selected_title: str,
) -> None:
    discovered, selected, alternatives, reason = commercial.rank_and_select_use_cases(_output(*candidates))
    assert len(discovered) >= 2
    assert selected is not None
    assert selected.title == selected_title
    assert len(alternatives) == len(discovered) - 1
    assert selected.why_selected
    assert "highest value-first priority score" in reason.lower()


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(
            title="Invoice approval reminders",
            business_problem="Approvals stall because managers forget pending invoice approvals.",
            desired_outcome="Send automatic reminders and escalations for stale approvals.",
            expected_value="Reduce payment delays and operational friction.",
            feasibility="high: approval states already tracked.",
            value_score=4.3,
            clarity_score=4.4,
            actionability_score=4.2,
        ),
        _candidate(
            title="Customer onboarding status updates",
            business_problem="Customers ask repeatedly for onboarding status due to poor visibility.",
            desired_outcome="Send proactive status updates at each process step.",
            expected_value="Reduce support load and improve customer confidence.",
            feasibility="medium: milestones are documented.",
            value_score=4.1,
            clarity_score=4.2,
            actionability_score=4.0,
        ),
    ],
)
def test_single_valid_use_case_is_selected(candidate: commercial.CommercialUseCaseCandidate) -> None:
    discovered, selected, alternatives, _reason = commercial.rank_and_select_use_cases(_output(candidate))
    assert len(discovered) == 1
    assert selected is not None
    assert alternatives == []


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(
            title="Idea",
            business_problem="Too short",
            desired_outcome="Automate",
            expected_value="Low",
            feasibility="unknown",
            value_score=2.0,
            clarity_score=1.0,
            actionability_score=1.0,
        ),
        _candidate(
            title="",
            business_problem="Manual work",
            desired_outcome="Faster",
            expected_value="Save time",
            feasibility="medium",
            value_score=3.0,
            clarity_score=1.0,
            actionability_score=1.0,
        ),
    ],
)
def test_no_valid_actionable_use_case_returns_empty(candidate: commercial.CommercialUseCaseCandidate) -> None:
    discovered, selected, alternatives, reason = commercial.rank_and_select_use_cases(_output(candidate))
    assert discovered == []
    assert selected is None
    assert alternatives == []
    assert "no actionable use case" in reason.lower()


@pytest.mark.parametrize(
    ("high_value", "easy_low_value"),
    [
        (
            _candidate(
                title="Renewal churn prevention automation",
                business_problem="At-risk renewals are detected late and churn impacts recurring revenue.",
                desired_outcome="Trigger proactive renewal playbooks before risk accounts churn.",
                expected_value="Protect ARR and prevent churn losses.",
                feasibility="low: needs orchestration across teams.",
                value_score=4.9,
                clarity_score=4.0,
                actionability_score=3.8,
            ),
            _candidate(
                title="Auto-post weekly team kudos",
                business_problem="Kudos are posted manually.",
                desired_outcome="Post kudos automatically.",
                expected_value="Minor team engagement gain.",
                feasibility="high: easy setup.",
                value_score=1.6,
                clarity_score=4.6,
                actionability_score=4.6,
            ),
        ),
        (
            _candidate(
                title="Revenue leakage alerting",
                business_problem="Billing anomalies are found too late and cause revenue leakage.",
                desired_outcome="Detect and alert anomalous billing events quickly.",
                expected_value="Prevent high-value revenue leakage.",
                feasibility="medium: data exists but needs mapping.",
                value_score=4.8,
                clarity_score=3.8,
                actionability_score=3.7,
            ),
            _candidate(
                title="Rename file attachments",
                business_problem="Ops team renames attachments manually.",
                desired_outcome="Rename attachments automatically.",
                expected_value="Small time savings.",
                feasibility="high: trivial to implement.",
                value_score=1.9,
                clarity_score=4.5,
                actionability_score=4.4,
            ),
        ),
    ],
)
def test_business_value_wins_even_if_not_easiest(
    high_value: commercial.CommercialUseCaseCandidate,
    easy_low_value: commercial.CommercialUseCaseCandidate,
) -> None:
    discovered, selected, alternatives, _reason = commercial.rank_and_select_use_cases(
        _output(high_value, easy_low_value)
    )
    assert len(discovered) == 2
    assert selected is not None
    assert selected.title == high_value.title
    assert all(item.title != selected.title for item in alternatives)


@pytest.mark.parametrize(
    ("candidate_a", "candidate_b"),
    [
        (
            _candidate(
                title="Contract renewal escalation",
                business_problem="High-value renewals are not escalated in time and deals are lost.",
                desired_outcome="Escalate renewal risks automatically to account leadership.",
                expected_value="Protect major contract revenue.",
                feasibility="medium: clear escalation matrix.",
                value_score=4.7,
                clarity_score=4.2,
                actionability_score=4.0,
            ),
            _candidate(
                title="Auto-create calendar reminders",
                business_problem="Some reminders are added manually.",
                desired_outcome="Create reminders automatically.",
                expected_value="Minor convenience.",
                feasibility="high: very easy.",
                value_score=1.7,
                clarity_score=4.4,
                actionability_score=4.5,
            ),
        ),
        (
            _candidate(
                title="SLA breach prevention",
                business_problem="Critical incidents breach SLA and trigger penalties.",
                desired_outcome="Detect and escalate SLA risk events in near real time.",
                expected_value="Avoid penalties and customer attrition.",
                feasibility="medium: requires alert rule definition.",
                value_score=4.6,
                clarity_score=4.1,
                actionability_score=3.9,
            ),
            _candidate(
                title="Auto-format internal notes",
                business_problem="Internal notes are formatted manually.",
                desired_outcome="Format notes automatically.",
                expected_value="Tiny productivity improvement.",
                feasibility="high: easy.",
                value_score=1.5,
                clarity_score=4.3,
                actionability_score=4.3,
            ),
        ),
    ],
)
def test_low_value_easy_case_not_selected(
    candidate_a: commercial.CommercialUseCaseCandidate,
    candidate_b: commercial.CommercialUseCaseCandidate,
) -> None:
    _discovered, selected, _alternatives, _reason = commercial.rank_and_select_use_cases(
        _output(candidate_a, candidate_b)
    )
    assert selected is not None
    assert selected.title == candidate_a.title


@pytest.mark.parametrize(
    "candidates",
    [
        [
            _candidate(
                title="Priority ticket escalation",
                business_problem="Priority tickets are escalated too late and customers churn.",
                desired_outcome="Escalate priority tickets automatically within SLA thresholds.",
                expected_value="Reduce churn risk and improve retention.",
                feasibility="medium: required fields already tracked.",
                value_score=4.7,
                clarity_score=4.1,
                actionability_score=4.0,
            ),
            _candidate(
                title="Internal reminder posting",
                business_problem="Internal reminders are manually posted.",
                desired_outcome="Auto-post reminders daily.",
                expected_value="Small team convenience gain.",
                feasibility="high: simple.",
                value_score=1.8,
                clarity_score=4.2,
                actionability_score=4.1,
            ),
        ],
        [
            _candidate(
                title="Collections follow-up sequencing",
                business_problem="Collections follow-up is inconsistent and receivables age increases.",
                desired_outcome="Trigger overdue follow-up sequences automatically.",
                expected_value="Improve collection rate and cashflow.",
                feasibility="medium: buckets already defined.",
                value_score=4.6,
                clarity_score=4.0,
                actionability_score=4.1,
            ),
            _candidate(
                title="Task color tagging",
                business_problem="Task colors are set manually.",
                desired_outcome="Tag tasks automatically by category.",
                expected_value="Minor visual consistency.",
                feasibility="high: easy.",
                value_score=1.7,
                clarity_score=4.1,
                actionability_score=4.0,
            ),
        ],
    ],
)
def test_alternative_use_cases_exclude_selected(
    candidates: list[commercial.CommercialUseCaseCandidate],
) -> None:
    discovered, selected, alternatives, _reason = commercial.rank_and_select_use_cases(_output(*candidates))
    assert selected is not None
    alt_ids = {item.id for item in alternatives}
    assert selected.id not in alt_ids
    assert len(alternatives) == len(discovered) - 1


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(
            title="Webhook node for invoice automation",
            business_problem="Use n8n node chain to fix manual invoice matching delays.",
            desired_outcome="Automate invoice matching process and reduce manual reconciliation.",
            expected_value="Reduce finance cycle time.",
            feasibility="medium: not complex.",
            value_score=4.2,
            clarity_score=4.0,
            actionability_score=4.0,
        ),
        _candidate(
            title="Credential-based support routing",
            business_problem="Teams discuss credentials and API keys instead of business support bottlenecks.",
            desired_outcome="Focus on support bottleneck automation without implementation specifics.",
            expected_value="Faster support response and better customer outcomes.",
            feasibility="medium: process rules exist.",
            value_score=4.1,
            clarity_score=3.9,
            actionability_score=3.8,
        ),
    ],
)
def test_technical_architecture_terms_are_not_invented(
    candidate: commercial.CommercialUseCaseCandidate,
) -> None:
    discovered, selected, _alternatives, _reason = commercial.rank_and_select_use_cases(_output(candidate))
    assert selected is not None
    combined = " ".join(
        [
            selected.title.lower(),
            selected.business_problem.lower(),
            selected.desired_outcome.lower(),
        ]
    )
    assert "node" not in combined
    assert "credential" not in combined
    assert "api key" not in combined
    assert "webhook" not in combined


def test_commercial_agent_node_sets_handoff_ready_state(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _output(
        _candidate(
            title="Customer escalation automation",
            business_problem="Escalations are manual and slow, causing SLA misses.",
            desired_outcome="Trigger escalation workflow as soon as SLA risk appears.",
            expected_value="Reduce SLA penalties and improve retention.",
            feasibility="medium: rules already documented.",
            value_score=4.8,
            clarity_score=4.1,
            actionability_score=4.2,
        )
    )
    monkeypatch.setattr(commercial, "analyze_commercial_discovery", lambda *args, **kwargs: output)
    state = {
        "user_query": "We need help with customer escalation delays and manual handling.",
        "routing_signals": ["test_route"],
        "workflow_context": {"model": None, "request_id": "req-test"},
        "missing_user_inputs": [],
    }

    updates = commercial.commercial_agent_node(state)
    assert updates["current_stage"] == "commercial_agent"
    assert updates["selected_use_case"] is not None
    assert updates["target_stage"] == AgentStage.product_manager_agent
    assert "handoff_ready_product_manager" in updates["routing_signals"]


def test_commercial_agent_does_not_call_retrieval_or_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        commercial,
        "_analyze_with_structured_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(
        commercial,
        "retrieve_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieve_context should not be called")),
        raising=False,
    )
    monkeypatch.setattr(
        commercial,
        "fetch_templates",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fetch_templates should not be called")),
        raising=False,
    )

    updates = commercial.commercial_agent_node(
        {
            "user_query": "Our manual approvals and support handoffs create delays and errors.",
            "routing_signals": [],
            "workflow_context": {"model": None, "request_id": "req-nonext"},
            "missing_user_inputs": [],
        }
    )
    assert updates["current_stage"] == "commercial_agent"


def test_commercial_deduplicates_candidates_deterministically() -> None:
    duplicate = _candidate(
        title="Invoice dispute triage",
        business_problem="Finance team manually triages invoice disputes with long delays.",
        desired_outcome="Route invoice disputes automatically by owner and SLA.",
        expected_value="Reduce penalties and improve cashflow reliability.",
        feasibility="medium",
        value_score=4.8,
        clarity_score=4.3,
        actionability_score=4.1,
    )
    other = _candidate(
        title="Support escalation automation",
        business_problem="Critical tickets are escalated late and breach SLA.",
        desired_outcome="Escalate high-priority tickets automatically.",
        expected_value="Reduce churn and SLA penalties.",
        feasibility="medium",
        value_score=4.7,
        clarity_score=4.0,
        actionability_score=3.9,
    )
    output = _output(duplicate, duplicate, other)
    discovered_1, selected_1, alternatives_1, _ = commercial.rank_and_select_use_cases(output)
    discovered_2, selected_2, alternatives_2, _ = commercial.rank_and_select_use_cases(output)

    assert len(discovered_1) == 2
    assert len(discovered_2) == 2
    assert selected_1 is not None and selected_2 is not None
    assert selected_1.id == selected_2.id
    assert [item.id for item in alternatives_1] == [item.id for item in alternatives_2]


def test_commercial_tie_break_is_stable_by_input_order() -> None:
    candidate_a = _candidate(
        title="Use case A",
        business_problem="Manual incident triage causes delays.",
        desired_outcome="Automate incident triage routing.",
        expected_value="Reduce delays and improve SLA.",
        feasibility="medium",
        value_score=4.0,
        clarity_score=4.0,
        actionability_score=4.0,
    )
    candidate_b = _candidate(
        title="Use case B",
        business_problem="Manual incident triage causes delays.",
        desired_outcome="Automate incident triage routing.",
        expected_value="Reduce delays and improve SLA.",
        feasibility="medium",
        value_score=4.0,
        clarity_score=4.0,
        actionability_score=4.0,
    )
    discovered, selected, _alternatives, _reason = commercial.rank_and_select_use_cases(
        _output(candidate_a, candidate_b)
    )
    assert len(discovered) == 2
    assert selected is not None
    assert selected.title == "Use case A"


def test_commercial_node_vague_input_does_not_fabricate_precise_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        commercial,
        "_analyze_with_structured_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    updates = commercial.commercial_agent_node(
        {
            "user_query": "Maybe automate something eventually, not sure yet.",
            "routing_signals": [],
            "workflow_context": {"model": None, "request_id": "req-vague"},
            "missing_user_inputs": [],
        }
    )
    assert updates["current_stage"] == "commercial_agent"
    assert updates["selected_use_case"] is None
    assert updates["discovered_use_cases"] == []
    assert updates["target_stage"] is None
