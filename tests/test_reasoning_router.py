from __future__ import annotations

from app.config import settings
from app.reasoning.router import route_prompt


def test_router_classifies_create(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ROUTER_USE_LLM", False, raising=False)
    output = route_prompt("Crea un workflow con webhook y Google Sheets")
    assert output.intent == "create"
    assert output.complexity_score in {1, 2, 3}


def test_router_classifies_fix(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ROUTER_USE_LLM", False, raising=False)
    output = route_prompt("Arregla este workflow, falla en el nodo HTTP Request")
    assert output.intent == "fix"


def test_router_classifies_extend(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ROUTER_USE_LLM", False, raising=False)
    output = route_prompt("Extiende el workflow para enviar notificacion a Slack")
    assert output.intent == "extend"
