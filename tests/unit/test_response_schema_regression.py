import json
from pathlib import Path

from app.schemas.api import ResearchResponse

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pfizer_response_pre_fix.json"


def test_complete_pfizer_response_fixture_accepts_initial_transition_without_from_state() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    response = ResearchResponse.model_validate(payload)

    assert response.run_trace.transitions[0].from_state is None
    serialized = response.model_dump(mode="json")
    assert serialized["run_trace"]["transitions"][0]["from_state"] is None
