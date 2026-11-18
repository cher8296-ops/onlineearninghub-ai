import json
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from backend.app import main as api


class RequestValidationTests(unittest.TestCase):
    def test_description_accepts_valid_text(self):
        task = api.Task(description="Create a practical work plan")
        self.assertEqual(task.description, "Create a practical work plan")

    def test_description_rejects_too_short_text(self):
        with self.assertRaises(ValidationError):
            api.Task(description="ab")

    def test_description_rejects_more_than_10000_characters(self):
        with self.assertRaises(ValidationError):
            api.Task(description="x" * 10001)


class GeminiResponseTests(unittest.TestCase):
    def test_parser_extracts_text_and_grounding_sources(self):
        body = json.dumps({
            "candidates": [{
                "content": {"parts": [{"text": "A useful answer."}]},
                "groundingMetadata": {
                    "groundingChunks": [{
                        "web": {
                            "uri": "https://example.com/source",
                            "title": "Example source"
                        }
                    }]
                }
            }]
        })

        result = api.parse_gemini_response(body)

        self.assertEqual(result["answer"], "A useful answer.")
        self.assertEqual(result["sources"][0]["url"], "https://example.com/source")

    def test_parser_rejects_missing_candidates(self):
        with self.assertRaises(HTTPException) as raised:
            api.parse_gemini_response('{"candidates": []}')
        self.assertEqual(raised.exception.status_code, 502)

    def test_parser_rejects_invalid_json(self):
        with self.assertRaises(HTTPException) as raised:
            api.parse_gemini_response("not-json")
        self.assertEqual(raised.exception.status_code, 502)


class EndpointContractTests(unittest.TestCase):
    @patch.object(api, "GEMINI_API_KEY", "test-only-key")
    def test_health_reports_configuration_without_exposing_key(self):
        result = api.health()
        self.assertTrue(result["ai_configured"])
        self.assertEqual(result["ai_status"], "configured")
        self.assertNotIn("test-only-key", json.dumps(result))

    @patch.object(api, "GEMINI_API_KEY", "")
    def test_health_reports_missing_ai_configuration(self):
        result = api.health()
        self.assertFalse(result["ai_configured"])
        self.assertEqual(result["ai_status"], "not_configured")

    @patch.object(api, "ask_ai")
    def test_plan_endpoint_preserves_frontend_response_contract(self, ask_ai):
        ask_ai.return_value = {
            "answer": "Step 1: Define the goal.",
            "sources": [],
            "model": "gemini-3.8-flash",
            "fallback_used": False,
        }

        result = api.plan(api.Task(description="Plan my first freelance project"))

        self.assertEqual(result["agent"], "TaskPlanner")
        self.assertIn("plan", result)
        self.assertEqual(result["model_used"], "gemini-3.8-flash")

    @patch.object(api, "ask_ai")
    def test_research_endpoint_discloses_no_live_search(self, ask_ai):
        ask_ai.return_value = {
            "answer": "General background information.",
            "sources": [],
            "model": "gemini-3.8-flash",
            "fallback_used": False,
        }

        result = api.research(api.Task(description="Explain remote work"))

        self.assertEqual(result["research_mode"], "AI knowledge; no live web search")
        ask_ai.assert_called_once()
        self.assertFalse(ask_ai.call_args.kwargs["use_search"])


if __name__ == "__main__":
    unittest.main()
