import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from backend.app import main as api
from backend.app import accounts as accounts_api


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


class OpenAIFallbackTests(unittest.TestCase):
    @patch.object(api, "OPENAI_API_KEY", "test-openai-key")
    @patch.object(api, "GEMINI_API_KEY", "test-gemini-key")
    @patch.object(api, "call_openai")
    @patch.object(api, "call_gemini")
    def test_openai_is_used_when_all_gemini_models_are_unavailable(
        self, call_gemini, call_openai
    ):
        call_gemini.side_effect = HTTPError(
            "https://example.invalid", 503, "Unavailable", {}, None
        )
        call_openai.return_value = {
            "answer": "Backup answer.",
            "sources": [],
        }

        result = api.ask_ai("Write a short plan")

        self.assertEqual(result["answer"], "Backup answer.")
        self.assertEqual(result["provider"], "openai")
        self.assertTrue(result["fallback_used"])
        call_openai.assert_called_once_with("Write a short plan")
        self.assertEqual(call_gemini.call_count, 3)

    @patch.object(api, "OPENAI_API_KEY", "test-openai-key")
    @patch.object(api, "GEMINI_API_KEY", "")
    @patch.object(api, "call_openai")
    @patch.object(api, "call_gemini")
    def test_openai_works_when_gemini_key_is_missing(self, call_gemini, call_openai):
        call_openai.return_value = {
            "answer": "Independent provider answer.",
            "sources": [],
        }

        result = api.ask_ai("Write a short plan")

        self.assertEqual(result["answer"], "Independent provider answer.")
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["model"], api.OPENAI_MODEL)
        self.assertTrue(result["fallback_used"])
        call_openai.assert_called_once_with("Write a short plan")
        call_gemini.assert_not_called()

    @patch.object(api, "OPENAI_API_KEY", "")
    @patch.object(api, "GEMINI_API_KEY", "test-gemini-key")
    @patch.object(api, "call_gemini")
    def test_no_openai_fallback_is_attempted_without_key(self, call_gemini):
        call_gemini.side_effect = HTTPError(
            "https://example.invalid", 503, "Unavailable", {}, None
        )
        with self.assertRaises(HTTPException) as raised:
            api.ask_ai("Write a short plan")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(call_gemini.call_count, 3)


class EndpointContractTests(unittest.TestCase):
    @patch.object(api, "GEMINI_API_KEY", "test-only-key")
    def test_health_reports_configuration_without_exposing_key(self):
        result = api.health()
        self.assertTrue(result["ai_configured"])
        self.assertEqual(result["ai_status"], "configured")
        self.assertEqual(result["available_providers"], ["gemini"])
        self.assertNotIn("test-only-key", json.dumps(result))

    @patch.object(api, "GEMINI_API_KEY", "")
    def test_health_reports_missing_ai_configuration(self):
        result = api.health()
        self.assertFalse(result["ai_configured"])
        self.assertEqual(result["ai_status"], "not_configured")

    @patch.object(api, "GEMINI_API_KEY", "")
    @patch.object(api, "OPENAI_API_KEY", "test-openai-key")
    def test_health_reports_openai_only_configuration(self):
        result = api.health()
        self.assertTrue(result["ai_configured"])
        self.assertEqual(result["ai_status"], "configured")
        self.assertEqual(result["available_providers"], ["openai"])

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
    def test_write_endpoint_preserves_frontend_response_contract(self, ask_ai):
        ask_ai.return_value = {
            "answer": "A professional draft.",
            "sources": [],
            "model": "gemini-3.5-flash-lite",
            "fallback_used": False,
        }

        result = api.write(api.Task(description="Write a short proposal"))

        self.assertEqual(result["agent"], "WritingAssistant")
        self.assertEqual(result["content"], "A professional draft.")
        self.assertEqual(result["model_used"], "gemini-3.5-flash-lite")
        ask_ai.assert_called_once()

    @patch.object(api, "ask_ai")
    def test_research_endpoint_uses_live_grounded_search(self, ask_ai):
        ask_ai.return_value = {
            "answer": "Current findings.",
            "sources": [{"title": "Example", "url": "https://example.com"}],
            "model": "gemini-3.5-flash-lite",
            "fallback_used": False,
        }

        result = api.research(api.Task(description="Current remote work options"))

        self.assertEqual(result["research_mode"], "Live Google Search grounding")
        self.assertEqual(result["sources"][0]["url"], "https://example.com")
        self.assertIsNone(result["warning"])
        ask_ai.assert_called_once()
        self.assertTrue(ask_ai.call_args.kwargs["use_search"])

    @patch.object(api, "ask_ai")
    def test_research_falls_back_transparently_when_live_search_quota_is_unavailable(self, ask_ai):
        ask_ai.side_effect = [
            HTTPException(status_code=503, detail="Quota unavailable"),
            {
                "answer": "General background overview.",
                "sources": [],
                "model": "gemini-3.5-flash-lite",
                "fallback_used": False,
            },
        ]

        result = api.research(api.Task(description="Explain remote work trends"))

        self.assertEqual(
            result["research_mode"],
            "AI knowledge fallback (live search unavailable)",
        )
        self.assertEqual(result["findings"], "General background overview.")
        self.assertEqual(result["sources"], [])
        self.assertIn("not been verified", result["warning"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(ask_ai.call_count, 2)
        self.assertTrue(ask_ai.call_args_list[0].kwargs["use_search"])
        self.assertNotIn("use_search", ask_ai.call_args_list[1].kwargs)

    @patch.object(api, "ask_ai")
    def test_research_does_not_hide_non_quota_errors(self, ask_ai):
        ask_ai.side_effect = HTTPException(status_code=502, detail="Invalid credentials")

        with self.assertRaises(HTTPException) as raised:
            api.research(api.Task(description="Research this topic"))

        self.assertEqual(raised.exception.status_code, 502)
        ask_ai.assert_called_once()

    @patch.object(api, "ask_ai")
    def test_quality_check_endpoint_preserves_frontend_response_contract(self, ask_ai):
        ask_ai.return_value = {
            "answer": "The draft is clear; improve the call to action.",
            "sources": [],
            "model": "gemini-3.5-flash-lite",
            "fallback_used": False,
        }

        result = api.check(api.Task(description="Please review this draft"))

        self.assertEqual(result["agent"], "QualityChecker")
        self.assertIn("review", result)
        self.assertEqual(result["model_used"], "gemini-3.5-flash-lite")
        ask_ai.assert_called_once()



class AccountSecurityTests(unittest.TestCase):
    def test_password_hash_verifies_and_rejects_wrong_password(self):
        encoded = accounts_api._password_hash("long-password-123")
        self.assertTrue(accounts_api._password_valid("long-password-123", encoded))
        self.assertFalse(accounts_api._password_valid("wrong-password-123", encoded))

    @patch.object(accounts_api, "JWT_SECRET", "test-secret-with-sufficient-entropy")
    def test_signed_session_token_round_trip(self):
        token = accounts_api._token({"sub": "42", "email": "person@example.com"}, accounts_api.JWT_SECRET, 60)
        claims = accounts_api._decode_token(token)
        self.assertEqual(claims["sub"], "42")
        self.assertEqual(claims["email"], "person@example.com")

    @patch.object(accounts_api, "JWT_SECRET", "test-secret-with-sufficient-entropy")
    def test_tampered_session_token_is_rejected(self):
        token = accounts_api._token({"sub": "42"}, accounts_api.JWT_SECRET, 60)
        parts = token.split(".")
        parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
        with self.assertRaises(HTTPException):
            accounts_api._decode_token(".".join(parts))

    def test_email_validation(self):
        self.assertTrue(accounts_api.re_email("person@example.com"))
        self.assertFalse(accounts_api.re_email("not-an-email"))

if __name__ == "__main__":
    unittest.main()
