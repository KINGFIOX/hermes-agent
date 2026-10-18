"""Z.AI / GLM provider profiles: OpenAI-compatible route + Coding Plan Anthropic routes.

The Coding Plan profiles (``zai-coding`` / ``zai-coding-cn``) declare
``api_mode="anthropic_messages"`` so the Anthropic Messages transport drives them; the
profile's chat_completions ``build_api_kwargs_extras`` projection must then be a no-op —
reasoning on that wire is mapped by ``agent.anthropic_adapter._glm_thinking_kwargs``.
"""

from __future__ import annotations


class TestZaiCodingProfiles:
    def test_coding_plan_profiles_registered(self) -> None:
        from providers import get_provider_profile

        coding = get_provider_profile("zai-coding")
        assert coding is not None
        assert coding.api_mode == "anthropic_messages"
        assert coding.base_url == "https://api.z.ai/api/anthropic"

        cn = get_provider_profile("zai-coding-cn")
        assert cn is not None
        assert cn.api_mode == "anthropic_messages"
        assert cn.base_url == "https://open.bigmodel.cn/api/anthropic"

        # Aliases resolve to the same profiles.
        assert get_provider_profile("zai-anthropic") is coding
        assert get_provider_profile("bigmodel-coding") is cn
        assert get_provider_profile("glm-anthropic") is coding

    def test_anthropic_mode_profile_projects_no_chat_completions_extras(self) -> None:
        """On the Anthropic Messages wire the zai profile must not leak
        extra_body.thinking / reasoning_effort (those are chat_completions fields)."""
        from providers import get_provider_profile

        coding = get_provider_profile("zai-coding")
        extra, top = coding.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"}, model="glm-5.3",
        )
        assert extra == {}
        assert top == {}

    def test_chat_completions_profile_still_projects(self) -> None:
        from providers import get_provider_profile

        zai = get_provider_profile("zai")
        assert zai is not None
        extra, top = zai.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"}, model="glm-5.3",
        )
        assert extra["thinking"] == {"type": "enabled", "clear_thinking": False}
        assert top["reasoning_effort"] == "max"


class TestZaiModelCapabilities:
    """Per-model capability declarations (canonical model_overrides schema)."""

    def test_capabilities_declared(self) -> None:
        from providers import get_provider_profile

        zai = get_provider_profile("zai")
        assert zai is not None
        caps = zai.model_capabilities
        assert caps["glm-5.3"] == {
            "supports_reasoning": True, "supports_vision": False, "model_family": "glm",
        }
        assert caps["glm-5.2[1m]"]["supports_vision"] is False
        assert caps["glm-4.5-flash"]["supports_reasoning"] is True
        assert caps["glm-5v-turbo"]["supports_vision"] is True
        assert caps["glm-4.6v-flashx"]["supports_vision"] is True

    def test_capabilities_patch_catalog_entry(self) -> None:
        """A catalog entry that wrongly claims vision gets corrected to text-only."""
        from agent.models_dev import _apply_overrides

        stale = {"limit": {"context": 1_000_000, "output": 128_000},
                 "modalities": {"input": ["text", "image"], "output": ["text"]},
                 "tool_call": True, "reasoning": True}
        patched = _apply_overrides("zai", "glm-5.3", stale, config=None)
        assert patched is not None
        from agent.models_dev import _entry_supports_vision

        assert _entry_supports_vision(patched) is False

    def test_per_model_max_output(self) -> None:
        from providers import get_provider_profile

        zai = get_provider_profile("zai")
        assert zai is not None
        assert zai.get_max_tokens("glm-5.3") == 128_000
        assert zai.get_max_tokens("glm-5.2[1m]") == 128_000
        assert zai.get_max_tokens("glm-4.7") == 131_072
        assert zai.get_max_tokens("glm-4.6v-flash") == 32_768
        assert zai.get_max_tokens("glm-4-9b") is None

    def test_default_reasoning_config_records_server_default(self) -> None:
        from providers import get_provider_profile

        zai = get_provider_profile("zai")
        assert zai is not None
        assert zai.default_reasoning_config("glm-5.3") == {"enabled": True}
        assert zai.default_reasoning_config("glm-4.6") == {"enabled": True}
        assert zai.default_reasoning_config("glm-4-9b") is None
        assert zai.default_reasoning_config(None) is None
