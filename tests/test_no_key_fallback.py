from core.generator import _call_openai_compatible_api


def test_cloud_provider_fails_cleanly_without_api_key(monkeypatch):
    def unexpected_network_call(*args, **kwargs):
        raise AssertionError("a missing API key must not trigger an HTTP request")

    monkeypatch.setattr("core.generator.requests.post", unexpected_network_call)

    result = _call_openai_compatible_api(
        provider_name="Test Provider",
        api_key=None,
        api_url="https://example.com/v1",
        model_name="example-model",
        prompt="hello",
    )

    assert "未配置 Test Provider API Key" in result
