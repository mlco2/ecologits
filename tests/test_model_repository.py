from datetime import date

from ecologits.model_repository import ModelRepository, model_family


def test_create_empty_repository():
    models = ModelRepository()
    assert isinstance(models, ModelRepository)
    assert not models.list_models()


def test_create_model_repository_default():
    models = ModelRepository.from_json()
    assert isinstance(models, ModelRepository)
    assert models.find_model(provider="openai", model_name="gpt-3.5-turbo") is not None


def test_find_unknown_provider():
    models = ModelRepository.from_json()
    assert models.find_model(provider="provider-test", model_name="gpt-3.5-turbo") is None


def test_find_unknown_model_name():
    models = ModelRepository.from_json()
    assert models.find_model(provider="openai", model_name="model-test") is None


def test_ambiguous_names():
    models = ModelRepository.from_json()
    assert models.find_model(provider="openai", model_name="gpt-4").name == "gpt-4"
    assert models.find_model(provider="openai", model_name="gpt-4-turbo").name == "gpt-4-turbo"
    assert models.find_model(provider="openai", model_name="gpt-4o").name == "gpt-4o"
    assert models.find_model(provider="openai", model_name="gpt-35-turbo").name == "gpt-35-turbo"


def test_add_custom_model():
    models = ModelRepository()
    custom_model_data = {
        "provider": "openai",
        "name": "gpt-4.1-2025-04-14",
        "architecture": {
            "type": "moe",
            "parameters": 1000
        }
    }
    models.add_model(custom_model_data)
    assert models.find_model("openai", "gpt-4.1-2025-04-14") is not None


def test_release_date():
    models = ModelRepository.from_json()
    model = models.find_model(provider="openai", model_name="gpt-4.1-mini")
    assert model.release_date == date(2025, 4, 14)
    alias = models.find_model(provider="anthropic", model_name="claude-opus-4-6")
    assert alias.release_date == date(2026, 2, 5)


def test_release_date_optional():
    models = ModelRepository()
    models.add_model({
        "provider": "openai",
        "name": "custom-model",
        "architecture": {"type": "dense", "parameters": 10}
    })
    assert models.find_model("openai", "custom-model").release_date is None


def test_model_family():
    assert model_family("gpt-4o-2024-08-06") == "gpt-4o"
    assert model_family("gpt-4o") == "gpt-4o"
    assert model_family("claude-opus-4-1-20250805") == "claude-opus-4-1"
    assert model_family("mistral-medium-2508") == "mistral-medium"
    assert model_family("codestral-2411-rc5") == "codestral"
    assert model_family("codestral-latest") == "codestral"
    assert model_family("gpt-5-chat-latest") == "gpt-5-chat"
    assert model_family("gemini-2.0-flash-001") == "gemini-2.0-flash"
    assert model_family("gpt-3.5-turbo-0125") == "gpt-3.5-turbo"
    assert model_family("gemini-3.1-pro-preview") == "gemini-3.1-pro-preview"


def test_count_active_models():
    models = ModelRepository()
    for name, release_date in [
        ("model-a", "2025-01-01"),
        ("model-a-2025-01-01", "2025-01-01"),
        ("model-a-latest", "2025-06-01"),
        ("model-b", "2025-06-01"),
        ("model-old", "2020-01-01"),
        ("model-future", "2030-01-01"),
        ("model-no-date", None),
    ]:
        models.add_model({
            "provider": "openai",
            "name": name,
            "architecture": {"type": "dense", "parameters": 10},
            "release_date": release_date,
        })
    assert models.count_active_models("openai", reference_date=date(2026, 1, 1)) == 2
    assert models.count_active_models("openai", reference_date=date(2028, 1, 1)) == 0
    assert models.count_active_models("anthropic", reference_date=date(2026, 1, 1)) == 0


def test_count_active_models_default_repository():
    models = ModelRepository.from_json()
    for provider in ["openai", "anthropic", "google_genai", "mistralai"]:
        assert models.count_active_models(provider, reference_date=date(2026, 9, 1)) > 0
