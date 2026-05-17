"""Tests for the NLP module — dataset, model, and inference."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False

from nlp.dataset import (
    OCCASION_SYNONYMS, OCCASION_UNKNOWN_IDX,
    OCCASIONS,
    STYLE_SYNONYMS, STYLE_UNKNOWN_IDX,
    STYLES,
    WEATHER_SYNONYMS, WEATHER_UNKNOWN_IDX,
    WEATHER_CLASSES,
    Vocabulary,
    WardrobeQueryDataset,
    generate_synthetic_data,
    preprocess_text,
)

if _torch_available:
    from nlp.model import WardrobeNLPModel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "nlp", "saved_models")
_model_available = _torch_available and os.path.exists(os.path.join(MODEL_DIR, "nlp_model.pth"))

_needs_torch = pytest.mark.skipif(not _torch_available, reason="PyTorch not installed")

from nlp.inference import _extract_explicit_temperature as _extract_temp


# =============================================================================
# preprocess_text
# =============================================================================

class TestPreprocessText:
    def test_lowercases(self):
        assert preprocess_text("FORMAL") == ["formal"]

    def test_strips_punctuation(self):
        assert preprocess_text("wedding!") == ["wedding"]

    def test_strips_numbers_leaving_words(self):
        result = preprocess_text("35 degrees")
        assert "degrees" in result

    def test_no_empty_tokens_from_digit_stripping(self):
        result = preprocess_text("35 40 100")
        assert "" not in result

    def test_empty_string_returns_empty_list(self):
        assert preprocess_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert preprocess_text("   ") == []

    def test_mixed_case_and_punctuation(self):
        result = preprocess_text("Going to the GYM, it's HOT!")
        assert "going" in result
        assert "gym" in result
        assert "hot" in result
        assert "" not in result


# =============================================================================
# generate_synthetic_data
# =============================================================================

class TestGenerateSyntheticData:
    def test_generates_correct_count(self):
        data = generate_synthetic_data(200)
        assert len(data) == 200

    def test_all_records_have_required_keys(self):
        data = generate_synthetic_data(50)
        for record in data:
            assert set(record.keys()) == {"text", "occasion", "weather", "style"}

    def test_occasion_values_are_valid(self):
        data = generate_synthetic_data(200)
        for record in data:
            assert record["occasion"] in OCCASIONS or record["occasion"] is None

    def test_weather_values_are_valid(self):
        data = generate_synthetic_data(200)
        for record in data:
            assert record["weather"] in WEATHER_CLASSES or record["weather"] is None

    def test_style_values_are_valid(self):
        data = generate_synthetic_data(200)
        for record in data:
            assert record["style"] in STYLES or record["style"] is None

    def test_data_has_some_none_occasions(self):
        data = generate_synthetic_data(500)
        assert any(r["occasion"] is None for r in data)

    def test_data_has_some_none_weather(self):
        data = generate_synthetic_data(500)
        assert any(r["weather"] is None for r in data)

    def test_data_has_some_style_labels(self):
        data = generate_synthetic_data(500)
        assert any(r["style"] is not None for r in data)

    def test_no_broken_templates(self):
        """Regression: a label must only be assigned when its word appears in the text."""
        data = generate_synthetic_data(500)
        all_occ_words = {w for syns in OCCASION_SYNONYMS.values() for w in syns}
        all_wea_words = {w for syns in WEATHER_SYNONYMS.values() for w in syns}

        for record in data:
            text_lower = record["text"].lower()
            text_has_occasion = any(w in text_lower for w in all_occ_words)
            text_has_weather  = any(w in text_lower for w in all_wea_words)

            if not text_has_occasion:
                assert record["occasion"] is None, (
                    f"Occasion label '{record['occasion']}' assigned but no occasion "
                    f"word found in: {record['text']!r}"
                )
            if not text_has_weather:
                assert record["weather"] is None, (
                    f"Weather label '{record['weather']}' assigned but no weather "
                    f"word found in: {record['text']!r}"
                )

    def test_texts_are_non_empty_strings(self):
        data = generate_synthetic_data(100)
        for record in data:
            assert isinstance(record["text"], str) and len(record["text"]) > 0


# =============================================================================
# Vocabulary
# =============================================================================

class TestVocabulary:
    def test_starts_with_pad_and_unk(self):
        v = Vocabulary()
        assert v.word2idx["<PAD>"] == 0
        assert v.word2idx["<UNK>"] == 1

    def test_initial_vocab_size_is_two(self):
        v = Vocabulary()
        assert v.vocab_size == 2

    def test_build_vocab_grows_size(self):
        v = Vocabulary()
        v.build_vocab(["hello world", "hello again"])
        assert v.vocab_size == 5  # PAD, UNK, hello, world, again

    def test_build_vocab_deduplicates(self):
        v = Vocabulary()
        v.build_vocab(["hello hello hello"])
        assert v.vocab_size == 3  # PAD, UNK, hello

    def test_encode_known_word(self):
        v = Vocabulary()
        v.build_vocab(["casual outfit"])
        encoded = v.encode("casual")
        assert encoded == [v.word2idx["casual"]]

    def test_encode_unknown_word_returns_unk(self):
        v = Vocabulary()
        v.build_vocab(["casual outfit"])
        encoded = v.encode("xyzunknown")
        assert encoded == [v.word2idx["<UNK>"]]

    def test_encode_empty_string(self):
        v = Vocabulary()
        v.build_vocab(["casual"])
        assert v.encode("") == []


# =============================================================================
# WardrobeQueryDataset
# =============================================================================

@_needs_torch
class TestWardrobeQueryDataset:
    def _make_dataset(self, n=20, max_len=25):
        data = generate_synthetic_data(n)
        vocab = Vocabulary()
        vocab.build_vocab([r["text"] for r in data])
        return WardrobeQueryDataset(data, vocab, max_len=max_len), len(data)

    def test_length_matches_data(self):
        ds, n = self._make_dataset(30)
        assert len(ds) == 30

    def test_item_has_correct_keys(self):
        ds, _ = self._make_dataset()
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "occasion_label", "weather_label", "style_label"}

    def test_input_ids_padded_to_max_len(self):
        ds, _ = self._make_dataset(max_len=25)
        for i in range(min(10, len(ds))):
            assert ds[i]["input_ids"].shape == (25,)

    def test_none_occasion_maps_to_unknown_idx(self):
        vocab = Vocabulary()
        vocab.build_vocab(["it is warm"])
        record = {"text": "it is warm", "occasion": None, "weather": "hot", "style": None}
        ds = WardrobeQueryDataset([record], vocab)
        assert ds[0]["occasion_label"].item() == OCCASION_UNKNOWN_IDX

    def test_none_weather_maps_to_unknown_idx(self):
        vocab = Vocabulary()
        vocab.build_vocab(["wear to wedding"])
        record = {"text": "wear to wedding", "occasion": "formal", "weather": None, "style": None}
        ds = WardrobeQueryDataset([record], vocab)
        assert ds[0]["weather_label"].item() == WEATHER_UNKNOWN_IDX

    def test_none_style_maps_to_unknown_idx(self):
        vocab = Vocabulary()
        vocab.build_vocab(["casual day out"])
        record = {"text": "casual day out", "occasion": "casual", "weather": "mild", "style": None}
        ds = WardrobeQueryDataset([record], vocab)
        assert ds[0]["style_label"].item() == STYLE_UNKNOWN_IDX

    def test_known_occasion_maps_correctly(self):
        vocab = Vocabulary()
        vocab.build_vocab(["formal wedding"])
        record = {"text": "formal wedding", "occasion": "formal", "weather": None, "style": None}
        ds = WardrobeQueryDataset([record], vocab)
        expected = OCCASIONS.index("formal")
        assert ds[0]["occasion_label"].item() == expected

    def test_label_tensors_are_long_dtype(self):
        ds, _ = self._make_dataset()
        item = ds[0]
        assert item["occasion_label"].dtype == torch.long
        assert item["weather_label"].dtype == torch.long
        assert item["style_label"].dtype == torch.long


# =============================================================================
# WardrobeNLPModel
# =============================================================================

@_needs_torch
class TestWardrobeNLPModel:
    def _make_model(self):
        return WardrobeNLPModel(
            vocab_size=50, embed_dim=16, hidden_dim=32,
            num_occasions=len(OCCASIONS),
            num_weather=len(WEATHER_CLASSES),
            num_styles=len(STYLES),
        )

    def test_forward_returns_three_tensors(self):
        model = self._make_model()
        x = torch.randint(0, 50, (2, 10))
        out = model(x)
        assert len(out) == 3

    def test_occasion_logits_shape(self):
        model = self._make_model()
        x = torch.randint(0, 50, (4, 8))
        occ, wea, sty = model(x)
        assert occ.shape == (4, len(OCCASIONS))

    def test_weather_logits_shape(self):
        model = self._make_model()
        x = torch.randint(0, 50, (4, 8))
        occ, wea, sty = model(x)
        assert wea.shape == (4, len(WEATHER_CLASSES))

    def test_style_logits_shape(self):
        model = self._make_model()
        x = torch.randint(0, 50, (4, 8))
        occ, wea, sty = model(x)
        assert sty.shape == (4, len(STYLES))

    def test_no_nan_in_outputs(self):
        model = self._make_model()
        x = torch.randint(0, 50, (3, 12))
        for tensor in model(x):
            assert not torch.isnan(tensor).any()

    def test_accepts_seq_len_5(self):
        model = self._make_model()
        x = torch.randint(0, 50, (2, 5))
        out = model(x)
        assert len(out) == 3

    def test_accepts_seq_len_25(self):
        model = self._make_model()
        x = torch.randint(0, 50, (2, 25))
        out = model(x)
        assert len(out) == 3

    def test_batch_size_1(self):
        model = self._make_model()
        x = torch.randint(0, 50, (1, 10))
        occ, wea, sty = model(x)
        assert occ.shape[0] == 1


# =============================================================================
# NLPInference._extract_explicit_temperature  (no trained model needed)
# =============================================================================

class TestExtractExplicitTemperature:
    """Tests for the regex-based temperature extractor in isolation."""

    def test_celsius_with_degree_symbol(self):
        assert _extract_temp("It's 22°C outside") == pytest.approx(22.0)

    def test_degrees_celsius_word(self):
        assert _extract_temp("35 degrees celsius") == pytest.approx(35.0)

    def test_fahrenheit_converts_correctly(self):
        result = _extract_temp("It's 90°F today")
        assert result == pytest.approx((90 - 32) * 5 / 9, abs=0.1)

    def test_bare_degrees_low_assumed_celsius(self):
        assert _extract_temp("only 5 degrees") == pytest.approx(5.0)

    def test_bare_degrees_high_assumed_fahrenheit(self):
        result = _extract_temp("95 degrees today")
        assert result == pytest.approx((95 - 32) * 5 / 9, abs=0.1)

    def test_minus_prefix(self):
        assert _extract_temp("minus 10 degrees") == pytest.approx(-10.0)

    def test_negative_prefix(self):
        assert _extract_temp("negative 5 degrees") == pytest.approx(-5.0)

    def test_no_number_returns_none(self):
        assert _extract_temp("it is cold outside") is None

    def test_bare_number_without_unit_returns_none(self):
        assert _extract_temp("I am 25 years old") is None

    def test_freezing_point(self):
        assert _extract_temp("0 degrees celsius") == pytest.approx(0.0)


# =============================================================================
# NLPInference.predict  (integration — requires trained model)
# =============================================================================

@pytest.mark.skipif(not _model_available, reason="Trained model not found — run nlp/train.py first")
class TestNLPInferenceIntegration:
    @pytest.fixture(scope="class")
    def infer(self):
        from nlp.inference import NLPInference
        return NLPInference(model_dir=MODEL_DIR)

    def test_predict_returns_required_keys(self, infer):
        result = infer.predict("I need a formal outfit, it's freezing")
        assert set(result.keys()) == {"occasion", "weather_class", "weather", "style"}

    def test_predict_formal_cold(self, infer):
        result = infer.predict("I need a formal outfit for a wedding, it's freezing")
        assert result["occasion"] == "formal"
        assert result["weather_class"] == "cold"

    def test_predict_sport_hot(self, infer):
        result = infer.predict("Going to the gym, it's boiling outside")
        assert result["occasion"] == "sport"
        assert result["weather_class"] == "hot"

    def test_predict_explicit_celsius_temperature(self, infer):
        result = infer.predict("It's 35 degrees outside, I need something casual")
        assert result["weather"] is not None
        assert result["weather"]["temperature"] == pytest.approx(35.0)
        assert result["weather_class"] == "hot"

    def test_predict_explicit_fahrenheit_temperature(self, infer):
        result = infer.predict("It's 32°F, what should I wear?")
        assert result["weather"] is not None
        assert result["weather"]["temperature"] == pytest.approx(0.0, abs=0.1)
        assert result["weather_class"] == "cold"

    def test_predict_empty_input_all_none(self, infer):
        result = infer.predict("")
        assert result["occasion"] is None
        assert result["weather"] is None
        assert result["style"] is None

    def test_predict_garbage_input_not_overconfident(self, infer):
        result = infer.predict("asdfjkl qwerty zzz")
        assert result["occasion"] is None or result["weather_class"] is None

    def test_cold_temperature_in_correct_filter_band(self, infer):
        result = infer.predict("It's chilly and wintry, dress accordingly")
        if result["weather"] is not None:
            assert result["weather"]["temperature"] <= 15

    def test_hot_temperature_in_correct_filter_band(self, infer):
        result = infer.predict("It's hot and sunny outside")
        if result["weather"] is not None:
            assert result["weather"]["temperature"] >= 25

    def test_cold_temp_mapping_is_in_winter_band(self, infer):
        result = infer.predict("It is freezing cold winter weather")
        if result["weather_class"] == "cold" and result["weather"] is not None:
            assert result["weather"]["temperature"] <= 5

    def test_predict_style_minimalist(self, infer):
        result = infer.predict("I want a minimal clean simple look for a casual day")
        assert result["style"] in STYLES or result["style"] is None

    def test_weather_dict_has_temperature_key(self, infer):
        result = infer.predict("Going to the office, it's warm")
        if result["weather"] is not None:
            assert "temperature" in result["weather"]
