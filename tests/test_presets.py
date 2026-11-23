"""Presets: Vollständigkeit, gültige Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung, Permalink begrenzt und rastet ein."""

import nfi_constants as C
import nfi_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds_and_options():
    pk = {v: k for k, v in P.PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        for state_key, spec in P.SETTING_SPECS.items():
            wert = preset[pk[state_key]]
            if spec.options is not None:
                assert wert in spec.options, (name, state_key, wert)
            else:
                assert spec.lo <= wert <= spec.hi, (name, state_key, wert)


def test_default_preset_equals_the_default_settings():
    erstes = C.PRESETS[next(iter(C.PRESETS))]
    pk = {v: k for k, v in P.PRESET_KEYS.items()}
    for state_key, spec in P.SETTING_SPECS.items():
        assert erstes[pk[state_key]] == spec.default, state_key


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_clamp_begrenzt_zahlen_und_rastet_auswahl_ein():
    assert P.clamp(P.SETTING_SPECS["seed_input"], 500) == C.SEED_MAX and P.clamp(P.SETTING_SPECS["seed_input"], -3) == 0
    assert P.clamp(P.SETTING_SPECS["n_select"], 60) == 50 and P.clamp(P.SETTING_SPECS["n_select"], 1000) == 200
    assert P.clamp(P.SETTING_SPECS["familie_select"], "kreis") is None and P.clamp(P.SETTING_SPECS["familie_select"], "gitter") == "gitter"
    assert P.clamp(P.SETTING_SPECS["faktor_select"], 9.0) == 10.0
    assert P.clamp(P.SETTING_SPECS["vx_slider"], 99) == C.VERSCHIEBUNG_MAX
