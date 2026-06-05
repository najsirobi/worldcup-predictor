from __future__ import annotations

from src.features.travel_burden import score_travel_burden, timezone_delta_hours


def test_same_venue_lowers_burden():
    different_venue = score_travel_burden(
        rest_days=3,
        distance_from_previous_km=500,
        timezone_delta_hours_value=0,
        elevation_delta_m=0,
        same_venue_flag=False,
        same_host_city_flag=False,
    )
    same_venue = score_travel_burden(
        rest_days=3,
        distance_from_previous_km=0,
        timezone_delta_hours_value=0,
        elevation_delta_m=0,
        same_venue_flag=True,
        same_host_city_flag=True,
    )

    assert same_venue["travel_burden_score"] < different_venue["travel_burden_score"]


def test_short_rest_increases_burden():
    long_rest = score_travel_burden(
        rest_days=5,
        distance_from_previous_km=0,
        timezone_delta_hours_value=0,
        elevation_delta_m=0,
        same_venue_flag=False,
        same_host_city_flag=False,
    )
    short_rest = score_travel_burden(
        rest_days=3,
        distance_from_previous_km=0,
        timezone_delta_hours_value=0,
        elevation_delta_m=0,
        same_venue_flag=False,
        same_host_city_flag=False,
    )

    assert short_rest["rest_days_penalty"] > long_rest["rest_days_penalty"]
    assert short_rest["travel_burden_score"] > long_rest["travel_burden_score"]


def test_timezone_delta_is_calculated_correctly():
    delta = timezone_delta_hours("America/Mexico_City", "America/New_York", "2026-06-18")

    assert delta == 2


def test_elevation_delta_penalty_is_directional():
    upward = score_travel_burden(
        rest_days=5,
        distance_from_previous_km=0,
        timezone_delta_hours_value=0,
        elevation_delta_m=1500,
        same_venue_flag=False,
        same_host_city_flag=False,
    )
    downward = score_travel_burden(
        rest_days=5,
        distance_from_previous_km=0,
        timezone_delta_hours_value=0,
        elevation_delta_m=-1500,
        same_venue_flag=False,
        same_host_city_flag=False,
    )

    assert upward["elevation_penalty"] > downward["elevation_penalty"]
