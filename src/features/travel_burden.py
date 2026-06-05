"""Travel / recovery burden scoring helpers.

The score is a transparent context metric for consecutive matches. It is not a
causal estimate and should not be applied as a hard prediction penalty without
a separate controlled backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from zoneinfo import ZoneInfo


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class TravelBurdenConfig:
    """Configurable conservative weights for travel / recovery burden."""

    rest_4_days: float = 0.15
    rest_3_days: float = 0.35
    rest_2_or_less_days: float = 0.55
    distance_750_km: float = 0.10
    distance_1500_km: float = 0.20
    distance_2500_km: float = 0.35
    distance_over_2500_km: float = 0.50
    timezone_1_hour: float = 0.05
    timezone_2_hours: float = 0.12
    timezone_3_plus_hours: float = 0.20
    elevation_500_m: float = 0.05
    elevation_1000_m: float = 0.12
    elevation_1500_plus_m: float = 0.20
    upward_elevation_multiplier: float = 1.25
    same_city_bonus: float = -0.08
    same_venue_bonus: float = -0.15
    max_score: float = 2.0


def haversine_km(
    previous_latitude: float,
    previous_longitude: float,
    latitude: float,
    longitude: float,
) -> float:
    """Return great-circle distance in kilometres."""

    previous_lat = radians(float(previous_latitude))
    current_lat = radians(float(latitude))
    delta_lat = radians(float(latitude) - float(previous_latitude))
    delta_lon = radians(float(longitude) - float(previous_longitude))

    a = sin(delta_lat / 2) ** 2 + cos(previous_lat) * cos(current_lat) * sin(delta_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return EARTH_RADIUS_KM * c


def timezone_delta_hours(
    previous_timezone: str,
    timezone: str,
    match_date: str | datetime,
) -> float:
    """Return current UTC offset minus previous UTC offset in hours."""

    when = datetime.fromisoformat(str(match_date)[:10]).replace(hour=12)
    previous_offset = ZoneInfo(previous_timezone).utcoffset(when)
    current_offset = ZoneInfo(timezone).utcoffset(when)
    if previous_offset is None or current_offset is None:
        raise ValueError("Could not calculate UTC offset for one or both time zones")
    return (current_offset - previous_offset).total_seconds() / 3600


def rest_days_penalty(rest_days: float | int | None, config: TravelBurdenConfig | None = None) -> float:
    """Penalty for short recovery between matches."""

    cfg = config or TravelBurdenConfig()
    if rest_days is None:
        return 0.0
    rest = float(rest_days)
    if rest >= 5:
        return 0.0
    if rest >= 4:
        return cfg.rest_4_days
    if rest >= 3:
        return cfg.rest_3_days
    return cfg.rest_2_or_less_days


def distance_penalty(distance_km: float | int | None, config: TravelBurdenConfig | None = None) -> float:
    """Penalty by travel-distance bucket."""

    cfg = config or TravelBurdenConfig()
    if distance_km is None:
        return 0.0
    distance = float(distance_km)
    if distance <= 100:
        return 0.0
    if distance <= 750:
        return cfg.distance_750_km
    if distance <= 1500:
        return cfg.distance_1500_km
    if distance <= 2500:
        return cfg.distance_2500_km
    return cfg.distance_over_2500_km


def timezone_penalty(delta_hours: float | int | None, config: TravelBurdenConfig | None = None) -> float:
    """Penalty by absolute time-zone change."""

    cfg = config or TravelBurdenConfig()
    if delta_hours is None:
        return 0.0
    delta = abs(float(delta_hours))
    if delta < 1:
        return 0.0
    if delta < 2:
        return cfg.timezone_1_hour
    if delta < 3:
        return cfg.timezone_2_hours
    return cfg.timezone_3_plus_hours


def elevation_penalty(
    elevation_delta_m: float | int | None,
    config: TravelBurdenConfig | None = None,
) -> float:
    """Penalty by elevation shift, with a small multiplier for upward moves."""

    cfg = config or TravelBurdenConfig()
    if elevation_delta_m is None:
        return 0.0
    delta = float(elevation_delta_m)
    abs_delta = abs(delta)
    if abs_delta < 500:
        base = 0.0
    elif abs_delta < 1000:
        base = cfg.elevation_500_m
    elif abs_delta < 1500:
        base = cfg.elevation_1000_m
    else:
        base = cfg.elevation_1500_plus_m
    if delta > 0:
        base *= cfg.upward_elevation_multiplier
    return base


def location_bonus(
    same_venue_flag: bool,
    same_host_city_flag: bool,
    config: TravelBurdenConfig | None = None,
) -> float:
    """Return a negative component for repeated venue or city."""

    cfg = config or TravelBurdenConfig()
    if same_venue_flag:
        return cfg.same_venue_bonus
    if same_host_city_flag:
        return cfg.same_city_bonus
    return 0.0


def score_travel_burden(
    *,
    rest_days: float | int | None,
    distance_from_previous_km: float | int | None,
    timezone_delta_hours_value: float | int | None,
    elevation_delta_m: float | int | None,
    same_venue_flag: bool,
    same_host_city_flag: bool,
    config: TravelBurdenConfig | None = None,
) -> dict[str, float]:
    """Return score and components for a consecutive-match transition."""

    cfg = config or TravelBurdenConfig()
    components = {
        "rest_days_penalty": rest_days_penalty(rest_days, cfg),
        "distance_penalty": distance_penalty(distance_from_previous_km, cfg),
        "timezone_penalty": timezone_penalty(timezone_delta_hours_value, cfg),
        "elevation_penalty": elevation_penalty(elevation_delta_m, cfg),
        "same_location_bonus": location_bonus(same_venue_flag, same_host_city_flag, cfg),
    }
    raw_score = sum(components.values())
    components["travel_burden_score"] = min(max(raw_score, 0.0), cfg.max_score)
    return components
