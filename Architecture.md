# Multi-City Architecture
# Polymarket Temperature Trading Bot - Multiple Cities Support

## 📋 Executive Summary

**Goal:** Transform the current Munich-only bot into a flexible, multi-city architecture that supports:
- Different weather data sources (Wunderground, Open-Meteo, etc.)
- Different geographical locations (lat/lon, timezone, airport codes)
- Different predictive features (city-specific vs generic)
- Different optimal thresholds per city
- Multiple cities running simultaneously or independently

---

## 🎯 Design Principles

### 1. **Separation of Concerns**
- **Domain Logic** (trading) ← independent of weather sources
- **Weather Abstractions** ← independent of specific cities
- **City-Specific Configurations** ← isolated in config files
- **Data Persistence** ← unified CSV format

### 2. **Dependency Inversion**
- Core bot doesn't import specific city modules directly
- City modules injected via configuration
- Easy to add new cities without modifying core

### 3. **Generic + Specific**
- **Generic components** work for ALL cities
- **City-specific** only where absolutely necessary (e.g., Foehn wind for Munich)

### 4. **Configuration-Driven**
- All city parameters in JSON/YAML config files
- No hardcoded lat/lon, timezone, API endpoints
- Easy to add/modify cities

---

## 🏗️ Proposed Structure

```
polyweather/
├── core/                          # CORE COMPONENTS (CITY-AGNOSTIC)
│   ├── weather.py               # Abstract weather interface
│   │   └── class Weather(ABC)
│   │       ├── fetch_latest()
│   │       ├── fetch_forecast_max()
│   │       └── bootstrap_today()
│   │
│   ├── models.py                # Generic model interface
│   │   ├── load_models(city)
│   │   ├── build_features(weather_data, city_config)
│   │   └── predict_ensemble(weather_data, city_config)
│   │
│   ├── features.py              # Feature builders
│   │   ├── build_generic_features()  # Works for ALL cities
│   │   ├── build_munich_features()  # Munich-specific (V2)
│   │   ├── build_atlanta_features() # Atlanta-specific (UV importance)
│   │   └── build_athens_features()   # Athens-specific (Mediterranean)
│   │
│   ├── entry.py                 # Entry logic (unchanged)
│   │   └── SingleEntry, PhasedEntry
│   │
│   ├── market.py                 # Polymarket integration (unchanged)
│   │   ├── ClobClient
│   │   ├── fetch_market()
│   │   └── compute_ev()
│   │
│   ├── display.py                # Terminal UI (unchanged)
│   │   └── log_tick(), display()
│   │
│ ├── polymarket/               # Polymarket integration (existing)
│   │   ├── clob.py
│   │   └── orders.py
│   │
│   └── utils/                    # Helpers (unchanged)
│       ├── time.py
│       └── ansi.py
│
├── cities/                         # CITY-SPECIFIC IMPLEMENTATIONS
│   ├── city_base.py            # Base city class
│   │   └── class BaseCity(ABC)
│   │       ├── load_config(city_id)
│   │       └── get_features_class()
│   │
│   ├── munich/                  # Munich implementation
│   │   ├── config.json
│   │   ├── weather.py
│   │   └── features.py
│   │
│   ├── atlanta/                 # Atlanta implementation
│   │   ├── config.json
│   │   ├── weather.py
│   │   └── features.py
│   │
│   └── athens/                 # Athens implementation
│       ├── config.json
│       ├── weather.py
│       └── features.py
│
├── bot/                            # TRADING BOT
│   ├── live_bot.py            # Main bot (refactored)
│   │   └── Uses core/ + cities/
│   │
│   ├── train.py                # Training (refactored)
│   │   └── Uses core/ + cities/
│   │
│   └── backtester.py            # Backtesting (refactored)
│       └── Uses core/ + cities/
│
├── historic/                       # HISTORICAL DATA
│   ├── munich.csv
│   ├── atlanta.csv
│   ├── athens.csv
│   └── ...
│
├── models/                          # TRAINED MODELS
│   ├── munich/
│   │   ├── lgbm_peak.pkl
│   │   ├── xgb_peak.pkl
│   │   └── peak_model_config.json
│   ├── atlanta/
│   │   └── ...
│   └── athens/
│       └── ...
│
└── configs/                         # GLOBAL CONFIG
    └── bot_config.json          # Default city, active cities, etc.
```

---

## 📝 City Configuration Format

### `cities/munich/config.json`
```json
{
  "city_id": "munich",
  "name": "Munich",
  "timezone": "Europe/Berlin",
  "latitude": 48.35,
  "longitude": 11.78,
  "airport_code": "EDDM",

  "weather": {
    "sources": ["wunderground", "openmeteo"],
    "wu_api_key": "${WU_API_KEY}",
    "wu_station": "EDDM",
    "wu_url": "https://api.weather.com/v1/location/EDDM/observations/historical.json",

    "om_station": "berlin",

    "forecast_sources": ["wunderground", "openmeteo"],
    "wu_forecast_url": "https://api.weather.com/v1/forecast",
    "om_forecast_url": "https://api.open-meteo.com/v1/forecast"
  },

  "data": {
    "historical_csv": "historic/munich.csv",
    "csv_has_v2_features": true,

    "peak_hour_mean": 14.5,     // Average hour of daily peak
    "peak_hour_std": 1.8,      // Standard deviation
    "peak_temp_mean": 15.5,    // Average peak temp
    "peak_temp_std": 5.2,       // Standard deviation

    "date_range": "2010-01-01:2026-04-12"
  },

  "model": {
    "fixed_threshold": 0.80,      // Backtested optimal value
    "adaptive_threshold": false,       // Disable adaptive (39-45% range is broken)

    "feature_override": {
      "use_v2_features": true,     // Use Foehn, UV, etc.
      "uv_importance": 0.1,         // Weight for UV feature
      "foehn_importance": 0.15        // Weight for Foehn feature
    },

    "polymarket": {
      "market_slug_pattern": "highest-temperature-in-{month}-{day}-{year}",
      "default_bet_size": 15.0,
      "default_threshold": 0.80
    }
  },

  "training": {
    "model_path": "models/munich",
    "target_column": "temp_c",
    "positive_labels": ["YES", "true", "1"],

    "ensemble_weights": {
      "lgbm": 0.50,
      "xgb": 0.30,
      "zscore": 0.20
    },

    "model_params": {
      "lgbm": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31
      },
      "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_leaves": 31
      }
    }
  }
}
```

### `cities/atlanta/config.json`
```json
{
  "city_id": "atlanta",
  "name": "Atlanta",
  "timezone": "America/New_York",
  "latitude": 33.75,
  "longitude": -84.39,
  "airport_code": "KATL",

  "weather": {
    "sources": ["openmeteo"],  // WU may not have good Atlanta data
    "wu_api_key": "${WU_API_KEY}",
    "wu_station": "KATL",

    "om_station": "atlanta",

    "forecast_sources": ["openmeteo"],
    "om_forecast_url": "https://api.open-meteo.com/v1/forecast"
  },

  "data": {
    "historical_csv": "historic/atlanta.csv",
    "csv_has_v2_features": true,

    "peak_hour_mean": 15.5,
    "peak_hour_std": 1.8,
    "peak_temp_mean": 32.0,
    "peak_temp_std": 6.5,

    "date_range": "2010-01-01:2026-04-12"
  },

  "model": {
    "fixed_threshold": 0.75,      // Different optimal threshold for Atlanta
    "adaptive_threshold": false,
    "enable_uv_feature": true,  // UV is VERY important in Atlanta (subtropical)
    "enable_humidity_features": false, // Less predictive in humid climate
  },

  "polymarket": {
    "market_slug_pattern": "highest-temperature-in-{month}-{day}-{year}",
    "default_bet_size": 15.0,
    "default_threshold": 0.75
  }
}
```

### `cities/athens/config.json`
```json
{
  "city_id": "athens",
  "name": "Athens",
  "timezone": "Europe/Athens",
  "latitude": 37.98,
  "longitude": 23.73,
  "airport_code": "LGAV",

  "weather": {
    "sources": ["openmeteo", "wunderground"],  // Both available
    "wu_api_key": "${WU_API_KEY}",
    "wu_station": "LGAV",

    "om_station": "athens",

    "forecast_sources": ["openmeteo", "wunderground"],
    "wu_forecast_url": "https://api.weather.com/v1/forecast",
    "om_forecast_url": "https://api.open-meteo.com/v1/forecast"
  },

  "data": {
    "historical_csv": "historic/athens.csv",
    "csv_has_v2_features": false,  // No V2 features in data
    "use_openmeteo_center": true,  // Better station choice
  },

  "peak_hour_mean": 16.5,
    "peak_hour_std": 2.0,
    "peak_temp_mean": 30.5,
    "peak_temp_std": 7.5,

    "date_range": "2010-01-01:2026-04-12"
  },

  "model": {
    "fixed_threshold": 0.78,
    "adaptive_threshold": false,
    "disable_pressure_features": true,  // Less predictive
    "enable_humidity_features": true,  // Important in Mediterranean climate
    "enable_sea_breeze_features": true  // Meltemi wind influence
  },

  "polymarket": {
    "market_slug_pattern": "highest-temperature-in-{month}-{day}-{year}",
    "default_bet_size": 15.0,
    "default_threshold": 0.78
  }
}
```

---

## 🔧 Core Interfaces

### `core/weather.py`
```python
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Optional

class Weather(ABC):
    """Abstract interface for weather data sources."""

    @abstractmethod
    def fetch_latest(self) -> Optional[Dict]:
        """Fetch latest observation from weather source."""

    @abstractmethod
    def fetch_forecast_max(self) -> Optional[float]:
        """Fetch maximum temperature forecast for today."""

    @abstractmethod
    def bootstrap_today(self) -> tuple[List[Dict], Dict]:
        """Bootstrap today's historical data for model initialization."""
```

### `core/models.py`
```python
from typing import Dict, List
import pandas as pd

def predict_ensemble(
    weather_data: Dict,
    city_config: Dict,
    models: Dict
) -> Dict:
    """
    Generic ensemble prediction function.

    Args:
        weather_data: Latest observation and historical data
        city_config: City configuration with feature overrides
        models: Loaded model objects

    Returns:
        Dict with p_ensemble, p_lgbm, p_xgb, p_zscore, etc.
    """
    pass
```

### `core/features.py`
```python
def build_generic_features(weather_data, city_config) -> pd.DataFrame:
    """Build generic features (work for ALL cities)."""
    # Features: slot_frac, doy_sin, doy_cos, temp_c, running_max,
    #           temp_vs_climatology, delta_30m, delta_1h, accel,
    #           recent_slope, temp_lag_3, roll3_std, plateau_indicator,
    #           morning_max, radiation_proxy, humidity_drop_1h, prev_7d_avg_max
    pass
```

### `cities/city_base.py`
```python
from abc import ABC, abstractmethod

class BaseCity(ABC):
    """Base class for all city implementations."""

    def __init__(self, city_id: str):
        self.city_id = city_id
        self.config = self.load_config()

    @abstractmethod
    def get_weather(self) -> Weather:
        """Return weather implementation for this city."""

    @abstractmethod
    def get_features_class(self) -> object:
        """Return features builder class for this city."""

    @abstractmethod
    def get_model_path(self) -> Path:
        """Return path to city's trained models."""
```

---

## 🚀 Migration Strategy

### Phase 1: Refactor Core (1-2 days)
1. Create `core/` directory structure
2. Implement abstract `Weather` interface
3. Implement generic `predict_ensemble()` in `core/models.py`
4. Implement generic features in `core/features.py`
5. Create `city_base.py` with `BaseCity` abstract class
6. Refactor `polymarket/` and `utils/` to remove city hardcodes

### Phase 2: Implement Cities (1 day per city)
1. Create `cities/munich/` with config and implementation
2. **Keep Munich working** as reference implementation
3. Add Atlanta (simpler, Open-Meteo only, UV focus)
4. Add Athens (Mediterranean focus, skip irrelevant features)

### Phase 3: Refactor Bot (2-3 days)
1. Create new `bot/live_bot.py` that uses city config
2. Add `--city` argument to select city
3. Update `bot/train.py` to work with city config
4. Update `bot/backtester.py` to work with city config

### Phase 4: Testing & Validation (2-3 days)
1. Test each city individually
2. Test multi-city mode (all cities running)
3. Backtest each city
4. Verify model performance per city

---

## 📊 Feature Strategy

### Generic Features (ALL cities - 18 features)
```python
["slot_frac", "doy_sin", "doy_cos", "temp_c", "running_max",
 "temp_vs_climatology", "delta_30m", "delta_1h", "accel",
 "recent_slope", "temp_lag_3", "roll3_std", "plateau_indicator",
 "morning_max", "radiation_proxy", "humidity_drop_1h", "prev_7d_avg_max",
 "seasonal_peak_prior"]
```

### City-Specific Features (V2)

**Munich (7 features):**
```python
["dewpoint_c", "temp_to_dewpoint_gap", "pressure_trend_3h",
 "wind_south_proxy", "wind_speed_kmh", "uv_index", "foehn_indicator"]
```

**Atlanta (5 features):**
```python
["uv_index", "pressure_trend_3h", "wind_speed_kmh", "humidity_drop_1h",
 "thunderstorm_indicator"]  # Important for subtropical thunderstorms
```

**Athens (5 features):**
```python
["humidity_drop_1h", "pressure_trend_3h", "sea_breeze_proxy",
 "land_breeze_proxy", "humidity_drop_6h"]  # Mediterranean sea breezes
```

### Feature Selection Logic
```python
def get_features_for_city(city_id: str) -> List[str]:
    config = load_city_config(city_id)

    generic = GENERIC_FEATURES  # Always included

    city_specific = []
    if config.get("use_v2_features", True):
        city_specific = CITY_V2_FEATURES.get(city_id, [])

    return generic + city_specific
```

---

## 🎯 Command Line Interface

### Multi-city bot:
```bash
# Run single city
python bot/live_bot.py --city munich --mode single --run paper
python bot/live_bot.py --city atlanta --mode single --run paper

# Run multiple cities
python bot/live_bot.py --city munich,atlanta,athens --mode single --run paper

# Train specific city
python bot/train.py --city munich
python bot/train.py --city atlanta --no-xgb  # XGB not needed for Atlanta

# Backtest all cities
python bot/backtester.py --all-cities
python bot/backtester.py --city munich --year 2023,2024
```

---

## ⚙️ Configuration Management

### `configs/bot_config.json`
```json
{
  "default_city": "munich",
  "active_cities": ["munich"],  // Expandable to multiple
  "telegram": {
    "enabled": true,
    "chat_id": "${TELEGRAM_CHAT_ID}"
  },
  "polymarket": {
    "private_key": "${POLY_PRIVATE_KEY}",
    "max_daily_loss": 50.0
  }
}
```

---

## 🔒️ Backward Compatibility

During refactoring, maintain compatibility:
1. **Keep old `munich_*.py` files** working (don't delete)
2. New `bot/live_bot.py` uses new architecture
3. Old files deprecated but not removed until migration complete
4. Clear deprecation warnings in documentation

---

## 📈 Scalability

This architecture supports:
- ✅ Adding new cities without modifying core logic
- ✅ Different data sources per city
- ✅ Different optimal thresholds per city
- ✅ City-specific feature sets
- ✅ Independent model training per city
- ✅ Multi-city parallel execution (future)
- ✅ A/B testing different configurations per city

---

## 🎓 Next Steps

1. Review and refine Architecture.md
2. Start Phase 1 refactoring
3. Test Munich functionality after each phase
4. Add cities one by one
5. Full regression testing before merging to main
