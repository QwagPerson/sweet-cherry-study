from src.thermal_models.chill_hours import chill_hours_7_2, chill_hours_10
from src.thermal_models.dynamic_model import dynamic_portions_proxy
from src.thermal_models.florida import florida_units
from src.thermal_models.growing_degree_hours import growing_degree_hours
from src.thermal_models.north_carolina import north_carolina_units
from src.thermal_models.utah import utah_units


THERMAL_PAIRS = {
    "chill_7_2__gdh_4_5": (chill_hours_7_2, growing_degree_hours),
    "chill_10__gdh_4_5": (chill_hours_10, growing_degree_hours),
    "utah__gdh_4_5": (utah_units, growing_degree_hours),
    "north_carolina__gdh_4_5": (north_carolina_units, growing_degree_hours),
    "florida__gdh_4_5": (florida_units, growing_degree_hours),
    "dynamic_proxy__gdh_4_5": (dynamic_portions_proxy, growing_degree_hours),
}
