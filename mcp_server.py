import os
import requests
import base64
import io
import math
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image
from math import radians, sin, cos, asin, sqrt
from dotenv import load_dotenv

# ====== Load env ======
load_dotenv()
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
PORT = int(os.getenv("PORT", "8086"))
if not AUTH_TOKEN:
    raise RuntimeError("AUTH_TOKEN missing. Put it in .env")
MY_NUMBER = os.getenv("MY_NUMBER", "").strip()

# ====== FastAPI app ======
app = FastAPI(title="Puch MCP Server", docs_url=None, redoc_url=None)

# ====== Simple in-memory storage ======
# TRIPS per puch_user_id: { "user123": [ {trip}, {trip2} ] }
TRIPS: dict[str, list[dict]] = {}

# ====== JSON-RPC helpers ======
class JSONRPCRequest(BaseModel):
    jsonrpc: str
    method: str
    id: Optional[Any] = None
    params: Optional[Dict[str, Any]] = None

def rpc_result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}

def rpc_error(id_, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}

# ====== Auth ======
def check_bearer_auth(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth.split(" ", 1)[1].strip()
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

# ====== Base tools (echo, image bw) ======
def tool_echo_text(params: Dict[str, Any]) -> Dict[str, Any]:
    text = (params or {}).get("text")
    if text is None:
        raise ValueError("Missing required param: text")
    return {"text": text}

def tool_image_bw(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    image_base64: base64-encoded image (png/jpg). Returns base64 PNG.
    """
    b64 = (params or {}).get("image_base64")
    if not b64:
        raise ValueError("Missing required param: image_base64")
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("L")  # grayscale
        out = io.BytesIO()
        img.save(out, format="PNG")
        out_b64 = base64.b64encode(out.getvalue()).decode("utf-8")
        return {"image_base64": out_b64}
    except Exception as e:
        raise ValueError(f"Invalid image_base64: {e}")

# ====== Puch validation tool ======
def tool_validate(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the registered WhatsApp number in {country_code}{number} format.
    Example: 919998881729
    """
    if not MY_NUMBER or not MY_NUMBER.isdigit():
        raise ValueError("Server misconfigured: set MY_NUMBER env as digits like 919998881729")
    return {"number": MY_NUMBER}


# ====== Trip Planner tools ======

# Reverse geocode via OpenStreetMap (to label map-picked points)
def reverse_geocode(lat: float, lon: float) -> dict:
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "jsonv2"}
    r = requests.get(url, params=params, headers={"User-Agent": "puch-mcp/1.0"}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return {"display_name": data.get("display_name", f"{lat:.5f},{lon:.5f}")}

# Geocode via OpenStreetMap (no API key)
def geocode_place(name: str) -> dict:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": name, "format": "json", "limit": 1}
    r = requests.get(url, params=params, headers={"User-Agent": "puch-mcp/1.0"}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"Could not geocode '{name}'")
    hit = data[0]
    return {"lat": float(hit["lat"]), "lon": float(hit["lon"]), "display_name": hit["display_name"]}

def tool_geocode(params: Dict[str, Any]) -> Dict[str, Any]:
    q = (params or {}).get("query")
    if not q:
        raise ValueError("Missing required param: query")
    return geocode_place(q)

# Route via OSRM public demo
def osrm_route(coords: list[dict]) -> dict:
    if len(coords) < 2:
        raise ValueError("Need at least two coordinates for routing")
    # OSRM expects lon,lat order
    path = ";".join([f'{c["lon"]},{c["lat"]}' for c in coords])
    url = f"https://router.project-osrm.org/route/v1/driving/{path}"
    params = {"overview": "false", "geometries": "geojson"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(f"OSRM routing failed: {data.get('message')}")
    route = data["routes"][0]
    # distance in meters, duration in seconds
    return {"distance_m": route["distance"], "duration_s": route["duration"]}

def tool_route_plan(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    stops: list of strings (place names/addresses) OR list of {lat,lon}
    """
    stops = (params or {}).get("stops")
    if not isinstance(stops, list) or len(stops) < 2:
        raise ValueError("Provide 'stops' as a list of 2+ places.")
    coords = []
    for s in stops:
        if isinstance(s, dict) and "lat" in s and "lon" in s:
            coords.append({"lat": float(s["lat"]), "lon": float(s["lon"])})
        elif isinstance(s, str):
            g = geocode_place(s)
            coords.append({"lat": g["lat"], "lon": g["lon"], "display_name": g["display_name"]})
        else:
            raise ValueError("Each stop must be a place string or {lat,lon}.")
    rt = osrm_route(coords)
    return {"stops": coords, **rt}

def tool_cost_estimate(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inputs:
      distance_m (number) OR stops (list -> compute distance)
      mode: "car"|"bike"|"bus" [optional]
      fuel_price_per_litre [optional] (default 105 INR)
      vehicle_mileage_kmpl [optional] (defaults by mode)
      tolls_total [optional]
      food_per_person [optional]
      misc_total [optional]
      people_count [required]
    """
    p = params or {}
    people = int(p.get("people_count", 0))
    if people <= 0:
        raise ValueError("people_count must be > 0")

    if "distance_m" in p:
        distance_m = float(p["distance_m"])
    elif "stops" in p:
        route = tool_route_plan({"stops": p["stops"]})
        distance_m = route["distance_m"]
    else:
        raise ValueError("Provide distance_m or stops.")

    distance_km = distance_m / 1000.0
    mode = p.get("mode", "car")
    defaults = {"car": 14.0, "bike": 35.0, "bus": 4.0}
    mileage = float(p.get("vehicle_mileage_kmpl", defaults.get(mode, 14.0)))
    fuel_price = float(p.get("fuel_price_per_litre", 105.0))

    fuel_litres = distance_km / max(mileage, 0.1)
    fuel_cost = fuel_litres * fuel_price

    tolls_total = float(p.get("tolls_total", 0.0))
    food_per_person = float(p.get("food_per_person", 0.0))
    misc_total = float(p.get("misc_total", 0.0))

    total = fuel_cost + tolls_total + misc_total + (food_per_person * people)
    per_person = total / people

    return {
        "distance_km": round(distance_km, 2),
        "fuel_litres": round(fuel_litres, 2),
        "fuel_cost": round(fuel_cost, 2),
        "tolls_total": round(tolls_total, 2),
        "food_total": round(food_per_person * people, 2),
        "misc_total": round(misc_total, 2),
        "people_count": people,
        "total_cost": round(total, 2),
        "cost_per_person": round(per_person, 2)
    }

def tool_split_bill(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    total (required)
    people: list of names [optional]
    weights: list of numbers same length as people [optional]
    OR count: number of unnamed people
    """
    p = params or {}
    total = float(p.get("total", 0))
    if total <= 0:
        raise ValueError("total must be > 0")

    people = p.get("people", [])
    weights = p.get("weights", [])

    if not people:
        n = int(p.get("count", 0))
        if n <= 0:
            raise ValueError("Provide people[] or count.")
        share = round(total / n, 2)
        return {"split": [{"name": f"Person {i+1}", "amount": share} for i in range(n)]}

    n = len(people)
    if weights and len(weights) != n:
        raise ValueError("weights length must match people length")

    if not weights:
        share = round(total / n, 2)
        return {"split": [{"name": people[i], "amount": share} for i in range(n)]}

    wsum = sum([float(w) for w in weights]) or 1.0
    split = []
    for name, w in zip(people, weights):
        amt = round(total * (float(w) / wsum), 2)
        split.append({"name": name, "amount": amt})
    return {"split": split}

def tool_save_trip(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    puch_user_id (required)
    trip: { name, stops, cost_summary, notes? }
    """
    p = params or {}
    uid = p.get("puch_user_id")
    trip = p.get("trip")
    if not uid or not trip:
        raise ValueError("puch_user_id and trip are required")
    TRIPS.setdefault(uid, []).append(trip)
    return {"ok": True, "saved_count": len(TRIPS[uid])}

def tool_list_trips(params: Dict[str, Any]) -> Dict[str, Any]:
    uid = (params or {}).get("puch_user_id")
    if not uid:
        raise ValueError("puch_user_id is required")
    return {"trips": TRIPS.get(uid, [])}

# ====== AI Trip Planner (Gemini-powered) ======
import json

# Load Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

@app.post("/api/ai_plan")
async def api_ai_plan(req: Request):
    body = await req.json()
    start = (body.get("from") or body.get("origin") or body.get("start") or "").strip()
    end = (body.get("to") or body.get("destination") or body.get("end") or "").strip()
    people = int(body.get("people", 1))
    requested_mode = (body.get("mode") or "").lower().strip()  # 'train'|'flight'|'cab'
    days = int(body.get("days", 2))
    season = (body.get("season") or "auto").lower()

    if not start or not end:
        return JSONResponse({"detail": "Enter both From and To"}, status_code=400)
    if requested_mode not in ("train","flight","cab"):
        return JSONResponse({"detail": "Choose a transport mode: train, flight or cab"}, status_code=400)
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_KEY_HERE":
        return JSONResponse({"detail": "Server is missing GEMINI_API_KEY in .env"}, status_code=503)

    # Ask Gemini for a single-mode plan + itinerary (season & duration aware, includes legs & safety)
    start_date = (body.get("start_date") or "").strip()
    end_date = (body.get("end_date") or "").strip()

    prompt = (
        "You are a travel cost planner for India. Respond ONLY with JSON (no prose, no code blocks).\n"
        "Use realistic mid-range Indian prices in INR.\n"
        "Schema: {\n"
        "  \"mode\": {\"name\": 'train|flight|cab', \"cost_total_inr\": number, \"duration_hours\": number, \"summary\": string},\n"
        "  \"segments\": [ {\"name\": string, \"notes\": string} ],\n"
        "  \"itinerary\": [ {\"name\": string, \"approx_cost_inr\": number, \"note\": string, \"nearby_food\": string} ],\n"
        "  \"stay_area\": string,\n"
        "  \"season_tips\": string,\n"
        "  \"weather_safety\": string\n"
        "}\n"
        f"Trip from {start} to {end} for {people} people.\n"
        f"Transport mode: {requested_mode}. Plan for {days} day(s). Season: {season}.\n"
        f"Dates (optional): {start_date or 'unspecified'} to {end_date or 'unspecified'}.\n"
        "Break the door-to-door flow as segments (e.g., cab to airport, flight, airport to hotel).\n"
        "For each attraction in itinerary, include one \"nearby_food\" string (a well-known cafe/bar near that spot).\n"
        "Return city highlights (famous places/monuments/attractions), suggested stay area (central & safe),\n"
        "itemised itinerary costs, and brief weather_safety advice based on season/date. Keep numbers rounded."
    )

    payload = {"contents": [ { "parts": [ {"text": prompt} ] } ]}

    try:
        resp = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60,
        )
        if resp.status_code != 200:
            return JSONResponse({"detail": f"Gemini error {resp.status_code}", "raw": resp.text}, status_code=resp.status_code)
        res_json = resp.json()
        candidates = res_json.get("candidates") or []
        if not candidates:
            return JSONResponse({"detail": "Gemini returned no candidates", "raw": res_json}, status_code=502)
        ai_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        if not ai_text:
            return JSONResponse({"detail": "Empty response from Gemini"}, status_code=502)

        def strip_fences(s: str) -> str:
            s = s.strip()
            if s.startswith("```"):
                lines = s.splitlines()[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                s = "\n".join(lines)
            return s.strip()

        cleaned = strip_fences(ai_text)
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return JSONResponse({"detail": "AI did not return valid JSON", "raw": ai_text}, status_code=502)

        mode_block = parsed.get("mode") or {}
        segments = parsed.get("segments", []) or []
        itinerary = parsed.get("itinerary", []) or []
        stay_area = parsed.get("stay_area", "")
        season_tips = parsed.get("season_tips", "")
        weather_safety = parsed.get("weather_safety", "")

        # compute itinerary total
        try:
            itin_total = sum(float(x.get("approx_cost_inr", 0) or 0) for x in itinerary)
        except Exception:
            itin_total = 0.0

        # keep response shape the UI expects (modes dict), but only fill the selected one
        modes = {"train": None, "flight": None, "cab": None}
        modes[requested_mode] = {
            "cost_total_inr": mode_block.get("cost_total_inr", 0),
            "duration_hours": mode_block.get("duration_hours", 0),
            "summary": mode_block.get("summary", ""),
        }
        modes = {k:v for k,v in modes.items() if v is not None}

        out = {
            "origin": start,
            "destination": end,
            "people": people,
            "days": days,
            "season": season,
            "start_date": start_date,
            "end_date": end_date,
            "modes": modes,
            "segments": segments,
            "itinerary": itinerary,
            "itinerary_total_inr": round(itin_total, 2),
            "stay_area": stay_area,
            "season_tips": season_tips,
            "weather_safety": weather_safety,
        }
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"detail": f"Server error: {e}"}, status_code=500)

# ====== Live fares stub endpoint ======
@app.post("/api/live_fares")
async def api_live_fares(req: Request):
    """Placeholder for live airline/train fares. Returns a helpful message if not configured."""
    return JSONResponse({
        "detail": "Live pricing providers are not configured. Add your provider credentials and implement here.",
        "providers": []
    })
@app.post("/api/split_bill_json")
async def api_split_bill_json(req: Request):
    body = await req.json()
    total = float(body.get("total", 0))
    people = body.get("people") or []
    count = int(body.get("count", 0))
    args = {"total": total}
    if people:
        args["people"] = [str(p) for p in people]
    else:
        args["count"] = count
    return JSONResponse(tool_split_bill(args))

# ====== Tool registry ======
TOOLS = {
    "echo_text": {
        "name": "echo_text",
        "description": "Echo back the given text",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo"}
            },
            "required": ["text"]
        },
        "handler": tool_echo_text,
    },
    "image_bw": {
        "name": "image_bw",
        "description": "Convert an input image (base64) to black & white PNG (base64)",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_base64": {"type": "string", "description": "Base64-encoded image"}
            },
            "required": ["image_base64"]
        },
        "handler": tool_image_bw,
    },
    "validate": {
        "name": "validate",
        "description": "Return the server's registered WhatsApp number for Puch validation.",
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_validate,
    },
    "geocode": {
        "name": "geocode",
        "description": "Geocode a place string to lat/lon using OSM Nominatim",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        },
        "handler": tool_geocode,
    },
    "route_plan": {
        "name": "route_plan",
        "description": "Compute route distance/time for given stops (strings or {lat,lon}).",
        "input_schema": {
            "type": "object",
            "properties": {
                "stops": {"type": "array", "items": {"anyOf":[{"type":"string"},{"type":"object"}]}}
            },
            "required": ["stops"]
        },
        "handler": tool_route_plan,
    },
    "cost_estimate": {
        "name": "cost_estimate",
        "description": "Estimate total and per-person cost for a trip.",
        "input_schema": {
            "type": "object",
            "properties": {
                "distance_m": {"type": "number"},
                "stops": {"type": "array"},
                "mode": {"type": "string"},
                "fuel_price_per_litre": {"type": "number"},
                "vehicle_mileage_kmpl": {"type": "number"},
                "tolls_total": {"type": "number"},
                "food_per_person": {"type": "number"},
                "misc_total": {"type": "number"},
                "people_count": {"type": "integer"}
            },
            "required": ["people_count"]
        },
        "handler": tool_cost_estimate,
    },
    "split_bill": {
        "name": "split_bill",
        "description": "Split a total among people equally or by weights.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "total": {"type":"number"},
                    "people": {"type":"array", "items":{"type":"string"}},
                    "weights": {"type":"array", "items":{"type":"number"}},
                    "count": {"type":"integer"}
                },
                "required": ["total"]
        },
        "handler": tool_split_bill,
    },
    "save_trip": {
        "name": "save_trip",
        "description": "Save a trip under a user's namespace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "puch_user_id": {"type":"string"},
                "trip": {"type":"object"}
            },
            "required": ["puch_user_id","trip"]
        },
        "handler": tool_save_trip,
    },
    "list_trips": {
        "name": "list_trips",
        "description": "List saved trips for a user.",
        "input_schema": {
            "type": "object",
            "properties": {"puch_user_id": {"type":"string"}},
            "required": ["puch_user_id"]
        },
        "handler": tool_list_trips,
    },
}

#
# ====== HTTP helpers for new UI ======
@app.post("/api/route_plan_geo")
async def api_route_plan_geo(req: Request):
    body = await req.json()
    return JSONResponse(tool_route_plan({"stops": body.get("stops", [])}) | {"geometry": (await _route_geometry(body.get("stops", [])))})

async def _route_geometry(stops):
    # internal helper to fetch OSRM geometry for map rendering
    if not isinstance(stops, list) or len(stops) < 2:
        return None
    coords = []
    for s in stops:
        if isinstance(s, dict) and "lat" in s and "lon" in s:
            coords.append({"lat": float(s["lat"]), "lon": float(s["lon"])})
        elif isinstance(s, str):
            g = geocode_place(s)
            coords.append({"lat": g["lat"], "lon": g["lon"]})
        else:
            raise HTTPException(status_code=400, detail="stop must be string or {lat,lon}")
    path = ";".join([f"{c['lon']},{c['lat']}" for c in coords])
    url = f"https://router.project-osrm.org/route/v1/driving/{path}"
    r = requests.get(url, params={"overview":"full","geometries":"geojson"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise HTTPException(status_code=502, detail=f"OSRM failed: {data.get('message')}")
    return data["routes"][0]["geometry"]["coordinates"]

@app.post("/api/air_distance")
async def api_air_distance(req: Request):
    body = await req.json()
    a, b = body.get("a"), body.get("b")
    if not a or not b:
        raise HTTPException(status_code=400, detail="a and b required")
    def to_ll(x):
        if isinstance(x, dict) and "lat" in x and "lon" in x:
            return float(x["lat"]), float(x["lon"])
        if isinstance(x, str):
            g = geocode_place(x)
            return g["lat"], g["lon"]
        raise HTTPException(status_code=400, detail="a/b must be string or {lat,lon}")
    lat1, lon1 = to_ll(a)
    lat2, lon2 = to_ll(b)
    R = 6371.0088
    from math import radians, sin, cos, asin, sqrt
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    aa = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    c = 2*asin(sqrt(aa))
    km = R*c
    return JSONResponse({"distance_km": round(km, 2), "a": {"lat": lat1, "lon": lon1}, "b": {"lat": lat2, "lon": lon2}})


@app.post("/api/revgeo")
async def api_revgeo(req: Request):
    body = await req.json()
    lat, lon = float(body.get("lat")), float(body.get("lon"))
    return JSONResponse(reverse_geocode(lat, lon))

# ====== Fetch POIs along route ======
@app.post("/api/nearby_along")
async def api_nearby_along(req: Request):
    body = await req.json()
    geometry = body.get("geometry")  # list of [lon,lat]
    amenity = body.get("amenity", "fuel")  # can be regex like "restaurant|fast_food|cafe"
    radius = int(body.get("radius_m", 800))
    if not geometry or not isinstance(geometry, list):
        raise HTTPException(status_code=400, detail="geometry (list of [lon,lat]) required")

    # sample up to ~20 points evenly along the geometry
    step = max(len(geometry) // 20, 1)
    samples = [geometry[i] for i in range(0, len(geometry), step)]

    # amenity selector (exact or regex)
    amenity_sel = f'["amenity"~"{amenity}"]' if '|' in amenity else f'["amenity"="{amenity}"]'

    # build Overpass query aggregating all samples to minimize requests
    parts = []
    for lon, lat in samples:
        parts.append(f'node{amenity_sel}(around:{radius},{lat},{lon});')
        parts.append(f'way{amenity_sel}(around:{radius},{lat},{lon});')
    q = f"[out:json][timeout:40];({''.join(parts)});out center 200;"

    try:
        r = requests.post("https://overpass-api.de/api/interpreter", data=q, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Overpass error: {e}")

    # Deduplicate by rounded lat/lon
    seen = set()
    pois = []
    for el in data.get("elements", []):
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        key = (round(lat, 5), round(lon, 5))
        if key in seen: 
            continue
        seen.add(key)
        pois.append({
            "lat": lat,
            "lon": lon,
            "name": el.get("tags", {}).get("name", ""),
            "amenity": el.get("tags", {}).get("amenity", "")
        })

    return JSONResponse({"count": len(pois), "pois": pois[:300]})

@app.get("/healthz")
async def healthz():
    return {"ok": True}

# ====== MCP JSON-RPC endpoint ======
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    check_bearer_auth(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if isinstance(payload, list):
        return JSONResponse([handle_rpc_obj(obj) for obj in payload])
    return JSONResponse(handle_rpc_obj(payload))

@app.get("/map", response_class=HTMLResponse)
async def map_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>Trip Planner</title>
  <script src=\"https://unpkg.com/htmx.org@2.0.2\"></script>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"/>
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <link href=\"https://cdn.jsdelivr.net/npm/tailwindcss@3.4.4/dist/tailwind.min.css\" rel=\"stylesheet\">
  <style>
    body{background: radial-gradient(1200px 600px at 10% 0%, #0f172a 0%, #020617 55%, #000 100%); color:#e2e8f0;}
    .glass{backdrop-filter: blur(12px); background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1);} 
    .neon{box-shadow: 0 0 25px rgba(99,102,241,.35), inset 0 0 8px rgba(34,197,94,.25)}
    .btn{padding:.7rem 1rem;border-radius:14px;font-weight:700}
    .btn-primary{background:linear-gradient(135deg,#22c55e,#06b6d4); color:#031b1d}
    .btn-ghost{border:1px solid rgba(255,255,255,.25)}
    .pill{border-radius:12px;padding:.55rem .9rem;border:1px solid rgba(255,255,255,.18); background:rgba(17,24,39,.55); color:#e5e7eb}
    .pill::placeholder{ color:#94a3b8 }
    select.pill option{ color:#000 }
    label{font-size:.8rem; opacity:.85}
    .badge{font-size:.8rem; padding:.3rem .6rem; border-radius:9999px; background:rgba(2,6,23,.65); border:1px solid rgba(148,163,184,.25)}
    .map-controls{position:absolute; right:14px; top:14px; display:flex; flex-direction:column; gap:8px; z-index:1000}
    .map-panel{position:absolute; left:14px; bottom:14px; z-index:1000}
    .range {appearance:none; height:6px; border-radius:9999px; background:linear-gradient(90deg,#22c55e, #06b6d4);} 
    .range::-webkit-slider-thumb{appearance:none; width:18px; height:18px; border-radius:50%; background:#e2e8f0; border:2px solid #0f172a}
  </style>
</head>
<body class=\"min-h-screen\">
  <div class=\"grid grid-cols-1 lg:grid-cols-3 gap-4 p-4\">
    <div class=\"relative col-span-2 rounded-2xl glass neon overflow-hidden\" style=\"height:80vh;\">
      <div id=\"map\" style=\"height:100%;\"></div>
      <div class=\"map-controls\">
        <button class=\"btn btn-primary\" onclick=\"plan()\">Plan Route</button>
        <button class=\"btn btn-ghost\" onclick=\"estimate()\">Estimate Cost</button>
        <button class=\"btn btn-ghost\" onclick=\"togglePOI('fuel')\">⛽ Petrol pumps</button>
        <button class=\"btn btn-ghost\" onclick=\"togglePOI('restaurant')\">🍽️ Restaurants</button>
      </div>
      <div class=\"map-panel\">
        <div id=\"summary\" class=\"glass badge\">Ready</div>
      </div>
    </div>

    <div class=\"glass rounded-2xl p-4 space-y-5\">
      <h1 class=\"text-xl font-extrabold\">Trip Planner</h1>

      <!-- Step 1 -->
      <div id=\"step1\" class=\"space-y-3\">
        <div class=\"text-sm opacity-80\">Step 1 of 3</div>
        <div class=\"grid grid-cols-2 gap-3\">
          <div class=\"col-span-2 font-semibold\">Where are you going?</div>
          <div><label>From</label><input id=\"from\" class=\"w-full pill\" placeholder=\"Connaught Place, Delhi\"></div>
          <div><label>To</label><input id=\"to\" class=\"w-full pill\" placeholder=\"Taj Mahal, Agra\"></div>
          <div class=\"col-span-2\"><button class=\"btn btn-ghost w-full\" onclick=\"pick('from')\">Set From on Map</button></div>
          <div class=\"col-span-2\"><button class=\"btn btn-ghost w-full\" onclick=\"pick('to')\">Set To on Map</button></div>
        </div>
      </div>

      <!-- Step 2 -->
      <div id=\"step2\" class=\"space-y-3\">
        <div class=\"text-sm opacity-80\">Step 2 of 3</div>
        <div class=\"font-semibold\">Travel details</div>
        <div class=\"grid grid-cols-2 gap-3\">
          <div><label>People</label><input id=\"people\" type=\"number\" value=\"4\" class=\"w-full pill\"></div>
          <div><label>Mode</label><select id=\"mode\" class=\"w-full pill\"><option>car</option><option>bike</option><option>bus</option></select></div>
          <div><label>Fuel ₹/L</label><input id=\"fuel\" type=\"number\" value=\"105\" class=\"w-full pill\"></div>
          <div><label>Mileage km/L</label><input id=\"mileage\" type=\"number\" value=\"14\" class=\"w-full pill\"></div>
          <div><label>Tolls ₹</label><input id=\"tolls\" type=\"number\" value=\"400\" class=\"w-full pill\"></div>
          <div><label>Food / person ₹</label><input id=\"food\" type=\"number\" value=\"250\" class=\"w-full pill\"></div>
          <div class=\"col-span-2\"><label>Misc total ₹</label><input id=\"misc\" type=\"number\" value=\"200\" class=\"w-full pill\"></div>
        </div>
        <div class=\"mt-2\">
          <label>Your budget ₹</label>
          <input id=\"budget\" type=\"range\" min=\"500\" max=\"20000\" step=\"100\" value=\"5000\" class=\"w-full range\" oninput=\"updateBudgetLabel()\">
          <div class=\"flex justify-between text-xs opacity-70\"><span>₹500</span><span id=\"budgetLabel\">₹5000</span><span>₹20k</span></div>
        </div>
      </div>

      <!-- Step 3 -->
      <div id=\"step3\" class=\"space-y-3\">
        <div class=\"text-sm opacity-80\">Step 3 of 3</div>
        <div class=\"font-semibold\">Group split</div>
        <input id=\"names\" class=\"w-full pill\" placeholder=\"Sam,Aayush,Riya,Meera\">
        <button class=\"btn btn-primary w-full\" onclick=\"split()\">Split Bill</button>
      </div>

      <pre id=\"out\" class=\"text-xs p-3 glass rounded-xl overflow-auto\" style=\"max-height:30vh;\"></pre>
      <div class=\"text-xs opacity-60\">Powered by OSM/OSRM · MCP server side · <a href=\"/healthz\">health</a></div>
    </div>
  </div>

<script>
let map = L.map('map',{zoomControl:false}).setView([22.6,79.9],5);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OSM'}).addTo(map);
let routeLayer = L.geoJSON().addTo(map);
let airLayer = L.polyline([], {dashArray:'6 6'}).addTo(map);
let poiLayer = L.layerGroup().addTo(map);
let fromMarker=null, toMarker=null, picking=null, lastRoute=null;

function updateBudgetLabel(){ document.getElementById('budgetLabel').innerText = '₹'+document.getElementById('budget').value; }
updateBudgetLabel();

function pick(which){ picking = which; alert('Click on the map to set '+which.toUpperCase()); }

map.on('click', async (e)=>{
  if(!picking) return;
  const {lat,lng} = e.latlng;
  if(picking==='from'){
    if(fromMarker) map.removeLayer(fromMarker);
    fromMarker = L.marker([lat,lng],{draggable:true}).addTo(map).bindPopup('From').openPopup();
    fromMarker.on('dragend', onDrag);
  }else{
    if(toMarker) map.removeLayer(toMarker);
    toMarker = L.marker([lat,lng],{draggable:true}).addTo(map).bindPopup('To').openPopup();
    toMarker.on('dragend', onDrag);
  }
  const label = await (await fetch('/api/revgeo',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({lat,lon:lng})})).json();
  document.getElementById(picking).value = label.display_name;
  picking=null;
});

async function onDrag(ev){
  const {lat,lng} = ev.target.getLatLng();
  const which = (ev.target===fromMarker)?'from':'to';
  const label = await (await fetch('/api/revgeo',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({lat,lon:lng})})).json();
  document.getElementById(which).value = label.display_name;
}

async function plan(){
  const a = document.getElementById('from').value.trim();
  const b = document.getElementById('to').value.trim();
  if(!a||!b){ alert('Enter From and To (or set on map)'); return; }
  const r = await fetch('/api/route_plan_geo',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({stops:[a,b]})});
  const data = await r.json();
  routeLayer.clearLayers();
  if(data.geometry){
    const coords = data.geometry.map(([lon,lat])=>[lat,lon]);
    routeLayer.addData({type:'LineString', coordinates:data.geometry});
    map.fitBounds(L.polyline(coords).getBounds(),{padding:[30,30]});
  }
  lastRoute = data;
  const air = await (await fetch('/api/air_distance',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({a:a,b:b})})).json();
  airLayer.setLatLngs([[air.a.lat,air.a.lon],[air.b.lat,air.b.lon]]);
  document.getElementById('summary').textContent = `Distance: ${(data.distance_m/1000).toFixed(1)} km · Drive: ${(data.duration_s/60).toFixed(0)} min · Air: ${air.distance_km} km`;
}

async function estimate(){
  if(!lastRoute){ alert('Plan a route first'); return; }
  const body = {
    people_count: Number(document.getElementById('people').value),
    mode: document.getElementById('mode').value,
    fuel_price_per_litre: Number(document.getElementById('fuel').value),
    vehicle_mileage_kmpl: Number(document.getElementById('mileage').value),
    tolls_total: Number(document.getElementById('tolls').value),
    food_per_person: Number(document.getElementById('food').value),
    misc_total: Number(document.getElementById('misc').value),
    distance_m: lastRoute?.distance_m
  };
  const r = await fetch('/api/cost_estimate',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await r.json();
  const budget = Number(document.getElementById('budget').value);
  const verdict = (data.total_cost <= budget) ? '✅ within budget' : '⚠️ over budget';
  document.getElementById('summary').textContent = `Total: ₹${data.total_cost.toFixed(0)} · /person: ₹${data.cost_per_person.toFixed(0)} · ${verdict}`;
  document.getElementById('out').textContent = JSON.stringify({...data, budget, verdict}, null, 2);
}

let showing = {fuel:false, restaurant:false};
async function togglePOI(kind){
  showing[kind] = !showing[kind];
  if(!lastRoute){ alert('Plan a route first'); return;}
  poiLayer.clearLayers();
  if(!showing[kind]) return;
  // fetch POIs *along* the route; for restaurants, include fast_food & cafe
  const amenity = (kind==='restaurant') ? 'restaurant|fast_food|cafe' : 'fuel';
  const r = await fetch('/api/nearby_along',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({geometry: lastRoute.geometry || [], amenity, radius_m: 800})});
  const data = await r.json();
  data.pois.forEach(p=> L.circleMarker([p.lat,p.lon],{radius:6,opacity:.95}).bindTooltip(p.name || kind).addTo(poiLayer));
}
</script>
</body>
</html>
    """


# ====== New landing page and chat/splitter UIs ======

@app.get("/", response_class=HTMLResponse)
async def landing():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>WanderSplit — Plan & Split</title>
  <link href='https://cdn.jsdelivr.net/npm/tailwindcss@3.4.4/dist/tailwind.min.css' rel='stylesheet'>
  <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
  <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
  <style>
    body{background: radial-gradient(1200px 600px at 10% 0%, #0b1220 0%, #050914 55%, #000 100%); color:#e6edf3}
    .glass{backdrop-filter:blur(12px);background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1)}
    .pill{border-radius:10px;padding:.55rem .9rem;background:rgba(15,23,42,.7);border:1px solid rgba(148,163,184,.35);color:#e6edf3}
    .pill::placeholder{color:#9fb0c2}
    .card{background:rgba(2,6,23,.55);border:1px solid rgba(148,163,184,.22)}
    .btn{border-radius:12px;padding:.6rem 1rem;font-weight:800}
    .btn-cta{background:linear-gradient(135deg,#22c55e,#06b6d4);color:#06221f}
    .tag{font-size:.75rem;padding:.25rem .6rem;border-radius:9999px;background:rgba(3,7,18,.6);border:1px solid rgba(148,163,184,.25)}
    .reveal{transition:opacity .5s ease, transform .5s ease}
  </style>
</head>
<body class='min-h-screen'>
  <div class='max-w-5xl mx-auto p-6 space-y-8'>
    <!-- Hero title -->
    <div class='text-center space-y-2'>
      <h1 class='text-4xl md:text-5xl font-extrabold tracking-tight'>WanderSplit</h1>
      <div class='text-sm md:text-base opacity-80'>Tell us where, we’ll price it, plan it, and split it — fast.</div>
    </div>

    <!-- Centered “chat-like” input card -->
    <div class='max-w-2xl mx-auto glass rounded-2xl p-5 space-y-3'>
      <div class='grid grid-cols-2 gap-3'>
        <select id='mode' class='pill col-span-2'>
          <option value='train'>Train</option>
          <option value='flight'>Flight</option>
          <option value='cab'>Cab / Car</option>
        </select>
        <input id='from' class='pill col-span-2' placeholder='From (city or area)'>
        <div class='col-span-2 flex gap-2'>
          <button class='btn tag' onclick='useLocation()'>📍 Use my location</button>
          <button class='btn tag' onclick='swap()'>⇄ Swap</button>
        </div>
        <input id='to' class='pill col-span-2' placeholder='Destination (e.g. Chennai)'>
        <input id='start_date' type='date' class='pill'>
        <input id='end_date' type='date' class='pill'>
        <input id='people' type='number' value='2' class='pill' placeholder='People'>
        <select id='season' class='pill'>
          <option value='auto'>Season: Auto</option>
          <option>summer</option>
          <option>monsoon</option>
          <option>winter</option>
          <option>spring</option>
        </select>
      </div>
      <button class='btn btn-cta w-full' onclick='ask()'>Try with AI</button>
      <div id='err' class='text-rose-400 text-sm hidden'></div>
    </div>

    <!-- Results section (initially hidden) -->
    <div id='resultsWrapper' class='space-y-4 opacity-0 translate-y-4 reveal hidden'>
      <div class='glass rounded-2xl p-2'>
        <div id='map' style='height:36vh;border-radius:1rem;'></div>
      </div>

      <div class='grid lg:grid-cols-2 gap-4'>
        <div id='results' class='grid gap-4'></div>
        <div class='glass rounded-2xl p-4'>
          <h2 class='font-semibold mb-2'>Door-to-door plan</h2>
          <ul id='segments' class='space-y-2 text-sm'></ul>
        </div>
      </div>

      <div class='glass rounded-2xl p-4'>
        <h2 class='font-semibold mb-2'>Suggested Itinerary</h2>
        <div class='text-sm opacity-80 mb-2'><span id='stayArea' class='tag hidden'></span> <span id='seasonTips' class='opacity-80'></span></div>
        <ul id='itin' class='space-y-2 text-sm'></ul>
        <div id='safety' class='text-xs mt-3 opacity-90'></div>
      </div>

      <div class='glass rounded-2xl p-4'>
        <h2 class='font-semibold mb-2'>Split the trip</h2>
        <div class='grid md:grid-cols-3 gap-3'>
          <input id='split_total' class='pill' placeholder='Total ₹'>
          <input id='split_names' class='pill md:col-span-2' placeholder='Names (comma separated)'>
        </div>
        <div class='mt-2'><button class='btn btn-cta' onclick='splitNow()'>Split Now</button></div>
        <pre id='split_out' class='text-xs p-3 card rounded-xl overflow-auto mt-2' style='max-height:28vh;'></pre>
      </div>
    </div>
  </div>

<script>
let map, routeLayer, airLayer;
function ensureMap(){
  if(map) return; // init once when results appear
  map = L.map('map',{zoomControl:false}).setView([23.3,77.4],5);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OSM'}).addTo(map);
  routeLayer = L.geoJSON().addTo(map);
  airLayer = L.polyline([], {dashArray:'6 6'}).addTo(map);
}

function showErr(msg){ const e=document.getElementById('err'); if(!msg){e.classList.add('hidden'); e.textContent=''} else{e.classList.remove('hidden'); e.textContent=msg} }
function swap(){ const a=document.getElementById('from'), b=document.getElementById('to'); const t=a.value; a.value=b.value; b.value=t; }

async function useLocation(){
  if(!navigator.geolocation){ return showErr('Geolocation not supported'); }
  navigator.geolocation.getCurrentPosition(async (pos)=>{
    try{
      const {latitude, longitude} = pos.coords;
      const label = await (await fetch('/api/revgeo',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({lat:latitude, lon:longitude})})).json();
      document.getElementById('from').value = label.display_name || `${latitude.toFixed(5)},${longitude.toFixed(5)}`;
    }catch(e){ showErr('Failed to use location'); }
  }, (err)=> showErr('Permission denied for location'));
}

async function ask(){
  const from = document.getElementById('from').value.trim();
  const to = document.getElementById('to').value.trim();
  const mode = document.getElementById('mode').value;
  const people = Number(document.getElementById('people').value)||1;
  const season = document.getElementById('season').value;
  const start_date = document.getElementById('start_date').value;
  const end_date = document.getElementById('end_date').value;
  if(!from||!to){ return showErr('Enter From and To'); }
  showErr('');

  // AI plan (people-aware)
  const r = await fetch('/api/ai_plan',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({from,to,mode,people,season,start_date,end_date})});
  const data = await r.json();
  if(data.detail){ return showErr(data.detail); }
  data.people = people; // ensure

  // reveal result section with animation
  ensureMap();
  const wrap = document.getElementById('resultsWrapper');
  wrap.classList.remove('hidden');
  requestAnimationFrame(()=>{ wrap.classList.remove('opacity-0','translate-y-4'); });

  render(data);

  // route + air distance
  try{
    const rr = await fetch('/api/route_plan_geo',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({stops:[from,to]})});
    const route = await rr.json();
    routeLayer.clearLayers();
    if(route.geometry){
      const coords = route.geometry.map(([lon,lat])=>[lat,lon]);
      routeLayer.addData({type:'LineString', coordinates: route.geometry});
      map.fitBounds(L.polyline(coords).getBounds(),{padding:[20,20]});
    }
    const air = await (await fetch('/api/air_distance',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({a:from,b:to})})).json();
    airLayer.setLatLngs([[air.a.lat,air.a.lon],[air.b.lat,air.b.lon]]);
  }catch(e){ console.warn('Map draw failed', e); }
}

function render(data){
  const results=document.getElementById('results'); results.innerHTML='';
  const itinerary=data.itinerary||[]; const itinTotal=data.itinerary_total_inr||0; const modes=data.modes||{}; const ppl = Number(data.people)||1;
  const modeKey = Object.keys(modes)[0]; const m = modes[modeKey]||{};
  const total = (Number(m.cost_total_inr||0) + Number(itinTotal||0));
  const per = Math.round(total / Math.max(ppl,1));

  const card = `<div class='card rounded-xl p-4 flex flex-col gap-2'>
      <div class='text-lg font-bold'>${modeKey? modeKey.toUpperCase(): 'TRAVEL'}</div>
      <div class='text-sm opacity-80'>${m.summary||''}</div>
      <div class='text-sm'>Duration: ${m.duration_hours||'?'} h</div>
      <div class='text-emerald-300 font-bold'>Travel: ₹${m.cost_total_inr||0}</div>
      <div class='text-cyan-300'>Itinerary: ₹${itinTotal}</div>
      <div class='text-white font-semibold'>Grand Total: ₹${total} <span class='opacity-80 text-sm'>(₹${per} per person × ${ppl})</span></div>
      <div class='text-xs opacity-70'>Explanation: we added travel + itinerary, then divided by people count.</div>
    </div>`;
  results.insertAdjacentHTML('beforeend', card);

  const seg=document.getElementById('segments'); seg.innerHTML='';
  (data.segments||[]).forEach(s=> seg.insertAdjacentHTML('beforeend', `<li class='flex gap-2'><span class='tag'>leg</span><span>${s.name}</span><span class='opacity-70'>${s.notes||''}</span></li>`));

  const ul=document.getElementById('itin'); ul.innerHTML='';
  itinerary.forEach(i=> ul.insertAdjacentHTML('beforeend', `<li class='flex justify-between'><span>${i.name}<span class='opacity-60 text-xs'> — ${i.nearby_food||''}</span></span><span class='opacity-80'>₹${i.approx_cost_inr||0}</span></li>`));
  document.getElementById('stayArea').textContent = data.stay_area || '';
  document.getElementById('stayArea').classList.toggle('hidden', !data.stay_area);
  document.getElementById('seasonTips').textContent = data.season_tips || '';
  document.getElementById('safety').textContent = data.weather_safety || '';

  // prefill splitter
  document.getElementById('split_total').value = total;
}

async function splitNow(){
  const total = Number(document.getElementById('split_total').value)||0;
  const namesRaw = document.getElementById('split_names').value.trim();
  const body = { total };
  if(namesRaw){ body.people = namesRaw.split(',').map(s=>s.trim()).filter(Boolean); } else { body.count = 2; }
  const r = await fetch('/api/split_bill_json',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await r.json();
  document.getElementById('split_out').textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
    """

@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Chat Planner</title>
  <link href='https://cdn.jsdelivr.net/npm/tailwindcss@3.4.4/dist/tailwind.min.css' rel='stylesheet'>
  <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
  <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
  <style>
    body{background: radial-gradient(1200px 600px at 10% 0%, #0f172a 0%, #020617 55%, #000 100%); color:#e5e7eb;}
    .glass{backdrop-filter:blur(12px);background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1)}
    .card{background:rgba(2,6,23,.55);border:1px solid rgba(148,163,184,.22)}
    .pill{border-radius:10px;padding:.55rem .9rem;background:rgba(15,23,42,.7);border:1px solid rgba(148,163,184,.35);color:#e5e7eb}
    .pill::placeholder{color:#94a3b8}
  </style>
</head>
<body class='min-h-screen text-slate-200'>
  <div class='max-w-6xl mx-auto p-6 space-y-6'>
    <h1 class='text-2xl font-extrabold'>Chat Mode</h1>

    <div class='grid lg:grid-cols-3 gap-4'>
      <!-- Left: Inputs + Map -->
      <div class='lg:col-span-1 space-y-3'>
        <div class='glass rounded-2xl p-4'>
          <div class='grid grid-cols-3 gap-3'>
            <input id='from' class='col-span-1 pill' placeholder='From e.g. Jaipur'>
            <input id='to' class='col-span-1 pill' placeholder='To e.g. Indore'>
            <input id='people' type='number' value='2' class='col-span-1 pill' placeholder='People'>
          </div>
          <button id='go' class='mt-3 px-4 py-2 rounded-lg bg-gradient-to-r from-emerald-400 to-cyan-400 text-slate-900 font-bold' onclick='ask()'>Ask AI</button>
          <div id='err' class='text-rose-400 text-sm mt-2 hidden'></div>
        </div>
        <div class='glass rounded-2xl p-2'>
          <div id='map' style='height:32vh;border-radius:1rem;'></div>
        </div>
      </div>

      <!-- Right: Results -->
      <div class='lg:col-span-2 space-y-4'>
        <div id='results' class='grid md:grid-cols-3 gap-4'></div>
        <div class='glass rounded-2xl p-4'>
          <h2 class='font-semibold mb-2'>Suggested Itinerary</h2>
          <ul id='itin' class='space-y-2 text-sm'></ul>
        </div>
      </div>
    </div>
  </div>

<script>
let map = L.map('map',{zoomControl:false}).setView([23.3,77.4],5);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OSM'}).addTo(map);
let routeLayer = L.geoJSON().addTo(map);
let airLayer = L.polyline([], {dashArray:'6 6'}).addTo(map);

async function ask(){
  const from = document.getElementById('from').value.trim();
  const to = document.getElementById('to').value.trim();
  const people = Number(document.getElementById('people').value)||1;
  if(!from||!to){ return showErr('Enter From and To'); }
  showErr('');

  // 1) Call Gemini planner
  const r = await fetch('/api/ai_plan',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({from, to, people})});
  const data = await r.json();
  if(data.detail){ return showErr(data.detail); }
  render(data);

  // 2) Draw route + air distance on the inline map
  try{
    const rr = await fetch('/api/route_plan_geo',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({stops:[from,to]})});
    const route = await rr.json();
    routeLayer.clearLayers();
    if(route.geometry){
      const coords = route.geometry.map(([lon,lat])=>[lat,lon]);
      routeLayer.addData({type:'LineString', coordinates: route.geometry});
      map.fitBounds(L.polyline(coords).getBounds(),{padding:[20,20]});
    }
    const air = await (await fetch('/api/air_distance',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({a:from,b:to})})).json();
    airLayer.setLatLngs([[air.a.lat,air.a.lon],[air.b.lat,air.b.lon]]);
  }catch(e){ console.warn('Map draw failed', e); }
}

function showErr(msg){ const e=document.getElementById('err'); if(!msg){e.classList.add('hidden'); e.textContent=''} else{e.classList.remove('hidden'); e.textContent=msg} }

function render(data){
  const results=document.getElementById('results'); results.innerHTML='';
  const modes=data.modes||{}; const itinerary=data.itinerary||[]; const itinTotal=data.itinerary_total_inr||0; const people=data.people||1;
  const cards=[['train','🚆','emerald'],['flight','✈️','cyan'],['cab','🚕','amber']];
  for(const [k,icon,color] of cards){ if(!modes[k]) continue; const m=modes[k];
    const total=(Number(m.cost_total_inr||0)+Number(itinTotal||0));
    const html=`<div class='card rounded-xl p-4 flex flex-col gap-2'>
      <div class='text-lg font-bold'>${icon} ${k.toUpperCase()}</div>
      <div class='text-sm opacity-80'>${m.summary||''}</div>
      <div class='text-sm'>Duration: ${m.duration_hours||'?'} h</div>
      <div class='text-emerald-300 font-bold'>Travel: ₹${m.cost_total_inr||0}</div>
      <div class='text-cyan-300'>Itinerary: ₹${itinTotal}</div>
      <div class='text-white font-semibold'>Grand Total: ₹${total} <span class='opacity-70'>(₹${Math.round(total/people)}/person)</span></div>
      <a class='mt-2 px-3 py-2 rounded-md bg-emerald-400 text-slate-900 font-bold text-center' href='/splitter?total=${encodeURIComponent(total)}&people=${people}'>Send to Splitter</a>
    </div>`; results.insertAdjacentHTML('beforeend', html); }
  const ul=document.getElementById('itin'); ul.innerHTML='';
  itinerary.forEach(i=> ul.insertAdjacentHTML('beforeend', `<li class='flex justify-between'><span>${i.name}</span><span class='opacity-80'>₹${i.approx_cost_inr||0}</span></li>`));
}
</script>
</body>
</html>
    """

@app.get("/splitter", response_class=HTMLResponse)
async def splitter_ui(request: Request):
    from urllib.parse import parse_qs
    # read defaults from query string
    query = parse_qs(str(request.url.query))
    total = query.get('total', [''])[0]
    people = query.get('people', [''])[0]
    return """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Expense Splitter</title>
  <link href='https://cdn.jsdelivr.net/npm/tailwindcss@3.4.4/dist/tailwind.min.css' rel='stylesheet'>
  <style>body{background: radial-gradient(1200px 600px at 10% 0%, #0f172a 0%, #020617 55%, #000 100%);} .glass{backdrop-filter:blur(12px);background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1)}</style>
</head>
<body class='min-h-screen text-slate-200'>
  <div class='max-w-2xl mx-auto p-6 space-y-4'>
    <h1 class='text-2xl font-extrabold'>Expense Splitter</h1>
    <div class='glass rounded-2xl p-4 space-y-3'>
      <div>
        <label class='text-sm'>Total ₹</label>
        <input id='total' class='w-full rounded-md bg-slate-900/40 border border-slate-600 px-3 py-2' value='""" + (str(total) if total else "") + """'>
      </div>
      <div>
        <label class='text-sm'>People (comma separated or count)</label>
        <input id='names' class='w-full rounded-md bg-slate-900/40 border border-slate-600 px-3 py-2' placeholder='Sam,Aya,Lee'>
        <div class='text-xs opacity-70 mt-1'>or Count: <input id='count' type='number' class='bg-slate-900/40 border border-slate-600 px-2 py-1 rounded-md w-20' value='""" + (str(people) if people else "4") + """'></div>
      </div>
      <button class='px-4 py-2 rounded-lg bg-gradient-to-r from-emerald-400 to-cyan-400 text-slate-900 font-bold' onclick='split()'>Split</button>
      <pre id='out' class='text-xs p-3 glass rounded-xl overflow-auto' style='max-height:40vh;'></pre>
    </div>
  </div>
<script>
async function split(){
  const total = Number(document.getElementById('total').value);
  const namesRaw = document.getElementById('names').value.trim();
  const count = Number(document.getElementById('count').value)||0;
  const body = { total };
  if(namesRaw){ body.people = namesRaw.split(',').map(s=>s.trim()).filter(Boolean); } else { body.count = count; }
  const r = await fetch('/api/split_bill',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const data = await r.json();
  document.getElementById('out').textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
    """

# keep last successful route in memory for UI convenience
_last_route = {"distance_m": None}

@app.post("/api/route_plan")
async def api_route_plan(stops: str = Form(...)):
    stop_list = [s.strip() for s in stops.replace(",", ";").split(";") if s.strip()]
    result = tool_route_plan({"stops": stop_list})
    _last_route["distance_m"] = result["distance_m"]
    return JSONResponse(result)

@app.post("/api/cost_estimate")
async def api_cost_estimate(
    people_count: int = Form(...),
    mode: str = Form("car"),
    fuel_price_per_litre: float = Form(105.0),
    vehicle_mileage_kmpl: float = Form(14.0),
    tolls_total: float = Form(0.0),
    food_per_person: float = Form(0.0),
    misc_total: float = Form(0.0),
    stops: str = Form("")
):
    args = {
        "people_count": people_count,
        "mode": mode,
        "fuel_price_per_litre": fuel_price_per_litre,
        "vehicle_mileage_kmpl": vehicle_mileage_kmpl,
        "tolls_total": tolls_total,
        "food_per_person": food_per_person,
        "misc_total": misc_total,
    }
    if stops.strip():
        args["stops"] = [s.strip() for s in stops.replace(",", ";").split(";") if s.strip()]
    elif _last_route["distance_m"]:
        args["distance_m"] = _last_route["distance_m"]
    result = tool_cost_estimate(args)
    return JSONResponse(result)

@app.post("/api/split_bill")
async def api_split_bill(
    total: float = Form(...),
    people: str = Form(""),
    count: int = Form(0)
):
    args = {"total": total}
    if people.strip():
        args["people"] = [p.strip() for p in people.split(",") if p.strip()]
    else:
        args["count"] = count
    result = tool_split_bill(args)
    return JSONResponse(result)

def handle_rpc_obj(obj: Dict[str, Any]):
    try:
        req = JSONRPCRequest(**obj)
    except Exception as e:
        return rpc_error(obj.get("id"), -32600, f"Invalid Request: {e}")

    if req.jsonrpc != "2.0":
        return rpc_error(req.id, -32600, "jsonrpc must be '2.0'")

    try:
        if req.method == "initialize":
            return rpc_result(req.id, {"serverInfo": {"name": "puch-mcp", "version": "0.2.0"}})

        elif req.method == "ping":
            return rpc_result(req.id, {"ok": True})

        elif req.method == "tools/list":
            tools_min = [{
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            } for t in TOOLS.values()]
            return rpc_result(req.id, {"tools": tools_min})

        elif req.method == "tools/call":
            params = req.params or {}
            name = params.get("name")
            arguments = params.get("arguments", {})
            if name not in TOOLS:
                return rpc_error(req.id, -32601, f"Unknown tool '{name}'")
            try:
                out = TOOLS[name]["handler"](arguments)
                return rpc_result(req.id, {"content": out})
            except ValueError as ve:
                return rpc_error(req.id, -32602, str(ve))
            except Exception as e:
                return rpc_error(req.id, -32000, f"Tool error: {e}")

        else:
            return rpc_error(req.id, -32601, "Method not found")

    except HTTPException as he:
        raise he
    except Exception as e:
        return rpc_error(req.id, -32603, f"Internal error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=PORT, reload=False)