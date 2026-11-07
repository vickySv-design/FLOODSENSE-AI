import urllib.request
import json
import ssl

context = ssl._create_unverified_context()

def test_api():
    # Direct OSM API for a small bbox: left,bottom,right,top = min_lon,min_lat,max_lon,max_lat
    # Chennai sample: 80.23, 13.02, 80.25, 13.04
    url = "https://api.openstreetmap.org/api/0.6/map.json?bbox=80.23,13.02,80.25,13.04"
    req = urllib.request.Request(url, headers={"User-Agent": "FloodSenseAI-Capstone/4.0"})
    print("Testing direct OSM 0.6 API...")
    try:
        with urllib.request.urlopen(req, context=context, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("OSM 0.6 SUCCESS! Elements:", len(data.get("elements", [])))
            ways = [e for e in data.get("elements", []) if e.get("type") == "way" and "highway" in e.get("tags", {})]
            print("Highway ways:", len(ways))
            blds = [e for e in data.get("elements", []) if e.get("type") == "way" and "building" in e.get("tags", {})]
            print("Building ways:", len(blds))
    except Exception as e:
        print("OSM 0.6 FAILED:", e)

if __name__ == "__main__":
    test_api()

