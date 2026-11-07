import urllib.request
import json

SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

def test():
    query = """[out:json][timeout:25];
    (
      way["highway"~"primary|secondary|trunk"](13.010,80.220,13.060,80.270);
    );
    out body;
    >;
    out skel qt;
    """
    for server in SERVERS:
        print(f"Testing {server}...")
        try:
            req = urllib.request.Request(
                server,
                data=query.encode("utf-8"),
                headers={"User-Agent": "FloodSenseAI-Capstone/4.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print(f"  SUCCESS! Elements: {len(data.get('elements', []))}")
                return server
        except Exception as e:
            print(f"  FAILED: {e}")
    return None

if __name__ == "__main__":
    test()
