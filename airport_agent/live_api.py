"""Live airport status from the FAA NAS Status public API (no key needed).

The feed lists airports with active FAA programs right now: ground stops, ground delay programs,
arrival/departure delays and closure notices. It is a snapshot of the current moment and is never
used in the score. Standard library only (urllib + xml).
"""
import urllib.request
import xml.etree.ElementTree as ET

FAA_URL = "https://nasstatus.faa.gov/api/airport-status-information"
TIMEOUT_SECONDS = 10


def fetch_nas_status() -> ET.Element:
    """Download and parse the FAA feed. Raises on network/HTTP/XML errors."""
    request = urllib.request.Request(FAA_URL, headers={"User-Agent": "airport-investment-agent"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return ET.fromstring(response.read())


def _flatten(element: ET.Element) -> dict:
    """All leaf texts and attributes under an event element, e.g. {'Reason': ..., 'Max': ...}."""
    out = {}
    for node in element.iter():
        if node is element:
            continue
        for key, value in node.attrib.items():
            out[f"{node.tag}_{key}"] = value
        text = (node.text or "").strip()
        if text and len(node) == 0:
            out[node.tag] = text
    return out


def parse_airport_events(root: ET.Element) -> dict:
    """Map airport code -> list of active events. Airspace-only programs (no airport) are skipped."""
    events = {}
    for delay_type in root.iter("Delay_type"):
        type_name = (delay_type.findtext("Name") or "Unknown").strip()
        for element in delay_type.iter():
            code = element.findtext("ARPT")
            if element.find("ARPT") is None or not code:
                continue
            details = _flatten(element)
            details.pop("ARPT", None)
            events.setdefault(code.strip().upper(), []).append({"type": type_name, **details})
    return events


def update_time(root: ET.Element) -> str:
    return (root.findtext("Update_Time") or "unknown").strip()
