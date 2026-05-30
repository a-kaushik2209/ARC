from datetime import datetime, timezone
from typing import Any, Dict, Optional

class RecoveryEventTracker:
    def __init__(self):
        self.events = [] 

    def log_event(
       self,
       step: int,
       event: str,
       metadata: Optional[Dict[str, Any]] = None
    ):
       if metadata is None:
           metadata ={}
 
       self.events.append({
            "timestamp":
    datetime.now(timezone.utc).isoformat(),
         "step": step,
         "event": event,
         "metadata": metadata
       })
       
    def get_events(self):
                return self.events.copy()