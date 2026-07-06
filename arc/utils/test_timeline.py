from arc.utils.recovery_tracker import RecoveryEventTracker
from arc.utils.recovery_timeline import plot_recovery_timeline

def main():
  """
  Demo script for the recovery event tracker and timeline plotter.
       Creates sample recovery events and generates a timeline plot.
       Run with: python -m arc.utils.test_timeline
       """
  
  tracker = RecoveryEventTracker()

  # sample events
  tracker.log_event(1, "rollback_triggered")
  tracker.log_event(2, "lr_reduced")
  tracker.log_event(3, "checkpoint_restored")
  tracker.log_event(4, "checkpoint_restored")

  print("Recovery Timeline Events:")

  for event in tracker.get_events():
    print(event)

  plot_recovery_timeline(tracker.get_events())

if __name__ == "__main__":
  main()