import matplotlib.pyplot as plt


def plot_recovery_timeline(events, save_path="recovery_timeline.png"):
    
    """Plot recovery events on a simple timeline."""
    

    if not events:
        print("No recovery events found.")
        return

    event_types = {
        "failure_detected": 1,
        "rollback_triggered": 2,
        "checkpoint_restored": 3,
        "lr_reduced": 4,
        "step_skipped": 5,
    }

    x = []
    y = []
    labels = []

    for event in events:
        event_name = event.get("event")
        step = event.get("step")

        if event_name in event_types:
            x.append(step)
            y.append(event_types[event_name])
            labels.append(event_name)
        elif event_name is not None:
            print(f"Warning: Unknown event type'{event_name}' skipped")

    plt.figure(figsize=(10, 4))
    plt.scatter(x, y)

    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]))

    plt.yticks(
        list(event_types.values()),
        list(event_types.keys())
    )

    plt.xlabel("Training Step")
    plt.ylabel("Recovery Event")
    plt.title("Recovery Timeline")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Timeline saved to {save_path}")