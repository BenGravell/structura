import streamlit as st


class DesignMonitor:
    def __init__(self) -> None:
        self.num_metrics = 2
        self.metric_columns = st.columns(self.num_metrics)
        self.metric_empties = []
        for i in range(self.num_metrics):
            with self.metric_columns[i]:
                self.metric_empties.append(st.empty())
        self.image_empty = st.empty()

    def update(self, frame, objective_value, loop_count):
        self.metric_empties[0].metric(
            "Total Objective", round(objective_value, 3) if objective_value is not None else None
        )
        self.metric_empties[1].metric("Iteration", loop_count)
        self.image_empty.image(frame, use_column_width=True)
