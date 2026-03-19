import os
import multiprocessing
import pandas as pd
from pathlib import Path
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Checkbox,
    Input,
    Button,
    RichLog,
    ProgressBar, Label, Select, Digits
)
from textual.containers import Horizontal, Vertical, VerticalScroll
from src.utils.data_lab import generate_distribution_plot, generate_scatter_plot
from src.utils.launcher import run_episode_task


class RlPlayground(App):
    """A Textual app to test RL agents"""

    CSS_PATH = "utils/playground.tcss"

    def __init__(self):
        super().__init__()
        # Application variables
        self.theme = "tokyo-night"
        self.config_path = None
        self.model_path = None
        self.render_mode = None
        self.num_trials = 1
        self.generate_csv_log = False
        self.generate_chart_report = False

        # Path stuff
        self.project_root = Path(__file__).parent.parent.resolve()
        self.config_dir = self.project_root / "config"
        self.models_dir = self.project_root / "models"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Left config sidebar
        with VerticalScroll(id="sidebar"):
            yield Label("Configuration", classes="section-title")

            yield Label("Config File:")
            yield Select(
                [(str(p.name), str(p)) for p in self.config_dir.glob("*.yaml")],
                prompt="Select Config...",
                id="config_selector"
            )

            yield Label("Model File:", classes="section-title")
            model_options = [(str(p.name), str(p)) for p in self.models_dir.glob("*.pth")]
            model_options.append(("RANDOM POLICY", "RANDOM"))
            yield Select(
                model_options,
                prompt="Select Model...",
                id="model_selector"
            )

            yield Label("Render Mode:", classes="section-title")
            yield Select(
                [("None", "No Rendering"), ("Human", "Human Rendering"), ("Video", "Video Rendering")],
                value="No Rendering",
                allow_blank=False,
                id="render_selector"
            )

            yield Label("Trials:", classes="section-title")
            yield Input(placeholder="1", value="1", type="integer", id="trial_input")

            yield Label("Options:", classes="section-title")
            yield Checkbox("Save CSV Log", id="csv_log")
            yield Checkbox("Generate Charts", id="chart_report")

            yield Button("Run Experiment", id="run_button", variant="success")

        # Right output area
        with Vertical(id="main_content"):
            with Horizontal(id="stat_display"):
                with Vertical():
                    yield Label("Last Reward")
                    yield Digits("0.00", id="score_display")
                with Vertical():
                    yield Label("Avg Reward")
                    yield Digits("0.00", id="avg_display")

            with Horizontal(classes="status_row"):
                yield Label("Progress: ", id="progress_label")
                yield ProgressBar(total=100, id="progress_bar")

            yield RichLog(id="debug_output", markup=True, wrap=True, highlight=True)

        yield Footer()

    @on(Button.Pressed, "#run_button")
    def start_experiments(self) -> None:
        """Launches the trials when Run button is pressed"""

        self.config_path = self.query_one("#config_selector", Select).value
        self.model_path = self.query_one("#model_selector", Select).value
        self.render_mode = self.query_one("#render_selector", Select).value

        trial_val = self.query_one("#trial_input", Input).value
        self.num_trials = int(trial_val) if trial_val and trial_val.isdigit() else 1

        self.generate_csv_log = self.query_one("#csv_log", Checkbox).value
        self.generate_chart_report = self.query_one("#chart_report", Checkbox).value

        if self.config_path == Select.BLANK or self.model_path == Select.BLANK or self.render_mode == Select.BLANK:
            self.log_message("[bold red]Error:[/bold red] Please select Config, Model, and Render Mode.")
            return

        self.query_one("#run_button", Button).disabled = True
        self.query_one("#progress_bar", ProgressBar).update(progress=0, total=self.num_trials)
        self.clear_log()
        self.log_summary()

        self.run_worker(self.run_trials, exclusive=True, thread=True)

    def run_trials(self):
        """Sets up the whole init-trial-report data process"""
        try:
            results_df = self._execute_trial_loop()

            if not results_df.empty:
                self._process_results(results_df)
            else:
                self.call_later(self.log_message, "[bold red]No results collected due to errors.[/bold red]")
        except Exception as e:
            self.call_later(self.log_message, f"[bold red]Critical Error:[/bold red] {e}")
        finally:
            self.call_later(self.log_message, "[bold blue]Trials finished[/bold blue]")
            self.call_later(lambda: setattr(self.query_one("#run_button", Button), "disabled", False))

    def _execute_trial_loop(self) -> pd.DataFrame:
        """Executes the trial loop in separate process to prevent OpenGL shenanigans."""
        data = pd.DataFrame(columns=["Trial", "Reward", "Steps"])
        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(processes=1) as pool:
            for i in range(self.num_trials):
                try:
                    result = pool.apply(
                        run_episode_task,
                        args=(self.config_path, self.model_path, self.render_mode)
                    )

                    if "error" in result:
                        self.call_later(self.log_message, f"[bold red]Trial {i} Error:[/bold red] {result['error']}")
                        continue

                    reward = result["reward"]
                    steps = result["steps"]

                    data.loc[len(data)] = {"Trial": i + 1, "Reward": reward, "Steps": steps}

                    self._log_trial_status(i, reward, data["Reward"].mean())

                except Exception as e:
                    self.call_later(self.log_message, f"[bold red]Process Error:[/bold red] {e}")

                self.call_later(lambda: self.query_one("#progress_bar", ProgressBar).advance(1))

        return data

    def _process_results(self, df: pd.DataFrame):
        """Processes the results DataFrame to log summary statistics and generate reports."""
        avg_reward = df["Reward"].mean()

        self.call_later(lambda: self.query_one("#avg_display", Digits).update(f"{avg_reward:.2f}"))
        self.call_later(self.log_message, f"[bold magenta]Average Reward:[/bold magenta] {avg_reward:.2f}")
        self.call_later(self.log_message, f"[bold magenta]Reward Std Dev:[/bold magenta] {df['Reward'].std():.2f}")

        output_dir = self.project_root / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = self.model_path.rsplit(os.sep, maxsplit=1)[-1]

        if self.generate_chart_report:
            self.call_later(generate_distribution_plot, df, run_name, output_dir)
            self.call_later(generate_scatter_plot, df, run_name, output_dir)

        if self.generate_csv_log:
            csv_path = str(output_dir / f"{run_name}_run.csv")
            df.to_csv(csv_path)

    def _log_trial_status(self, i, last_reward, avg_reward):
        """Helper to log single trial result to GUI."""
        self.call_later(lambda: self.query_one("#score_display", Digits).update(f"{last_reward:.2f}"))

        avg_widget = self.query_one("#avg_display", Digits)
        previous_avg = float(avg_widget.value)
        avg_widget.styles.color = "#44FF44" if avg_reward > previous_avg else "#FF4444"
        self.call_later(lambda: self.query_one("#avg_display", Digits).update(f"{avg_reward:.2f}"))

        self.call_later(
            self.log_message,
            f"[bold yellow]Trial {i + 1} Reward:[/bold yellow] {last_reward:.2f}",
        )

    def log_summary(self):
        """Thread-safe method for writing trial summary"""
        debug_output = self.query_one("#debug_output", RichLog)
        debug_output.clear()
        self.query_one("#score_display", Digits).update("0.00")
        self.query_one("#avg_display", Digits).update("0.00")

        debug_output.write(
            f"[bold green]Configuration File:[/bold green] {self.config_path}"
        )
        debug_output.write(f"[bold green]Model File:[/bold green] {self.model_path}")
        debug_output.write(f"[bold green]Render Mode:[/bold green] {self.render_mode}")
        debug_output.write(
            f"[bold green]Number of Trials:[/bold green] {self.num_trials}"
        )
        debug_output.write(
            f"[bold green]Generate CSV Log:[/bold green] {self.generate_csv_log}"
        )
        debug_output.write(
            f"[bold green]Generate Chart Report:[/bold green] {self.generate_chart_report}"
        )
        debug_output.write("[bold blue]Running trials...[/bold blue]")

    def log_message(self, text: str):
        """Thread-safe method for logging outputs"""
        self.query_one("#debug_output", RichLog).write(text)

    def clear_log(self):
        """Thread-safe method for clearing log"""
        self.query_one("#debug_output", RichLog).clear()


if __name__ == "__main__":
    app = RlPlayground()
    app.run()
