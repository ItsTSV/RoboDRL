import os
import numpy as np
import pandas as pd
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    RadioSet,
    RadioButton,
    Markdown,
    Checkbox,
    Input,
    Button,
    RichLog,
    ProgressBar,
)
from textual.containers import VerticalScroll, Horizontal
from textual.validation import Regex
from src.utils.data_lab import generate_distribution_plot, generate_scatter_plot
from src.shared.wandb_wrapper import WandbWrapper
from src.shared.environment_manager import EnvironmentManager
from src.ppo.agent_continuous import PPOAgentContinuous
from src.sac.agent import SACAgent
from src.td3.agent import TD3Agent


class RlPlayground(App):
    """A Textual app to test RL agents"""

    CSS = """
        #debug_output {
            height: 12;       
            border: solid $primary;
            margin-top: 1;
            background: $surface;
        }
        
        .row {
        height: auto;           
        width: 100%;  
        margin-bottom: 1;     
        align: left middle;
        }
        
        Button {
        margin-left: 2;
        }
        
        ProgressBar {
        width: 1fr;           
        margin-left: 2; 
        align: left middle;
        }
        """

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def __init__(self):
        super().__init__()
        # Application variables
        self.config_path = None
        self.model_path = None
        self.render_mode = None
        self.num_trials = 1
        self.generate_csv_log = False
        self.delay_rendering = False
        self.generate_chart_report = False

        # Path stuff
        self.project_root = Path(__file__).parent.parent.resolve()
        self.config_dir = self.project_root / "config"
        self.models_dir = self.project_root / "models"

    def compose(self) -> ComposeResult:
        """Create child widgets for the app"""
        yield Header(show_clock=True)

        with VerticalScroll():
            # Configuration file selector
            yield Markdown("# Select Configuration File")
            with RadioSet(id="config_selector"):
                for config_file_path in self.config_dir.glob("*.yaml"):
                    display_name = str(config_file_path.relative_to(self.project_root))
                    yield RadioButton(display_name)

            # Model selector
            yield Markdown("# Select Model File")
            with RadioSet(id="model_selector"):
                for model_file_path in self.models_dir.glob("*.pth"):
                    display_name = str(model_file_path.relative_to(self.project_root))
                    yield RadioButton(display_name)
                yield RadioButton("RANDOM POLICY")

            # Render settings
            yield Markdown("# Render Settings")
            with RadioSet(id="render_selector"):
                yield RadioButton("No Rendering")
                yield RadioButton("Human Rendering")
                yield RadioButton("Video Rendering")
            yield Checkbox("Delay Rendering", id="delay_rendering_checkbox")

            # Trial runner
            yield Markdown("# How many trials?")
            yield Input(
                placeholder="Enter number of trials",
                id="trial_input",
                validators=[
                    Regex(
                        regex=r"^\d+$", failure_description="Must be a positive integer"
                    )
                ],
            )

            # Additional settings
            yield Markdown("# Additional Settings")
            with Horizontal(classes="row"):
                yield Checkbox("Generate CSV Log", id="csv_log_checkbox")
                yield Checkbox("Generate Chart Report", id="chart_report_checkbox")

            with Horizontal(classes="row"):
                yield Button("Confirm and run", id="run_button", variant="primary")
            with Horizontal(classes="row"):
                yield ProgressBar(show_eta=False)
            yield RichLog(id="debug_output", markup=True, wrap=True)

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press event to run the trials"""
        if event.button.id == "run_button":
            trial_input = self.query_one("#trial_input", Input)
            self.num_trials = (
                int(trial_input.value)
                if trial_input.value and trial_input.is_valid
                else 1
            )
            self.run_worker(self.run_trials, exclusive=True, thread=True)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio set change events to log selections"""
        if event.radio_set.id == "config_selector":
            relative_path_str = str(event.pressed.label)
            self.config_path = str(self.project_root / relative_path_str)
        elif event.radio_set.id == "model_selector":
            relative_path_str = str(event.pressed.label)
            self.model_path = str(self.project_root / relative_path_str)
        elif event.radio_set.id == "render_selector":
            self.render_mode = os.path.normpath(str(event.pressed.label))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox change events to log additional settings"""
        if event.checkbox.id == "csv_log_checkbox":
            self.generate_csv_log = bool(event.value)
        elif event.checkbox.id == "chart_report_checkbox":
            self.generate_chart_report = bool(event.value)
        elif event.checkbox.id == "delay_rendering_checkbox":
            self.delay_rendering = bool(event.value)

    def _update_progress_bar(self) -> None:
        self.query_one(ProgressBar).advance(1)

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode"""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

    def run_trials(self):
        """Sets up the whole init-trial-report data process"""
        self.call_later(self.clear_log)
        self.query_one(ProgressBar).update(progress=0, total=self.num_trials)

        if not self._validate_configuration():
            return

        self.call_later(self.log_summary)
        try:
            wdb = WandbWrapper(self.config_path, mode="disabled")
            env = self._setup_environment(wdb)
            agent = self._setup_agent(env, wdb)
            results_df = self._execute_trial_loop(agent)
            self._process_results(results_df)
        except Exception as e:
            self.call_later(
                self.log_message, f"[bold red]Critical Error:[/bold red] {e}"
            )
        finally:
            env.close()
            wdb.finish()
            self.call_later(self.log_message, "[bold blue]Trials finished[/bold blue]")

    def _validate_configuration(self) -> bool:
        """Checks if all necessary paths and modes are selected."""
        if not all([self.config_path, self.model_path, self.render_mode]):
            self.call_later(
                self.log_message,
                "[bold red]Error:[/bold red] Please make all selections before running trials.",
            )
            return False
        return True

    def _setup_environment(self, wdb):
        """Initializes the gym environment based on config."""
        env_name = wdb.get_hyperparameter("environment")
        render_mode = "human" if self.render_mode == "Human Rendering" else "rgb_array"

        env = EnvironmentManager(env_name, render_mode)
        env.build_continuous()

        if self.render_mode == "Video Rendering":
            env.build_video_recorder()

        return env

    def _setup_agent(self, env, wdb):
        """Initializes the agent and loads weights."""
        algorithm = wdb.get_hyperparameter("algorithm")

        if algorithm == "PPO Continuous":
            agent = PPOAgentContinuous(env, wdb)
        elif algorithm == "SAC":
            agent = SACAgent(env, wdb)
        elif algorithm == "TD3":
            agent = TD3Agent(env, wdb)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        if "RANDOM POLICY" not in self.model_path:
            try:
                agent.load_model(agent.actor, self.model_path)
            except RuntimeError:
                self.call_later(
                    self.log_message,
                    "[bold red]Make sure the model is compatible with selected agent![/bold red]",
                )
                raise

        return agent

    def _execute_trial_loop(self, agent) -> pd.DataFrame:
        """Runs the episodes and collects data."""
        data = []

        for i in range(self.num_trials):
            reward, steps, info = agent.play(delay=self.delay_rendering)

            if isinstance(reward, (tuple, list, np.ndarray)):
                reward = float(reward[0])

            success_rate = info.get("success_rate", -1)
            data.append(
                {
                    "Trial": i,
                    "Reward": reward,
                    "Steps": steps,
                    "Success Rate": success_rate,
                }
            )

            self._log_trial_status(i, reward, success_rate)
            self._update_progress_bar()

        return pd.DataFrame(data)

    def _process_results(self, df: pd.DataFrame):
        """Calculates stats, saves CSV and generates charts."""
        self.call_later(
            self.log_message,
            f"[bold magenta]Average Reward:[/bold magenta] {df['Reward'].mean()}",
        )
        self.call_later(
            self.log_message,
            f"[bold magenta]Reward Std Dev:[/bold magenta] {df['Reward'].std()}",
        )

        success_rate = df["Success Rate"].mean()
        if success_rate != -1:
            self.call_later(
                self.log_message,
                f"[bold magenta]Avg Success Rate:[/bold magenta] {df['Success Rate'].mean()}",
            )
            msg = (
                "[bold green] Success! [/bold green]"
                if success_rate >= 0.9
                else "[bold red] Failure! [/bold red]"
            )
            self.call_later(self.log_message, msg)

        output_dir = self.project_root / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_name = self.model_path.rsplit(os.sep, maxsplit=1)[-1]

        if self.generate_chart_report:
            self.call_later(generate_distribution_plot, df, run_name, output_dir)
            self.call_later(generate_scatter_plot, df, run_name, output_dir)

        if self.generate_csv_log:
            csv_path = str(output_dir / f"{run_name}_run.csv")
            df.to_csv(csv_path)

    def _log_trial_status(self, i, reward, success_rate):
        """Helper to log single trial result to GUI."""
        self.call_later(
            self.log_message,
            f"[bold yellow]Trial {i + 1} Reward:[/bold yellow] {reward}",
        )

    def log_summary(self):
        """Thread-safe method for writing trial summary"""
        debug_output = self.query_one(RichLog)
        debug_output.clear()
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
        debug_output = self.query_one(RichLog)
        debug_output.write(text)

    def clear_log(self):
        """Thread-safe method for clearing log"""
        debug_output = self.query_one(RichLog)
        debug_output.clear()


if __name__ == "__main__":
    app = RlPlayground()
    app.run()
