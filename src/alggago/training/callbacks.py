from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        # 실제 목표 타임스텝(= learn에 넘긴 total_timesteps)로 표시
        self.start_num = self.model.num_timesteps
        self.pbar = tqdm(total=self.total_timesteps,
                         desc="학습 진행률",
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    def _on_step(self):
        # 현재 진행 타임스텝 = (모델 누적) - (학습 시작 시점)
        done_ts = self.model.num_timesteps - self.start_num
        # pbar 위치를 직접 맞춰줌
        if self.pbar:
            self.pbar.n = min(done_ts, self.total_timesteps)
            self.pbar.refresh()
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None


def print_overall_progress(current_stage, total_stages, current_timesteps, total_timesteps):
    stage_progress = (current_stage / total_stages) * 100
    timestep_progress = (current_timesteps / total_timesteps) * 100
    print(f"\n{'='*60}\n?? ?? ???\n   ????: {current_stage}/{total_stages} ({stage_progress:.1f}%)\n   ????: {current_timesteps:,}/{total_timesteps:,} ({timestep_progress:.1f}%)\n{'='*60}")
