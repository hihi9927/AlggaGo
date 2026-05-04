import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from alggago.env import AlggaGoEnv
from alggago.reward import reward_fn
from .config import LOG_DIR, SAVE_DIR, TRAINING_STATE_FILE


# --- Rule-based 珥덇린???⑥닔 ---
def initialize_to_rule_based(model):
    """
    紐⑤뜽???뺤콉??'洹쒖튃 湲곕컲' ?됰룞???섎룄濡?珥덇린?뷀빀?덈떎.
    - ?꾨왂: '?쇰컲 怨듦꺽'???뺣룄?곸쑝濡??좏샇?섎룄濡??ㅼ젙
    - ?뚮씪誘명꽣: 洹쒖튃 湲곕컲??媛믪쓣 洹몃?濡??곕Ⅴ?꾨줉 ?ㅽ봽?뗭쓣 0?쇰줈 ?ㅼ젙
    """
    with torch.no_grad():
        # ?좉꼍留앹쓽 留덉?留?異쒕젰 ?덉씠??action_net)瑜?媛?몄샃?덈떎.
        action_net = model.policy.action_net

        # action_net??媛以묒튂??紐⑤몢 0?쇰줈 ?ㅼ젙?섏뿬, 異쒕젰???명뼢(bias)???섑빐?쒕쭔 寃곗젙?섎룄濡??⑸땲??
        action_net.weight.data.fill_(0.0)

        # 紐⑤뜽??理쒖쥌 異쒕젰? 5媛쒖쓽 媛믪쓣 媛吏묐땲??
        # [0]: ?쇰컲 怨듦꺽 ?좏샇??(Regular Attack Preference)
        # [1]: ?덉깉 怨듦꺽 ?좏샇??(Split Shot Preference)
        # [2]: raw_index (???좏깮 ?ㅽ봽??
        # [3]: raw_angle (媛곷룄 ?ㅽ봽??
        # [4]: raw_force (???ㅽ봽??

        # 1. ?꾨왂 ?좏깮 珥덇린??
        # '?쇰컲 怨듦꺽' ?좏샇?꾨뒗 留ㅼ슦 ?믨쾶, '?덉깉 怨듦꺽' ?좏샇?꾨뒗 留ㅼ슦 ??쾶 ?ㅼ젙
        action_net.bias[0].data.fill_(10.0)  # ?쇰컲 怨듦꺽 ?좏샇
        action_net.bias[1].data.fill_(-10.0) # ?덉깉 怨듦꺽 鍮꾩꽑??

        # 2. ?뚮씪誘명꽣 ?ㅽ봽??珥덇린??
        # ?? 媛곷룄, ?섏뿉 ????섏젙媛??ㅽ봽??? 紐⑤몢 0?쇰줈 ?ㅼ젙
        action_net.bias[2].data.fill_(0.0)
        action_net.bias[3].data.fill_(0.0)
        action_net.bias[4].data.fill_(0.0)

        # 3. ?됰룞??遺꾩궛(log_std)??留ㅼ슦 ?묎쾶 留뚮뱾?? 珥덇린 ?됰룞??嫄곗쓽 ?쇱젙?섍쾶 留뚮벊?덈떎.
        if isinstance(model.policy.log_std, torch.nn.Parameter):
            model.policy.log_std.data.fill_(-20.0)

# --- 紐⑤뜽 ?뚮씪誘명꽣 ?뺤씤 ?⑥닔 ---
def print_model_parameters(model: PPO):
    print("\n==== 紐⑤뜽 珥덇린 ?뚮씪誘명꽣 ?곹깭 ?뺤씤 ====")
    params = model.get_parameters()['policy']
    for name, tensor in params.items():
        if name in ["mlp_extractor", "value_net"] and isinstance(tensor, dict):
            for sub_name, sub_tensor in tensor.items():
                if hasattr(sub_tensor, 'cpu'):
                    arr = sub_tensor.cpu().numpy()
                    print(f"policy.{name}.{sub_name}: max={np.max(arr):.6f}, min={np.min(arr):.6f}")
        elif hasattr(tensor, 'cpu'):
            arr = tensor.cpu().numpy()
            print(f"policy.{name}: max={np.max(arr):.6f}, min={np.min(arr):.6f}")
    print("=" * 31, "\n")

# --- ?섍꼍 ?앹꽦 ?ы띁 ?⑥닔 ---
def make_env_fn():
    def _init():
        env = AlggaGoEnv(reward_fn=reward_fn)
        monitored_env = Monitor(env, filename=LOG_DIR)
        return monitored_env
    return _init

# --- 湲고? ?ы띁 ?⑥닔 ---
def clean_models(model_A_path, model_B_path, best_model_paths):
    if not os.path.exists(SAVE_DIR): return
    all_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".zip")]
    to_keep_names = {os.path.basename(p) for p in [model_A_path, model_B_path] if p} | {os.path.basename(p) for p in best_model_paths}
    for fname in all_files:
        if fname in to_keep_names: continue
        try:
            file_to_remove = os.path.join(SAVE_DIR, fname)
            if os.path.exists(file_to_remove): os.remove(file_to_remove)
        except OSError as e: print(f"[WARN] ?뚯씪 ??젣 ?ㅽ뙣: {e}")


def update_best_models(current_best_models, new_model_path, reward, max_to_keep=5):
    current_best_models.append((new_model_path, reward))
    current_best_models.sort(key=lambda x: x[1], reverse=True)
    return current_best_models[:max_to_keep]


def load_training_state():
    if os.path.exists(TRAINING_STATE_FILE):
        try:
            state = np.load(TRAINING_STATE_FILE, allow_pickle=True).item()
            print(f"[INFO] ?? ?? ?? ?? ??: {state}")
            return state
        except Exception as e:
            print(f"[ERROR] ?? ?? ?? ??: {e}")
    return None


def save_training_state(state_dict):
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(TRAINING_STATE_FILE, state_dict)


def reload_with_env(model: PPO, new_env):
    """?? ?? ????? ??? ? env? ??? ?? ?? ? ?????."""
    tmp = os.path.join(SAVE_DIR, "_tmp_reload_swap_env.zip")
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(tmp)
    new_model = PPO.load(tmp, env=new_env, device=model.device)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return new_model

