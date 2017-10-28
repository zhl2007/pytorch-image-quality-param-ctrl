import copy
import glob

import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from arguments import get_args
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from usb_cam_env import *

ENV_IMG_W = 640
ENV_IMG_H = 480
"""done_reward: if env get reward >= done_reward, then env return done"""
env_done_reward = 45
"""search_done_reward: if env get reward >= search_done_reward, then ppo stop"""
search_done_reward = 48

args = get_args()

assert args.num_processes * args.num_steps % args.batch_size == 0

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.json'))
    for f in files:
        os.remove(f)


def main():
    os.environ['OMP_NUM_THREADS'] = '1'

    envs = UsbCamEnv(ENV_IMG_W, ENV_IMG_H, env_done_reward)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    actor_critic = MLPPolicy(obs_shape[0], envs.action_space)
    action_shape = envs.action_space.shape[0]

    print('+++++++++++++++++++++++++++++++++++++')
    print('obs_shape:', obs_shape)
    print('action_shape:', action_shape)
    print('+++++++++++++++++++++++++++++++++++++')

    if args.cuda:
        actor_critic.cuda()

    optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space)
    current_state = torch.zeros(args.num_processes, *obs_shape)

    def update_current_state(state):
        shape_dim0 = envs.observation_space.shape[0]
        state = torch.from_numpy(state).float()
        if args.num_stack > 1:
            current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
        current_state[:, -shape_dim0:] = state

    state = envs.reset()
    update_current_state(state)

    rollouts.states[0].copy_(current_state)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_state = current_state.cuda()
        rollouts.cuda()

    old_model = copy.deepcopy(actor_critic)

    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value, action = actor_critic.act(Variable(rollouts.states[step], volatile=True))
            cpu_actions = action.data.cpu().numpy()

            # Obser reward and next state
            state, reward, done, info = envs.step(cpu_actions)

            print('%3d  [%3d  %3d  %3d  %3d]  %3d' % (step,
                                                      int(envs.convert_2_real_action(cpu_actions)[0, 0]),
                                                      int(envs.convert_2_real_action(cpu_actions)[0, 1]),
                                                      int(envs.convert_2_real_action(cpu_actions)[0, 2]),
                                                      int(envs.convert_2_real_action(cpu_actions)[0, 3]),
                                                      reward[0]))

            if reward[0] >= search_done_reward:
                sys.exit()

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_state.dim() == 4:
                current_state *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_state *= masks

            update_current_state(state)
            rollouts.insert(step, current_state, action.data, value.data, reward, masks)

        next_value = actor_critic(Variable(rollouts.states[-1], volatile=True))[0].data

        if hasattr(actor_critic, 'obs_filter'):
            actor_critic.obs_filter.update(rollouts.states[:-1].view(-1, *obs_shape))

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        old_model.load_state_dict(actor_critic.state_dict())
        if hasattr(actor_critic, 'obs_filter'):
            old_model.obs_filter = actor_critic.obs_filter

        for _ in range(args.ppo_epoch):
            sampler = BatchSampler(SubsetRandomSampler(range(args.num_processes * args.num_steps)), args.batch_size * args.num_processes, drop_last=False)
            for indices in sampler:
                indices = torch.LongTensor(indices)
                if args.cuda:
                    indices = indices.cuda()
                states_batch = rollouts.states[:-1].view(-1, *obs_shape)[indices]
                actions_batch = rollouts.actions.view(-1, action_shape)[indices]
                return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(Variable(states_batch), Variable(actions_batch))

                _, old_action_log_probs, _ = old_model.evaluate_actions(Variable(states_batch, volatile=True), Variable(actions_batch, volatile=True))

                ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
                adv_targ = Variable(advantages.view(-1, 1)[indices])
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                value_loss = (Variable(return_batch) - values).pow(2).mean()

                optimizer.zero_grad()
                (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                optimizer.step()

        rollouts.states[0].copy_(rollouts.states[-1])

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            print("Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, j * args.num_processes * args.num_steps,
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), -dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))


if __name__ == "__main__":
    main()
